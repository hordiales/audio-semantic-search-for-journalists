import { defineTool } from "eve/tools";
import { z } from "zod";
import { searchSemanticNeon, serializeAgentResults } from "../../src/db.js";

export default defineTool({
  description:
    "Busca texto semánticamente en las transcripciones del corpus de audio periodístico almacenado en Neon pgvector.",
  inputSchema: z.object({
    query: z.string().min(1).describe("Consulta del periodista en lenguaje natural."),
    k: z.number().int().min(1).max(20).default(5).describe("Cantidad máxima de segmentos a devolver (1 a 20)."),
    dataset_id: z.string().optional().describe("ID opcional del dataset en Neon."),
  }),
  async execute({ query, k, dataset_id }) {
    if (!query.trim()) {
      return { status: "error", error: "La consulta no puede estar vacía.", results: [] };
    }
    try {
      const { results } = await searchSemanticNeon(query, k, dataset_id);
      return {
        status: "success",
        modality: "text",
        results: serializeAgentResults(
          results,
          "text",
          "Índice de texto (transcripciones)",
        ),
      };
    } catch (error) {
      return {
        status: "error",
        error: error instanceof Error ? error.message : String(error),
        results: [],
      };
    }
  },
});
