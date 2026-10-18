import { defineTool } from "eve/tools";
import { z } from "zod";
import { searchYamnetNeon, serializeAgentResults } from "../../src/db.js";

export default defineTool({
  description:
    "Busca segmentos por clases acústicas AudioSet detectadas por YAMNet (como Applause, Speech o Music). Pondera cobertura de términos y score del clasificador.",
  inputSchema: z.object({
    query: z.string().min(1).describe("Evento o combinación de eventos acústicos a buscar."),
    k: z.number().int().min(1).max(20).default(5).describe("Cantidad máxima de segmentos a devolver (1 a 20)."),
    dataset_id: z.string().optional().describe("ID opcional del dataset en Neon."),
  }),
  async execute({ query, k, dataset_id }) {
    if (!query.trim()) {
      return { status: "error", error: "La consulta no puede estar vacía.", results: [] };
    }
    try {
      const { results, translatedQuery } = await searchYamnetNeon(query, k, {
        datasetId: dataset_id,
      });
      return {
        status: "success",
        modality: "yamnet",
        translated_query: translatedQuery,
        results: serializeAgentResults(
          results,
          "yamnet",
          "Clases de audio (YAMNet/AudioSet)",
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
