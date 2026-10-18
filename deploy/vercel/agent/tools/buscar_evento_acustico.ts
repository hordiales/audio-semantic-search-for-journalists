import { defineTool } from "eve/tools";
import { z } from "zod";
import { searchAudioNeon, serializeAgentResults } from "../../src/db.js";

export default defineTool({
  description:
    "Busca eventos acústicos a partir de una descripción textual (aplausos, música, gritos, risas u otros sonidos que pueden no estar en la transcripción).",
  inputSchema: z.object({
    query: z.string().min(1).describe("Descripción textual del evento acústico."),
    k: z.number().int().min(1).max(20).default(5).describe("Cantidad máxima de segmentos a devolver (1 a 20)."),
    dataset_id: z.string().optional().describe("ID opcional del dataset en Neon."),
  }),
  async execute({ query, k, dataset_id }) {
    if (!query.trim()) {
      return { status: "error", error: "La consulta no puede estar vacía.", results: [] };
    }
    try {
      const { results, translatedQuery } = await searchAudioNeon(query, k, {
        datasetId: dataset_id,
      });
      return {
        status: "success",
        modality: "audio",
        translated_query: translatedQuery,
        results: serializeAgentResults(
          results,
          "audio",
          "Índice de audio (CLAP)",
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
