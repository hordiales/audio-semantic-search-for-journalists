import { defineTool } from "eve/tools";
import { z } from "zod";
import { getSegmentInfoNeon } from "../../src/db.js";

export default defineTool({
  description: "Obtiene los metadatos completos de un segmento recuperado previamente.",
  inputSchema: z.object({
    segment_id: z.number().int().min(0).describe("Identificador estable del segmento."),
    dataset_id: z.string().optional().describe("ID opcional del dataset en Neon."),
  }),
  async execute({ segment_id, dataset_id }) {
    try {
      const segment = await getSegmentInfoNeon(segment_id, dataset_id);
      if (!segment) {
        return { status: "error", error: `Segmento ${segment_id} no encontrado.` };
      }
      return { status: "success", segment };
    } catch (error) {
      return {
        status: "error",
        error: error instanceof Error ? error.message : String(error),
      };
    }
  },
});
