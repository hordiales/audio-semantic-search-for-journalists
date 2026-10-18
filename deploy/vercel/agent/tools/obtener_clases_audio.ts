import { defineTool } from "eve/tools";
import { z } from "zod";
import { getAudioClassesNeon } from "../../src/db.js";

export default defineTool({
  description:
    "Obtiene las clases de AudioSet detectadas por YAMNet para un segmento. Sus scores son probabilidades del clasificador, no similitud vectorial.",
  inputSchema: z.object({
    segment_id: z.number().int().min(0).describe("Identificador estable del segmento."),
    dataset_id: z.string().optional().describe("ID opcional del dataset en Neon."),
  }),
  async execute({ segment_id, dataset_id }) {
    try {
      const classes = await getAudioClassesNeon(segment_id, dataset_id);
      if (classes === null) {
        return { status: "error", error: `Segmento ${segment_id} no encontrado.` };
      }
      if (classes.length === 0) {
        return {
          status: "not_available",
          error:
            "El dataset no contiene clases YAMNet para este segmento. Reprocésalo habilitando yamnet en config/embeddings.toml.",
          classes: [],
        };
      }
      return { status: "success", classifier: "yamnet", classes };
    } catch (error) {
      return {
        status: "error",
        error: error instanceof Error ? error.message : String(error),
      };
    }
  },
});
