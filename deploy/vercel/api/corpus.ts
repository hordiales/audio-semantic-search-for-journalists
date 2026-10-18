import type { VercelRequest, VercelResponse } from "@vercel/node";
import { resolveDatasetMeta } from "../src/db.js";
import { withCors } from "../src/cors.js";

async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "GET") {
    res.status(405).json({ detail: "Method not allowed" });
    return;
  }
  try {
    const datasetId = typeof req.query.dataset_id === "string" ? req.query.dataset_id : undefined;
    const meta = await resolveDatasetMeta(datasetId);
    res.status(200).json({
      dataset_id: meta.dataset_id,
      total_segments: meta.total_segments,
      files: [],
      active_indexes: meta.active_indexes,
      dataset_version: meta.dataset_release,
    });
  } catch (error) {
    console.error("Failed to query corpus metadata from Neon:", error);
    res.status(502).json({ detail: "No fue posible consultar el corpus." });
  }
}

export default withCors(handler);
