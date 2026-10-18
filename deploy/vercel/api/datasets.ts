import type { VercelRequest, VercelResponse } from "@vercel/node";
import { listDatasets } from "../src/db.js";
import { withCors } from "../src/cors.js";

async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "GET") {
    res.status(405).json({ detail: "Method not allowed" });
    return;
  }
  try {
    const datasets = await listDatasets();
    res.status(200).json({ datasets });
  } catch (error) {
    console.error("Failed to list datasets from Neon:", error);
    res.status(502).json({ detail: "No fue posible listar los datasets." });
  }
}

export default withCors(handler);
