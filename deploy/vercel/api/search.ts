import type { VercelRequest, VercelResponse } from "@vercel/node";
import { executeDirectSearchNeon, parseSearchRequest } from "../src/direct-search.js";
import { withCors } from "../src/cors.js";

async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ detail: "Method not allowed" });
    return;
  }
  try {
    const parsed = parseSearchRequest(req.body);
    const result = await executeDirectSearchNeon(parsed);
    res.status(200).json(result);
  } catch (error) {
    res.status(400).json({
      detail: error instanceof Error ? error.message : "Solicitud inválida.",
    });
  }
}

export default withCors(handler);
