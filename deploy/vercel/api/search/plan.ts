import type { VercelRequest, VercelResponse } from "@vercel/node";
import {
  effectivePlan,
  parseSearchRequest,
  rewriteSearchPlanWithLLM,
} from "../../src/direct-search.js";
import { withCors } from "../../src/cors.js";

async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ detail: "Method not allowed" });
    return;
  }
  try {
    const requestedIndexes =
      req.body && typeof req.body === "object"
        ? (req.body as Record<string, unknown>).indexes
        : undefined;
    const parsed = parseSearchRequest({
      ...req.body,
      indexes: requestedIndexes,
      k: 10,
    });
    const plan = await rewriteSearchPlanWithLLM(parsed);
    res.status(200).json(effectivePlan({ ...parsed, plan }));
  } catch (error) {
    res.status(400).json({
      detail: error instanceof Error ? error.message : "Solicitud inválida.",
    });
  }
}

export default withCors(handler);
