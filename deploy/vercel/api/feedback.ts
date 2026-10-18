import type { VercelRequest, VercelResponse } from "@vercel/node";
import { insertFeedbackNeon } from "../src/db.js";
import { withCors } from "../src/cors.js";

async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ detail: "Method not allowed" });
    return;
  }
  const body = req.body as Record<string, unknown> | undefined;
  const messageId = body?.messageId;
  const feedback = body?.feedback;
  const question = body?.question;
  const answer = body?.answer;

  if (
    typeof messageId !== "string" ||
    (feedback !== "thumbsUp" && feedback !== "thumbsDown") ||
    typeof question !== "string" ||
    typeof answer !== "string" ||
    question.length > 10_000 ||
    answer.length > 10_000
  ) {
    res.status(400).json({ detail: "Feedback inválido." });
    return;
  }

  try {
    await insertFeedbackNeon({ messageId, feedback, question, answer });
    res.status(200).json({ ok: true });
  } catch (error) {
    console.error("Failed to record feedback in Neon:", error);
    res.status(502).json({ detail: "No se pudo registrar el feedback." });
  }
}

export default withCors(handler);
