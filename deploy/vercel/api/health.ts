import type { VercelRequest, VercelResponse } from "@vercel/node";
import { withCors } from "../src/cors.js";

async function handler(_req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ status: "ok", target: "vercel-neon-eve" });
}

export default withCors(handler);
