/**
 * CORS support for the direct-search API.
 *
 * The Vercel frontend is same-origin and needs no CORS, but other frontends
 * (e.g. the GitHub Pages `frontend-direct-search` build pointed at this
 * deployment via VITE_API_BASE_URL) require `Access-Control-Allow-Origin`.
 * Origins come from the `ALLOWED_ORIGINS` env var (comma-separated, or `*`).
 * When the var is unset no CORS headers are emitted at all.
 */

import type { VercelRequest, VercelResponse } from "@vercel/node";

type Handler = (req: VercelRequest, res: VercelResponse) => unknown | Promise<unknown>;

export function resolveAllowedOrigin(
  requestOrigin: string | undefined,
  configured: string | undefined = process.env.ALLOWED_ORIGINS,
): string | undefined {
  const value = configured?.trim();
  if (!value || !requestOrigin) return undefined;
  const allowed = value
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);
  if (allowed.includes("*")) return "*";
  return allowed.includes(requestOrigin) ? requestOrigin : undefined;
}

export function withCors(handler: Handler): Handler {
  return async (req, res) => {
    const origin = resolveAllowedOrigin(req.headers.origin);
    if (origin) {
      res.setHeader("Access-Control-Allow-Origin", origin);
      res.setHeader("Vary", "Origin");
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    }
    if (req.method === "OPTIONS") {
      res.status(origin ? 204 : 405).end();
      return;
    }
    return handler(req, res);
  };
}
