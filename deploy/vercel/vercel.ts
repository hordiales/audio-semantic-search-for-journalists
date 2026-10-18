import { withEve } from "eve/vercel";

export default await withEve({
  services: {
    // Vite SPA (web/). Vercel does not build api/ inside a service, so the
    // API is a separate Express-backed service below.
    web: {
      framework: "vite",
      root: ".",
      buildCommand: "npm run build:web",
      outputDirectory: "dist-web",
    },
    // Direct-search + agent-chat HTTP API. server.ts mounts the api/ handler
    // functions on /api/* and is bundled as one Vercel Function.
    api: {
      framework: "express",
      root: ".",
      entrypoint: "server.ts",
      // Override the inherited `npm run build` (= eve build): the Express
      // builder bundles server.ts itself; typecheck is the only useful step.
      buildCommand: "npm run typecheck",
    },
  },
  routes: [
    // The generated eve agent service mounts at /eve/v1/* automatically.
    { src: "^/api(/.*)?$", destination: { type: "service", service: "api" } },
    { src: "^(.*)$", destination: { type: "service", service: "web" } },
  ],
});
