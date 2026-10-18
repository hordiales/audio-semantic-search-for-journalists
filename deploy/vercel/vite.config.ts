import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/",
  plugins: [react()],
  build: {
    outDir: "dist-web",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    // Only used by `npm run dev:web` standalone. Inside `vercel dev` the
    // gateway routes /api/* to this service's own functions, so the proxy
    // is skipped via the VERCEL check below.
    ...(process.env.VERCEL
      ? {}
      : {
          proxy: {
            "/api": "http://localhost:3000",
            "/eve": "http://localhost:3000",
          },
        }),
  },
});
