import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // GitHub Pages serves this app below /direct-search/, not at the origin root.
  // Relative asset URLs also keep standalone static hosting portable.
  base: "./",
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8080",
    },
  },
});
