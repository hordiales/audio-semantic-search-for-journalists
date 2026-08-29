import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import cssInjectedByJsPlugin from "vite-plugin-css-injected-by-js";

/** Build one compressed, self-contained ESM file for third-party websites. */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const runtimeUrl = env.VITE_COPILOT_RUNTIME_URL || "http://localhost:8080/api/copilotkit";

  return {
    plugins: [react(), cssInjectedByJsPlugin()],
    define: {
      "process.env.NODE_ENV": JSON.stringify("production"),
      __RUNTIME_URL__: JSON.stringify(runtimeUrl),
    },
    resolve: {
      alias: {
        "@segment/analytics-node": fileURLToPath(
          new URL("./src/stubs/segment-analytics-node.ts", import.meta.url),
        ),
        "katex/dist/katex.min.css": fileURLToPath(
          new URL("./src/stubs/katex-css.ts", import.meta.url),
        ),
        katex: fileURLToPath(new URL("./src/stubs/katex.ts", import.meta.url)),
      },
    },
    build: {
      outDir: "dist-widget",
      emptyOutDir: true,
      cssCodeSplit: false,
      rollupOptions: {
        input: {
          widget: fileURLToPath(new URL("./widget.html", import.meta.url)),
        },
        output: {
          inlineDynamicImports: true,
          entryFileNames: "audio-search-widget.js",
          assetFileNames: "[name].[ext]",
        },
      },
    },
  };
});
