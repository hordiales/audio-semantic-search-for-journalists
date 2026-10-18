import { defineAgent } from "eve";

export default defineAgent({
  model: `openai/${process.env.OPENAI_MODEL?.trim() || "gpt-4o-mini"}`,
});
