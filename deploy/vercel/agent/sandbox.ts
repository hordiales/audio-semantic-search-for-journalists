import { defineSandbox } from "eve/sandbox";
import { JustBashSandbox } from "eve/sandbox/just-bash";

// This agent's tools only query Neon Postgres, so a pure-JavaScript sandbox is
// enough and keeps `eve dev`/`eve build` working without Docker, microsandbox,
// or a provisioned Vercel Sandbox.
export const environment = JustBashSandbox.environment();
export default defineSandbox(() => environment.open());
