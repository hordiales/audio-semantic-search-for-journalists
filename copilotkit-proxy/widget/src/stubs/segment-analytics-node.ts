/** Browser stub for CopilotKit's server-only Segment telemetry dependency. */
export class Analytics {
  constructor(_options?: unknown) {}

  track(..._args: unknown[]) {}

  identify(..._args: unknown[]) {}

  flush(..._args: unknown[]) {
    return Promise.resolve();
  }

  closeAndFlush(..._args: unknown[]) {
    return Promise.resolve();
  }
}
