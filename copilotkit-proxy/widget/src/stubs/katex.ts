export function renderToString(tex: string, _options?: unknown): string {
  return tex;
}

export function renderToDomTree(_tex: string, _options?: unknown): null {
  return null;
}

export const version = "0.0.0";

export default { renderToString, renderToDomTree, version };
