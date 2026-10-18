import type {
  AgentChatMessage,
  DatasetsResponse,
  SearchRequest,
  SearchResponse,
} from "./types";

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export async function fetchDatasets(signal?: AbortSignal): Promise<DatasetsResponse> {
  const response = await fetch(`${apiBaseUrl}/api/datasets`, { signal });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || `No se pudieron cargar los datasets (${response.status}).`);
  }
  return (await response.json()) as DatasetsResponse;
}

export async function search(
  request: SearchRequest,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const response = await fetch(`${apiBaseUrl}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || `La búsqueda falló (${response.status}).`);
  }
  return (await response.json()) as SearchResponse;
}

export async function sendAgentChatMessage(
  messages: Array<Pick<AgentChatMessage, "role" | "content">>,
  datasetId?: string,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch(`${apiBaseUrl}/api/agent-chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, dataset_id: datasetId }),
    signal,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || `El agente no pudo responder (${response.status}).`);
  }
  const data = (await response.json()) as { reply?: string };
  return data.reply || "No obtuve respuesta del agente.";
}

export async function submitAgentFeedback(payload: {
  messageId: string;
  feedback: "thumbsUp" | "thumbsDown";
  question: string;
  answer: string;
}): Promise<void> {
  await fetch(`${apiBaseUrl}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
