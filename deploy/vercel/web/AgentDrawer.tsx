import { type FormEvent, useRef, useState } from "react";
import { sendAgentChatMessage, submitAgentFeedback } from "./api";
import type { UiCopy } from "./i18n";
import type { AgentChatMessage } from "./types";

let nextMessageId = 0;
const newMessageId = () => `msg-${Date.now()}-${nextMessageId++}`;

export function AgentDrawer({
  copy,
  datasetId,
  onClose,
}: {
  copy: UiCopy;
  datasetId?: string;
  onClose: () => void;
}) {
  const [messages, setMessages] = useState<AgentChatMessage[]>([
    {
      id: newMessageId(),
      role: "assistant",
      content: copy.agentWelcome,
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastUserQuestion = useRef("");
  const listRef = useRef<HTMLDivElement>(null);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || sending) return;
    lastUserQuestion.current = text;
    setInput("");
    setError(null);
    setSending(true);

    const userMessage: AgentChatMessage = { id: newMessageId(), role: "user", content: text };
    const history = [...messages, userMessage];
    setMessages(history);

    try {
      const reply = await sendAgentChatMessage(
        history
          .filter((message) => message.role === "user" || message.role === "assistant")
          .map(({ role, content }) => ({ role, content })),
        datasetId,
      );
      setMessages((current) => [
        ...current,
        {
          id: newMessageId(),
          role: "assistant",
          content: reply,
          question: lastUserQuestion.current,
        },
      ]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSending(false);
      requestAnimationFrame(() => {
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
      });
    }
  };

  const sendFeedback = (message: AgentChatMessage, feedback: "thumbsUp" | "thumbsDown") => {
    void submitAgentFeedback({
      messageId: message.id,
      feedback,
      question: message.question ?? "",
      answer: message.content,
    }).catch(() => undefined);
    setMessages((current) =>
      current.map((item) => (item.id === message.id ? { ...item, feedback } : item)),
    );
  };

  return (
    <aside className="agent-drawer" aria-label={copy.agentTitle}>
      <header>
        <strong>{copy.agentTitle}</strong>
        <button onClick={onClose}>{copy.closeAgent}</button>
      </header>
      <div className="agent-messages" ref={listRef}>
        {messages.map((message) => (
          <div key={message.id}>
            <div
              className={`agent-msg ${message.role === "user" ? "agent-msg-user" : "agent-msg-assistant"}`}
            >
              {message.content}
            </div>
            {message.role === "assistant" && message.question && (
              <div className="agent-feedback">
                <button
                  className={message.feedback === "thumbsUp" ? "selected" : ""}
                  onClick={() => sendFeedback(message, "thumbsUp")}
                >
                  👍 {copy.helpful}
                </button>
                <button
                  className={message.feedback === "thumbsDown" ? "selected" : ""}
                  onClick={() => sendFeedback(message, "thumbsDown")}
                >
                  👎 {copy.notHelpful}
                </button>
              </div>
            )}
          </div>
        ))}
        {sending && <div className="agent-msg agent-msg-assistant">{copy.agentThinking}</div>}
        {error && <p className="notice error">{error}</p>}
      </div>
      <form className="agent-composer" onSubmit={(event) => void onSubmit(event)}>
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={copy.agentPlaceholder}
          aria-label={copy.agentPlaceholder}
        />
        <button className="primary" type="submit" disabled={sending || !input.trim()}>
          {copy.agentSend}
        </button>
      </form>
    </aside>
  );
}
