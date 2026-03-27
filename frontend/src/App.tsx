import { FormEvent, useState } from "react";
import { DEFAULT_FASTAPI_URL, GROUP_MEMBERS, PROJECT_TITLE } from "./config";
import { ChatMessage } from "./types";

const starterMessages: ChatMessage[] = [
  {
    id: "system-welcome",
    role: "system",
    text: "Connect this UI to your FastAPI route and start sending JSON chat requests.",
    payload: {
      requestShape: {
        message: "Your question here"
      },
      note: "Update the endpoint URL if your FastAPI server uses a different route."
    }
  }
];

function createId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatJson(payload: unknown) {
  return JSON.stringify(payload, null, 2);
}

type FrontendTab = "chat" | "visualization";

export default function App() {
  const [activeTab, setActiveTab] = useState<FrontendTab>("chat");
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>(starterMessages);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedMessage = draft.trim();
    if (!trimmedMessage || isSending) {
      return;
    }

    const requestPayload = { message: trimmedMessage };
    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      text: trimmedMessage,
      payload: requestPayload
    };

    setMessages((currentMessages) => [...currentMessages, userMessage]);
    setDraft("");
    setError("");
    setIsSending(true);

    try {
      const response = await fetch(DEFAULT_FASTAPI_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json"
        },
        body: JSON.stringify(requestPayload)
      });

      const responseText = await response.text();
      let responsePayload: unknown = null;

      try {
        responsePayload = responseText ? JSON.parse(responseText) : null;
      } catch {
        responsePayload = { raw: responseText };
      }

      if (!response.ok) {
        throw new Error(
          `Request failed with status ${response.status}: ${response.statusText}`
        );
      }

      const assistantText =
        typeof responsePayload === "object" && responsePayload !== null
          ? "FastAPI responded with JSON."
          : String(responsePayload ?? "No response body");

      const assistantMessage: ChatMessage = {
        id: createId(),
        role: "assistant",
        text: assistantText,
        payload: responsePayload
      };

      setMessages((currentMessages) => [...currentMessages, assistantMessage]);
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "The request failed for an unknown reason.";

      setError(message);
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: createId(),
          role: "assistant",
          text: "The FastAPI request failed.",
          payload: {
            error: message
          }
        }
      ]);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="page-shell">
      <div className="background-orb background-orb-left" />
      <div className="background-orb background-orb-right" />

      <section className="hero-card">
        <p className="eyebrow">Group 7</p>
        <h1>{PROJECT_TITLE}</h1>
        <p className="project-credit">
          Project by <span>{GROUP_MEMBERS.join(", ")}</span>
        </p>
        <p className="hero-copy">
          Accurate demand estimation to prevent stockouts, reduce
          overstocking, and improve operational efficiency.
        </p>
      </section>

      <section className="chat-layout">
        <div className="tab-bar" role="tablist" aria-label="Frontend sections">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "chat"}
            className={`tab-button ${activeTab === "chat" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("chat")}
          >
            Chat Interface
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "visualization"}
            className={`tab-button ${activeTab === "visualization" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("visualization")}
          >
            Visualization
          </button>
        </div>

        {activeTab === "chat" ? (
          <div className="chat-panel">
            <div className="panel-header">
              <h2>Chat Interface</h2>
            </div>

            {/* <div className="json-preview">
              <p>Outgoing JSON</p>
              <pre>{formatJson({ message: draft || "Your message" })}</pre>
            </div> */}

            {error ? <p className="error-banner">{error}</p> : null}

            <div className="messages">
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={`message-card message-${message.role}`}
                >
                  <div className="message-meta">
                    <span>{message.role}</span>
                  </div>
                  <p>{message.text}</p>
                  {message.payload !== undefined ? (
                    <pre>{formatJson(message.payload)}</pre>
                  ) : null}
                </article>
              ))}
            </div>

            <form className="chat-form" onSubmit={handleSubmit}>
              <label className="field-group">
                <span>Message</span>
                <textarea
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="Ask for a demand forecast, inventory risk, or model metadata..."
                  rows={4}
                />
              </label>

              <button type="submit" disabled={isSending || !draft.trim()}>
                {isSending ? "Sending..." : "Send"}
              </button>
            </form>
          </div>
        ) : (
          <section
            className="chat-panel visualization-panel"
            role="tabpanel"
            aria-label="Visualization"
          />
        )}
      </section>
    </main>
  );
}
