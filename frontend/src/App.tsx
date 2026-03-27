import { FormEvent, useEffect, useState } from "react";
import { DEFAULT_API_BASE_URL, GROUP_MEMBERS, PROJECT_TITLE } from "./config";
import {
  ChatMessage,
  DaysOfSupplyResponse,
  LeadTimeRiskResponse,
  SellThroughResponse,
  ShrinkageResponse,
  StockHealthResponse,
  StockLevelsResponse,
  StoresResponse
} from "./types";

const starterMessages: ChatMessage[] = [
  {
    id: "system-welcome",
    role: "system",
    text: "Ask about stock levels, sell-through, days of supply, stock health, lead time risk, shrinkage, or available stores.",
    payload: {
      examples: [
        "Show stock levels for S001",
        "Give me stock health for S003",
        "What is the shrinkage for S005?",
        "List available stores"
      ]
    }
  }
];

type FrontendTab = "chat" | "visualization";
type InventoryIntent =
  | "stores"
  | "stock-levels"
  | "sell-through"
  | "days-of-supply"
  | "stock-health"
  | "lead-time-risk"
  | "shrinkage";

type InventoryResponse =
  | StoresResponse
  | StockLevelsResponse
  | SellThroughResponse
  | DaysOfSupplyResponse
  | StockHealthResponse
  | LeadTimeRiskResponse
  | ShrinkageResponse;

function createId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatJson(payload: unknown) {
  return JSON.stringify(payload, null, 2);
}

function extractStoreId(message: string) {
  const match = message.match(/\bS\d{3}\b/i);
  return match ? match[0].toUpperCase() : null;
}

function detectIntent(message: string): InventoryIntent | null {
  const normalized = message.toLowerCase();

  if (
    normalized.includes("available stores") ||
    normalized.includes("list stores") ||
    normalized.includes("show stores") ||
    normalized.includes("what stores")
  ) {
    return "stores";
  }

  if (normalized.includes("sell through") || normalized.includes("sell-through")) {
    return "sell-through";
  }

  if (normalized.includes("days of supply") || /\bdos\b/.test(normalized)) {
    return "days-of-supply";
  }

  if (normalized.includes("stock health") || normalized.includes("health breakdown")) {
    return "stock-health";
  }

  if (normalized.includes("lead time") || normalized.includes("risk")) {
    return "lead-time-risk";
  }

  if (normalized.includes("shrinkage") || normalized.includes("loss")) {
    return "shrinkage";
  }

  if (
    normalized.includes("stock level") ||
    normalized.includes("current stock") ||
    normalized.includes("inventory") ||
    normalized.includes("stock")
  ) {
    return "stock-levels";
  }

  return null;
}

function getUnsupportedQueryMessage() {
  return {
    text: "I can query the inventory backend for stock levels, sell-through, days of supply, stock health, lead time risk, shrinkage, or available stores.",
    payload: {
      examples: [
        "Show stock levels for S001",
        "What is the sell-through for S002?",
        "Give me lead time risk for S004"
      ]
    }
  };
}

function summarizeResponse(
  intent: Exclude<InventoryIntent, "stores">,
  response: Exclude<InventoryResponse, StoresResponse>
) {
  switch (intent) {
    case "stock-levels": {
      const data = response as StockLevelsResponse;
      const lowestStock = data.products
        .slice(0, 3)
        .map((product) => `${product.product_id} (${product.current_stock})`)
        .join(", ");

      return `Store ${data.store} has ${data.total_stock.toLocaleString()} total units. Health summary: ${data.summary.critical} critical, ${data.summary.low} low, ${data.summary.healthy} healthy. Lowest-stock products: ${lowestStock}.`;
    }
    case "sell-through": {
      const data = response as SellThroughResponse;
      const leaders = data.products
        .slice(0, 3)
        .map(
          (product) =>
            `${product.product_id} (${product.sell_through_rate.toFixed(2)}%)`
        )
        .join(", ");

      return `Top sell-through products for ${data.store}: ${leaders}.`;
    }
    case "days-of-supply": {
      const data = response as DaysOfSupplyResponse;
      const lowestCoverage = data.products
        .slice(0, 3)
        .map(
          (product) => `${product.product_id} (${product.days_of_supply.toFixed(2)} days)`
        )
        .join(", ");

      return `Store ${data.store} uses thresholds of under ${data.thresholds.critical_below} days for critical and under ${data.thresholds.low_below} days for low stock. Lowest days-of-supply products: ${lowestCoverage}.`;
    }
    case "stock-health": {
      const data = response as StockHealthResponse;
      const breakdown = data.breakdown
        .map((entry) => `${entry.status} ${entry.count} (${entry.percentage}%)`)
        .join(", ");

      return `Stock health for ${data.store} across ${data.total_products} products: ${breakdown}.`;
    }
    case "lead-time-risk": {
      const data = response as LeadTimeRiskResponse;
      const highestRiskProducts = [...data.products]
        .sort((left, right) => left.days_of_supply - right.days_of_supply)
        .slice(0, 3)
        .map(
          (product) =>
            `${product.product_id} (${product.days_of_supply.toFixed(2)} days of supply, ${product.lead_time_days} lead-time days)`
        )
        .join(", ");

      return `Products with the thinnest coverage in ${data.store}: ${highestRiskProducts}.`;
    }
    case "shrinkage": {
      const data = response as ShrinkageResponse;
      const largestLosses = data.products
        .slice(0, 3)
        .map((product) => `${product.product_id} (${product.shrinkage})`)
        .join(", ");

      return `Total shrinkage for ${data.store} is ${data.total_shrinkage.toLocaleString()} units. Largest product-level losses: ${largestLosses}.`;
    }
  }
}

async function fetchJson<T>(url: string) {
  const response = await fetch(url, {
    method: "GET",
    headers: {
      Accept: "application/json"
    }
  });

  const responseText = await response.text();
  const payload = responseText ? (JSON.parse(responseText) as T) : null;

  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload !== null && "detail" in payload
        ? String(payload.detail)
        : `Request failed with status ${response.status}: ${response.statusText}`;

    throw new Error(detail);
  }

  if (payload === null) {
    throw new Error("The backend returned an empty response body.");
  }

  return payload;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<FrontendTab>("chat");
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>(starterMessages);
  const [stores, setStores] = useState<string[]>([]);
  const [selectedStore, setSelectedStore] = useState("");
  const [isLoadingStores, setIsLoadingStores] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadStores() {
      try {
        const data = await fetchJson<StoresResponse>(
          `${DEFAULT_API_BASE_URL}/api/stores`
        );

        setStores(data.stores);
        setSelectedStore(data.stores[0] ?? "");
      } catch (loadError) {
        const message =
          loadError instanceof Error
            ? loadError.message
            : "Failed to load stores from the backend.";

        setError(message);
        setMessages((currentMessages) => [
          ...currentMessages,
          {
            id: createId(),
            role: "assistant",
            text: "The backend store list could not be loaded.",
            payload: {
              error: message,
              baseUrl: DEFAULT_API_BASE_URL
            }
          }
        ]);
      } finally {
        setIsLoadingStores(false);
      }
    }

    void loadStores();
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedMessage = draft.trim();
    if (!trimmedMessage || isSending) {
      return;
    }

    const intent = detectIntent(trimmedMessage);
    const messageStore = extractStoreId(trimmedMessage);
    const resolvedStore = messageStore ?? selectedStore;

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      text: trimmedMessage,
      payload: {
        intent: intent ?? "unsupported",
        store: resolvedStore || null
      }
    };

    setMessages((currentMessages) => [...currentMessages, userMessage]);
    setDraft("");
    setError("");
    setIsSending(true);

    try {
      if (intent === "stores") {
        const storesPayload: StoresResponse = { stores };
        setMessages((currentMessages) => [
          ...currentMessages,
          {
            id: createId(),
            role: "assistant",
            text:
              stores.length > 0
                ? `Available stores: ${stores.join(", ")}.`
                : "No stores were loaded from the backend.",
            payload: storesPayload
          }
        ]);
        return;
      }

      if (!intent) {
        const unsupported = getUnsupportedQueryMessage();
        setMessages((currentMessages) => [
          ...currentMessages,
          {
            id: createId(),
            role: "assistant",
            text: unsupported.text,
            payload: unsupported.payload
          }
        ]);
        return;
      }

      if (!resolvedStore) {
        throw new Error("No store is selected. Wait for the store list to load, or mention a store like S001.");
      }

      const requestUrl = new URL(
        `${DEFAULT_API_BASE_URL}/api/${intent}`
      );
      requestUrl.searchParams.set("store", resolvedStore);

      const responsePayload = await fetchJson<
        Exclude<InventoryResponse, StoresResponse>
      >(requestUrl.toString());

      const assistantMessage: ChatMessage = {
        id: createId(),
        role: "assistant",
        text: summarizeResponse(intent, responsePayload),
        payload: responsePayload
      };

      setMessages((currentMessages) => [...currentMessages, assistantMessage]);
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "The backend request failed for an unknown reason.";

      setError(message);
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: createId(),
          role: "assistant",
          text: "The backend request failed.",
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
              <p>
                Connected to <code>{DEFAULT_API_BASE_URL}</code>
              </p>
            </div>

            <div className="connection-note">
              <p>
                Supported queries: stock levels, sell-through, days of supply,
                stock health, lead time risk, shrinkage, and available stores.
              </p>
            </div>

            <label className="field-group">
              <span>Default store</span>
              <select
                value={selectedStore}
                onChange={(event) => setSelectedStore(event.target.value)}
                disabled={isLoadingStores || stores.length === 0}
              >
                {stores.length === 0 ? (
                  <option value="">
                    {isLoadingStores ? "Loading stores..." : "No stores available"}
                  </option>
                ) : null}
                {stores.map((store) => (
                  <option key={store} value={store}>
                    {store}
                  </option>
                ))}
              </select>
            </label>

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
                  placeholder="Try: show stock health for S001"
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
