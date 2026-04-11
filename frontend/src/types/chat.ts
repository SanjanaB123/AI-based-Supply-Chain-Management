// ── Chat message types ────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant' | 'system';
export type MessageStatus = 'sent' | 'error';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  status?: MessageStatus;
}

// ── API request / response shapes ─────────────────────────────────────────────

export interface ChatRequest {
  message: string;
  username: string;
}

export interface ChatResponse {
  response: string;
  agent: string;
}

export interface AiStatusResponse {
  status: string;
  ai_enabled: boolean;
  model_loaded: boolean;
  mcp_server: string;
}
