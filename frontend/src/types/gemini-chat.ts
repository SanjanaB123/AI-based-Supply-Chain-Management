export interface GeminiMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: number;
  status?: 'sent' | 'error';
  functionCalls?: string[];
}

export interface GeminiChatRequest {
  message: string;
}

export interface GeminiChatResponse {
  response: string;
  function_calls_made: string[];
}

export interface GeminiHistoryResponse {
  messages: { role: string; text: string }[];
}

export interface GeminiStatusResponse {
  status: string;
  gemini_enabled: boolean;
  model: string;
}

export interface Conversation {
  convo_id: string;
  title: string;
  active: boolean;
  updated_at: string;
}

export interface ContactFormData {
  name: string;
  email: string;
  subject: string;
  message: string;
}

export interface ContactResponse {
  success: boolean;
  message: string;
}
