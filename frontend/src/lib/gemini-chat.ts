import { apiFetch } from './api';
import type {
  GeminiChatResponse,
  GeminiHistoryResponse,
  GeminiStatusResponse,
  Conversation,
} from '../types/gemini-chat';

export async function sendGeminiMessage(
  message: string,
  token: string | null,
): Promise<GeminiChatResponse> {
  return apiFetch<GeminiChatResponse>(
    '/api/gemini-chat',
    {
      method: 'POST',
      body: JSON.stringify({ message }),
    },
    token,
  );
}

export async function fetchGeminiHistory(
  token: string | null,
): Promise<GeminiHistoryResponse> {
  return apiFetch<GeminiHistoryResponse>('/api/gemini-chat/history', {}, token);
}

export async function clearGeminiHistory(
  token: string | null,
): Promise<{ success: boolean }> {
  return apiFetch<{ success: boolean }>(
    '/api/gemini-chat/history',
    { method: 'DELETE' },
    token,
  );
}

export async function fetchConversations(
  token: string | null,
): Promise<{ conversations: Conversation[] }> {
  return apiFetch<{ conversations: Conversation[] }>(
    '/api/gemini-chat/conversations',
    {},
    token,
  );
}

export async function activateConversation(
  convoId: string,
  token: string | null,
): Promise<{ success: boolean }> {
  return apiFetch<{ success: boolean }>(
    `/api/gemini-chat/conversations/${convoId}/activate`,
    { method: 'POST' },
    token,
  );
}

export async function fetchGeminiStatus(): Promise<GeminiStatusResponse> {
  return apiFetch<GeminiStatusResponse>('/api/gemini-status');
}
