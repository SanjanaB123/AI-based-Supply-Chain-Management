import { apiFetch } from './api';
import type { ChatRequest, ChatResponse, AiStatusResponse } from '../types/chat';

/**
 * Send a message to the AI assistant.
 * Requires a valid Clerk Bearer token and the Clerk userId as username.
 */
export async function sendChatMessage(
  message: string,
  username: string,
  token: string | null,
): Promise<ChatResponse> {
  return apiFetch<ChatResponse>(
    '/api/chat',
    {
      method: 'POST',
      body: JSON.stringify({ message, username } satisfies ChatRequest),
    },
    token,
  );
}

/**
 * Fetch the current AI backend status.
 * This is a public status endpoint; no auth token is required.
 */
export async function fetchAiStatus(): Promise<AiStatusResponse> {
  return apiFetch<AiStatusResponse>('/api/ai-status');
}
