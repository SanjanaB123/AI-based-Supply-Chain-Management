import { useState, useEffect, useCallback, useRef } from 'react';
import { useCurrentUser } from './useCurrentUser';
import {
  sendGeminiMessage,
  fetchGeminiHistory,
  clearGeminiHistory,
  fetchConversations,
  activateConversation,
} from '../lib/gemini-chat';
import type { GeminiMessage, Conversation } from '../types/gemini-chat';

export interface UseGeminiChatReturn {
  messages: GeminiMessage[];
  conversations: Conversation[];
  isSending: boolean;
  isLoading: boolean;
  error: string | null;
  sendMessage: (content: string) => Promise<void>;
  newChat: () => Promise<void>;
  switchConversation: (convoId: string) => Promise<void>;
}

export function useGeminiChat(): UseGeminiChatReturn {
  const { userId, getToken } = useCurrentUser();

  const [messages, setMessages] = useState<GeminiMessage[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadedForUser = useRef<string | null>(null);

  const loadConversations = useCallback(async () => {
    try {
      const token = await getToken();
      const data = await fetchConversations(token);
      setConversations(data.conversations);
    } catch {
      // ignore
    }
  }, [getToken]);

  const loadHistory = useCallback(async () => {
    try {
      const token = await getToken();
      const data = await fetchGeminiHistory(token);
      const loaded: GeminiMessage[] = data.messages.map((m, i) => ({
        id: `hist-${i}`,
        role: (m.role === 'model' || m.role === 'assistant') ? 'assistant' as const : 'user' as const,
        content: (m as Record<string, string>).text || (m as Record<string, string>).content || '',
        createdAt: Date.now() - (data.messages.length - i) * 1000,
        status: 'sent' as const,
      }));
      setMessages(loaded);
    } catch {
      // start fresh
    }
  }, [getToken]);

  // Load on mount
  useEffect(() => {
    if (!userId) return;
    if (loadedForUser.current === userId) return;
    loadedForUser.current = userId;

    let cancelled = false;
    (async () => {
      try {
        await Promise.all([loadHistory(), loadConversations()]);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [userId, loadHistory, loadConversations]);

  // Fallback timeout
  useEffect(() => {
    const t = setTimeout(() => { if (isLoading) setIsLoading(false); }, 3000);
    return () => clearTimeout(t);
  }, [isLoading]);

  const sendMessage = useCallback(
    async (content: string) => {
      const trimmed = content.trim();
      if (!trimmed || isSending || !userId) return;

      const userMsg: GeminiMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: trimmed,
        createdAt: Date.now(),
        status: 'sent',
      };

      setMessages((prev) => [...prev, userMsg]);
      setIsSending(true);
      setError(null);

      try {
        const token = await getToken();
        const result = await sendGeminiMessage(trimmed, token);

        const assistantMsg: GeminiMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.response,
          createdAt: Date.now(),
          status: 'sent',
          functionCalls: result.function_calls_made,
        };

        setMessages((prev) => [...prev, assistantMsg]);
        // Refresh conversation list (title may have updated)
        await loadConversations();
      } catch {
        setError('Failed to get a response. Please try again.');
      } finally {
        setIsSending(false);
      }
    },
    [isSending, userId, getToken, loadConversations],
  );

  const newChat = useCallback(async () => {
    try {
      const token = await getToken();
      await clearGeminiHistory(token);
      setMessages([]);
      setError(null);
      await loadConversations();
    } catch {
      // ignore
    }
  }, [getToken, loadConversations]);

  const switchConversation = useCallback(async (convoId: string) => {
    try {
      const token = await getToken();
      await activateConversation(convoId, token);
      await loadHistory();
      await loadConversations();
    } catch {
      // ignore
    }
  }, [getToken, loadHistory, loadConversations]);

  return {
    messages,
    conversations,
    isSending,
    isLoading,
    error,
    sendMessage,
    newChat,
    switchConversation,
  };
}
