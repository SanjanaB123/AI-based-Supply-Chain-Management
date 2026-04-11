import { useState, useEffect, useCallback, useRef } from 'react';
import { useCurrentUser } from './useCurrentUser';
import { sendChatMessage } from '../lib/chat';
import type { ChatMessage } from '../types/chat';

const STORAGE_KEY_PREFIX = 'stratos-chat:';

function getStorageKey(userId: string): string {
  return `${STORAGE_KEY_PREFIX}${userId}`;
}

function loadMessages(userId: string): ChatMessage[] {
  try {
    const raw = localStorage.getItem(getStorageKey(userId));
    if (!raw) return [];
    return JSON.parse(raw) as ChatMessage[];
  } catch {
    return [];
  }
}

function persistMessages(userId: string, messages: ChatMessage[]): void {
  try {
    localStorage.setItem(getStorageKey(userId), JSON.stringify(messages));
  } catch {
    // Storage quota exceeded or unavailable — ignore silently
  }
}

export interface UseChatAssistantReturn {
  isOpen: boolean;
  messages: ChatMessage[];
  isSending: boolean;
  error: string | null;
  open: () => void;
  close: () => void;
  toggle: () => void;
  sendMessage: (content: string) => Promise<void>;
  clearChat: () => void;
}

export function useChatAssistant(): UseChatAssistantReturn {
  const { userId, getToken } = useCurrentUser();

  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track whether we have loaded from storage for this user
  const loadedForUser = useRef<string | null>(null);

  // Load persisted messages when userId becomes available
  useEffect(() => {
    if (userId && loadedForUser.current !== userId) {
      loadedForUser.current = userId;
      const saved = loadMessages(userId);
      setMessages(saved);
    }
  }, [userId]);

  // Persist messages whenever they change (only when userId is available)
  useEffect(() => {
    if (userId && loadedForUser.current === userId) {
      persistMessages(userId, messages);
    }
  }, [messages, userId]);

  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  const toggle = useCallback(() => setIsOpen((prev) => !prev), []);

  const sendMessage = useCallback(
    async (content: string) => {
      const trimmed = content.trim();
      if (!trimmed || isSending || !userId) return;

      const userMsg: ChatMessage = {
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
        const result = await sendChatMessage(trimmed, userId, token);

        const assistantMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.response,
          createdAt: Date.now(),
          status: 'sent',
        };

        setMessages((prev) => [...prev, assistantMsg]);
      } catch {
        setError('Failed to get a response. Please try again.');
      } finally {
        setIsSending(false);
      }
    },
    [isSending, userId, getToken],
  );

  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
    if (userId) {
      try {
        localStorage.removeItem(getStorageKey(userId));
      } catch {
        // ignore
      }
    }
  }, [userId]);

  return {
    isOpen,
    messages,
    isSending,
    error,
    open,
    close,
    toggle,
    sendMessage,
    clearChat,
  };
}
