import { useEffect, useRef } from 'react';
import { ChatMessageBubble } from './ChatMessageBubble';
import { TypingIndicator } from './TypingIndicator';
import type { ChatMessage } from '../../types/chat';

interface ChatMessageListProps {
  messages: ChatMessage[];
  isSending: boolean;
  error: string | null;
}

export function ChatMessageList({ messages, isSending, error }: ChatMessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
      {messages.map((msg) => (
        <ChatMessageBubble key={msg.id} message={msg} />
      ))}

      {isSending && <TypingIndicator />}

      {error && (
        <div className="rounded-xl border border-red-100 bg-red-50 px-3.5 py-2.5 text-[12px] text-red-600 dark:border-red-800/30 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
