import { useEffect, useRef } from 'react';
import { AiMessageBubble } from './AiMessageBubble';
import type { GeminiMessage } from '../../types/gemini-chat';

interface AiMessageListProps {
  messages: GeminiMessage[];
  isSending: boolean;
  error: string | null;
}

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white">
        <svg width="14" height="14" viewBox="0 0 15 15" fill="none" aria-hidden="true">
          <path
            d="M7.5 1v2M7.5 12v2M1 7.5h2M12 7.5h2M3.05 3.05l1.42 1.42M10.53 10.53l1.42 1.42M10.53 4.47l1.42-1.42M3.05 11.95l1.42-1.42"
            stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"
          />
          <circle cx="7.5" cy="7.5" r="2" fill="currentColor" />
        </svg>
      </div>
      <div className="rounded-2xl rounded-tl-sm bg-slate-100 px-4 py-3 dark:bg-slate-800">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 dark:bg-slate-500" style={{ animationDelay: '0ms' }} />
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 dark:bg-slate-500" style={{ animationDelay: '150ms' }} />
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 dark:bg-slate-500" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
}

export function AiMessageList({ messages, isSending, error }: AiMessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 sm:px-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-4">
        {messages.map((msg) => (
          <AiMessageBubble key={msg.id} message={msg} />
        ))}

        {isSending && <TypingIndicator />}

        {error && (
          <div className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
