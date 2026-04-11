import { useState, useRef, useEffect } from 'react';

function SendIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M1.5 1.5l11 5.5-11 5.5V8.5l7.5-1.5-7.5-1.5V1.5z" fill="currentColor" />
    </svg>
  );
}

interface ChatComposerProps {
  isSending: boolean;
  onSend: (message: string) => void;
}

export function ChatComposer({ isSending, onSend }: ChatComposerProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Refocus the textarea after a message is sent
  useEffect(() => {
    if (!isSending && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isSending]);

  function handleInput(e: React.FormEvent<HTMLTextAreaElement>) {
    const el = e.currentTarget;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleSend() {
    const trimmed = value.trim();
    if (!trimmed || isSending) return;
    onSend(trimmed);
    setValue('');
    // Reset height after clearing
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }

  const canSend = value.trim().length > 0 && !isSending;

  return (
    <div className="shrink-0 border-t border-slate-100 px-3 py-3 dark:border-slate-800">
      <div className="flex items-end gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 transition-all focus-within:border-blue-300 focus-within:ring-2 focus-within:ring-blue-100 dark:border-slate-700 dark:bg-slate-800/60 dark:focus-within:border-blue-600 dark:focus-within:ring-blue-900/30">
        <textarea
          ref={textareaRef}
          value={value}
          rows={1}
          disabled={isSending}
          placeholder="Ask about inventory, stock health, shrinkage…"
          onChange={(e) => setValue(e.target.value)}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          className="max-h-32 flex-1 resize-none overflow-y-auto bg-transparent text-[13px] leading-relaxed text-slate-800 placeholder:text-slate-400 focus:outline-none disabled:opacity-60 dark:text-slate-200 dark:placeholder:text-slate-500"
          style={{ minHeight: '20px' }}
        />

        <button
          onClick={handleSend}
          disabled={!canSend}
          aria-label="Send message"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white transition-all hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
        >
          <SendIcon />
        </button>
      </div>

      <p className="mt-1.5 text-center text-[10px] text-slate-300 select-none dark:text-slate-600">
        Enter to send · Shift+Enter for newline
      </p>
    </div>
  );
}
