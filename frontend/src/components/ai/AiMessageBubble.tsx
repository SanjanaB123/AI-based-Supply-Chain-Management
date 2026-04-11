import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { GeminiMessage } from '../../types/gemini-chat';

interface AiMessageBubbleProps {
  message: GeminiMessage;
}

function BotIcon() {
  return (
    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 text-white shadow-sm">
      <svg width="16" height="16" viewBox="0 0 15 15" fill="none" aria-hidden="true">
        <path
          d="M7.5 1v2M7.5 12v2M1 7.5h2M12 7.5h2M3.05 3.05l1.42 1.42M10.53 10.53l1.42 1.42M10.53 4.47l1.42-1.42M3.05 11.95l1.42-1.42"
          stroke="currentColor"
          strokeWidth="1.3"
          strokeLinecap="round"
        />
        <circle cx="7.5" cy="7.5" r="2" fill="currentColor" />
      </svg>
    </div>
  );
}

export function AiMessageBubble({ message }: AiMessageBubbleProps) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-blue-600 px-4 py-3 text-sm leading-relaxed text-white shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <BotIcon />
      <div className="min-w-0 max-w-[85%] overflow-hidden rounded-2xl rounded-tl-sm bg-white px-5 py-4 text-sm leading-relaxed text-slate-800 shadow-sm ring-1 ring-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:ring-slate-700">
        <div className="ai-markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ children, ...props }) => (
                <div className="table-wrap">
                  <table {...props}>{children}</table>
                </div>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        {message.functionCalls && message.functionCalls.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5 border-t border-slate-100 pt-2.5 dark:border-slate-700">
            {message.functionCalls.map((fn, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 rounded-full bg-blue-50 px-2.5 py-0.5 text-[10px] font-medium text-blue-600 dark:bg-blue-900/30 dark:text-blue-400"
              >
                <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><circle cx="5" cy="5" r="2" fill="currentColor" /></svg>
                {fn}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
