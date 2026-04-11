const SUGGESTED_PROMPTS = [
  "What's the stock status at store S001?",
  'Which products need reordering at S003?',
  'Order 500 units of P0010 for store S001',
  'Show me vendor options for Snacks',
  'What orders have been placed for S002?',
  'Which items are below 50 units at S004?',
] as const;

function SparkleIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path
        d="M7.5 1v2M7.5 12v2M1 7.5h2M12 7.5h2M3.05 3.05l1.42 1.42M10.53 10.53l1.42 1.42M10.53 4.47l1.42-1.42M3.05 11.95l1.42-1.42"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
      <circle cx="7.5" cy="7.5" r="2" fill="currentColor" />
    </svg>
  );
}

interface AiEmptyStateProps {
  onSend: (message: string) => void;
}

export function AiEmptyState({ onSend }: AiEmptyStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-6 py-12">
      {/* Icon + intro */}
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-50 text-blue-500 dark:bg-blue-900/30 dark:text-blue-400">
          <SparkleIcon />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Stratos AI Assistant
          </h2>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Check inventory, get reorder suggestions, and place orders through conversation.
          </p>
        </div>
      </div>

      {/* Suggested prompts */}
      <div className="grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTED_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            onClick={() => onSend(prompt)}
            className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-left text-[13px] text-slate-700 transition-colors hover:border-blue-200 hover:bg-blue-50/60 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 dark:border-slate-700 dark:bg-slate-800/50 dark:text-slate-300 dark:hover:border-blue-700 dark:hover:bg-blue-900/20 dark:hover:text-blue-400"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
