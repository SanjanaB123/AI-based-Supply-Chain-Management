const SUGGESTED_PROMPTS = [
  'What are the low stock items in this store?',
  'Summarize inventory health for the selected store',
  'Which products are at lead-time risk?',
  'What shrinkage issues should I look at?',
  'Give me the top operational risks right now',
] as const;

function SparkleIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 15 15" fill="none" aria-hidden="true">
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

interface ChatEmptyStateProps {
  onSend: (message: string) => void;
}

export function ChatEmptyState({ onSend }: ChatEmptyStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-5 px-5 py-6 overflow-y-auto">
      {/* Icon + intro */}
      <div className="flex flex-col items-center gap-3 text-center">
        <div className="flex h-11 w-11 items-center justify-center rounded-full bg-blue-50 text-blue-500 dark:bg-blue-900/30 dark:text-blue-400">
          <SparkleIcon />
        </div>
        <div>
          <p className="text-[13px] font-semibold text-slate-800 dark:text-slate-200">
            Stratos AI Assistant
          </p>
          <p className="mt-0.5 text-[12px] text-slate-400 dark:text-slate-500">
            Ask anything about your inventory
          </p>
        </div>
      </div>

      {/* Suggested prompts */}
      <div className="w-full space-y-2">
        {SUGGESTED_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            onClick={() => onSend(prompt)}
            className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-left text-[12px] text-slate-700 transition-colors hover:border-blue-200 hover:bg-blue-50/60 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 dark:border-slate-700 dark:bg-slate-800/50 dark:text-slate-300 dark:hover:border-blue-700 dark:hover:bg-blue-900/20 dark:hover:text-blue-400"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
