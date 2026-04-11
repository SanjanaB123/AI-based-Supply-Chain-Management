export function TypingIndicator() {
  return (
    <div className="flex justify-start" aria-label="Assistant is thinking">
      <div className="rounded-2xl rounded-bl-sm bg-slate-100 dark:bg-slate-800 px-4 py-3">
        <div className="flex items-center gap-1">
          <span
            className="h-1.5 w-1.5 rounded-full bg-slate-400 dark:bg-slate-500 animate-bounce"
            style={{ animationDelay: '0ms', animationDuration: '900ms' }}
          />
          <span
            className="h-1.5 w-1.5 rounded-full bg-slate-400 dark:bg-slate-500 animate-bounce"
            style={{ animationDelay: '150ms', animationDuration: '900ms' }}
          />
          <span
            className="h-1.5 w-1.5 rounded-full bg-slate-400 dark:bg-slate-500 animate-bounce"
            style={{ animationDelay: '300ms', animationDuration: '900ms' }}
          />
        </div>
      </div>
    </div>
  );
}
