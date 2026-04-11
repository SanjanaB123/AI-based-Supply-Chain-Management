function XIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
      <path
        d="M1 1l11 11M12 1L1 12"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
      <path
        d="M1 3.5h11M4.5 3.5V2.5a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1M10.5 3.5l-.75 7.5h-7.5L1.5 3.5h9z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Outward-pointing arrows — expand to larger panel */
function ExpandIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
      <path
        d="M1.5 5V1.5H5M8 1.5h3.5V5M11.5 8v3.5H8M5 11.5H1.5V8"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Inward-pointing arrows — collapse back to normal */
function CollapseIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
      <path
        d="M5 1.5V5H1.5M11.5 5H8V1.5M8 11.5V8h3.5M1.5 8H5v3.5"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

interface ChatHeaderProps {
  messageCount: number;
  isExpanded: boolean;
  onExpand: () => void;
  onClose: () => void;
  onClear: () => void;
}

const iconBtnClass =
  'flex h-7 w-7 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 dark:text-slate-500 dark:hover:bg-slate-800 dark:hover:text-slate-300';

export function ChatHeader({ messageCount, isExpanded, onExpand, onClose, onClear }: ChatHeaderProps) {
  return (
    <div className="flex shrink-0 items-center gap-3 border-b border-slate-100 px-4 py-3 dark:border-slate-800">
      {/* Status + title */}
      <div className="flex flex-1 items-center gap-2.5">
        <span className="h-2 w-2 rounded-full bg-emerald-400 ring-2 ring-emerald-100 dark:ring-emerald-900/30" />
        <div>
          <p className="text-[13px] font-semibold leading-tight text-slate-900 dark:text-slate-100">
            Stratos AI Assistant
          </p>
          <p className="text-[10px] leading-tight text-slate-400 dark:text-slate-500">
            Inventory copilot
          </p>
        </div>
      </div>

      {/* Actions: expand · clear · close */}
      <div className="flex items-center gap-0.5">
        {/* Expand / collapse — always visible */}
        <button
          onClick={onExpand}
          aria-label={isExpanded ? 'Collapse chat panel' : 'Expand chat panel'}
          className={iconBtnClass}
        >
          {isExpanded ? <CollapseIcon /> : <ExpandIcon />}
        </button>

        {/* Clear — only when there are messages */}
        {messageCount > 0 && (
          <button
            onClick={onClear}
            title="Clear conversation"
            aria-label="Clear conversation"
            className={iconBtnClass}
          >
            <TrashIcon />
          </button>
        )}

        <button
          onClick={onClose}
          aria-label="Close assistant"
          className={iconBtnClass}
        >
          <XIcon />
        </button>
      </div>
    </div>
  );
}
