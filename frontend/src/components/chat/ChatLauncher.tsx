function SparkleIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 15 15" fill="none" aria-hidden="true">
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

interface ChatLauncherProps {
  isOpen: boolean;
  onToggle: () => void;
}

/**
 * Branded CTA button that opens / closes the Stratos AI chat panel.
 *
 * Always rendered as a filled blue button regardless of light/dark mode —
 * this is a primary action anchor and should remain visually prominent on
 * both the dark footer (light mode) and light footer (dark mode) surfaces.
 *
 * Hover: one shade darker (blue-700) for gentle depth feedback.
 */
export function ChatLauncher({ isOpen, onToggle }: ChatLauncherProps) {
  return (
    <button
      onClick={onToggle}
      aria-label={isOpen ? 'Close AI assistant' : 'Open AI assistant'}
      aria-expanded={isOpen}
      className="flex items-center gap-2 rounded-lg bg-blue-600 px-3.5 py-2 text-[12px] font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-1 focus-visible:ring-offset-transparent"
    >
      <SparkleIcon />
      <span>Ask Stratos AI</span>
    </button>
  );
}
