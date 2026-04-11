import { AiStatusIndicator } from './AiStatusIndicator';
import { ChatLauncher } from './ChatLauncher';

interface DashboardFooterBarProps {
  isOpen: boolean;
  onToggle: () => void;
}

/**
 * Dashboard footer "control dock".
 *
 * Opposite-theme surface: dark in light mode, light in dark mode.
 * This creates a strong visual separation from the main dashboard canvas
 * and visually anchors the bottom of the shell (matching the sidebar
 * user-account panel in visual weight).
 *
 * Light mode: bg-slate-900 / border-slate-700  (dark dock)
 * Dark mode:  bg-slate-100 / border-slate-300  (light dock)
 */
export function DashboardFooterBar({ isOpen, onToggle }: DashboardFooterBarProps) {
  return (
    <footer className="flex h-14 shrink-0 items-center border-t border-slate-700 bg-slate-900 px-5 dark:border-slate-300 dark:bg-slate-100">
      {/* Left: AI connection status */}
      <div className="flex flex-1 items-center">
        <AiStatusIndicator inverted />
      </div>

      {/* Center: copyright */}
      <div className="flex flex-1 items-center justify-center">
        <p className="select-none text-[11px] text-slate-400 dark:text-slate-500">
          © 2026 Stratos. All rights reserved.
        </p>
      </div>

      {/* Right: AI assistant launcher CTA */}
      <div className="flex flex-1 items-center justify-end">
        <ChatLauncher isOpen={isOpen} onToggle={onToggle} />
      </div>
    </footer>
  );
}
