import { useState } from 'react';
import { ChatHeader } from './ChatHeader';
import { ChatMessageList } from './ChatMessageList';
import { ChatComposer } from './ChatComposer';
import { ChatEmptyState } from './ChatEmptyState';
import type { ChatMessage } from '../../types/chat';

interface ChatPanelProps {
  isOpen: boolean;
  messages: ChatMessage[];
  isSending: boolean;
  error: string | null;
  onClose: () => void;
  onClear: () => void;
  onSend: (message: string) => void;
}

/**
 * Floating AI assistant panel anchored above the footer bar (bottom-14 = 56px,
 * matching the new h-14 footer height).
 *
 * Expand / collapse:
 *   Normal   — 420px wide, up to 600px tall
 *   Expanded — 680px wide, up to 760px tall
 *
 * Both states use min() so the panel never exceeds the viewport height.
 * max-w-[calc(100vw-2rem)] prevents horizontal overflow on small screens.
 * Transitions are smooth (300ms) to match premium messaging-app behaviour.
 */
export function ChatPanel({
  isOpen,
  messages,
  isSending,
  error,
  onClose,
  onClear,
  onSend,
}: ChatPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const hasMessages = messages.length > 0 || isSending;

  function handleToggleExpand() {
    setIsExpanded((prev) => !prev);
  }

  // Derive dimension classes based on expanded state
  const widthClass = isExpanded
    ? 'w-[680px] max-w-[calc(100vw-2rem)]'
    : 'w-[420px] max-w-[calc(100vw-2rem)]';

  const panelHeight = isExpanded
    ? 'min(760px, calc(100vh - 80px))'
    : 'min(600px, calc(100vh - 80px))';

  return (
    <div
      role="dialog"
      aria-label="Stratos AI Assistant"
      aria-hidden={!isOpen}
      className={[
        // Position — sits directly above the h-14 footer (3.5rem = bottom-14)
        'fixed right-4 bottom-14 z-50',
        // Layout
        'flex flex-col',
        // Dimensions
        widthClass,
        // Visual chrome
        'rounded-2xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900',
        // Open / close animation
        'transition-all duration-300 ease-out',
        isOpen
          ? 'pointer-events-auto translate-y-0 opacity-100'
          : 'pointer-events-none translate-y-4 opacity-0',
      ].join(' ')}
      style={{
        height: panelHeight,
        boxShadow: isOpen
          ? '0 25px 60px -12px rgba(0,0,0,0.22), 0 0 0 1px rgba(0,0,0,0.04)'
          : undefined,
      }}
    >
      {/* Fixed header */}
      <ChatHeader
        messageCount={messages.length}
        isExpanded={isExpanded}
        onExpand={handleToggleExpand}
        onClose={onClose}
        onClear={onClear}
      />

      {/* Scrollable message area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {hasMessages ? (
          <ChatMessageList messages={messages} isSending={isSending} error={error} />
        ) : (
          <ChatEmptyState onSend={onSend} />
        )}
      </div>

      {/* Pinned composer */}
      <ChatComposer isSending={isSending} onSend={onSend} />
    </div>
  );
}
