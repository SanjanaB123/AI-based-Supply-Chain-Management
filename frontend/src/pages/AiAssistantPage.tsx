import { useEffect } from 'react';
import { useGeminiChat } from '../hooks/useGeminiChat';
import { useTopBar } from '../app/TopBarContext';
import { AiChatSidebar } from '../components/ai/AiChatSidebar';
import { AiMessageList } from '../components/ai/AiMessageList';
import { AiChatComposer } from '../components/ai/AiChatComposer';
import { AiEmptyState } from '../components/ai/AiEmptyState';

export default function AiAssistantPage() {
  const { setPageMeta } = useTopBar();
  const {
    messages, conversations, isSending, isLoading, error,
    sendMessage, newChat, switchConversation,
  } = useGeminiChat();

  useEffect(() => {
    setPageMeta('AI Assistant', 'Chat with Stratos AI to manage inventory and place orders');
  }, [setPageMeta]);

  const hasMessages = messages.length > 0 || isSending;

  return (
    <div className="flex h-full overflow-hidden bg-white dark:bg-slate-900">
      {/* Sidebar — hidden on small screens, fixed height */}
      <div className="hidden md:flex md:h-full">
        <AiChatSidebar
          conversations={conversations}
          onNewChat={newChat}
          onSelectConversation={switchConversation}
        />
      </div>

      {/* Main chat area — flex column, no scroll on outer */}
      <div className="flex flex-1 flex-col min-h-0">
        {/* Mobile header */}
        <div className="flex shrink-0 items-center justify-between border-b border-slate-200 px-4 py-3 md:hidden dark:border-slate-700">
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
            AI Assistant
          </h2>
          <button
            onClick={newChat}
            className="rounded-lg px-3 py-1.5 text-[12px] font-medium text-slate-600 transition-colors hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
          >
            New Chat
          </button>
        </div>

        {/* Messages area — this is the only scrollable part */}
        {isLoading ? (
          <div className="flex flex-1 items-center justify-center min-h-0">
            <div className="flex flex-col items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
              <p className="text-sm text-slate-500 dark:text-slate-400">Loading conversation...</p>
            </div>
          </div>
        ) : hasMessages ? (
          <AiMessageList messages={messages} isSending={isSending} error={error} />
        ) : (
          <AiEmptyState onSend={sendMessage} />
        )}

        {/* Composer — pinned to bottom, never scrolls */}
        {!isLoading && (
          <div className="shrink-0">
            <AiChatComposer isSending={isSending} onSend={sendMessage} />
          </div>
        )}
      </div>
    </div>
  );
}
