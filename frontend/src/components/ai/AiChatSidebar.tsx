import { useState } from 'react';
import type { Conversation } from '../../types/gemini-chat';

interface AiChatSidebarProps {
  conversations: Conversation[];
  onNewChat: () => void;
  onSelectConversation: (convoId: string) => void;
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.3" />
      <path d="M9.5 9.5L12.5 12.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

function ChatBubbleIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path
        d="M2 3.5A1.5 1.5 0 013.5 2h9A1.5 1.5 0 0114 3.5v7a1.5 1.5 0 01-1.5 1.5H6L3 14.5V12H3.5A1.5 1.5 0 012 10.5v-7z"
        stroke="currentColor"
        strokeWidth="1.2"
      />
    </svg>
  );
}

function ClearIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M3 3l6 6M9 3L3 9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

export function AiChatSidebar({ conversations, onNewChat, onSelectConversation }: AiChatSidebarProps) {
  const [search, setSearch] = useState('');

  const filtered = search.trim()
    ? conversations.filter((c) => c.title.toLowerCase().includes(search.toLowerCase()))
    : conversations;

  return (
    <div className="flex w-64 flex-col border-r border-slate-200 bg-slate-50 dark:border-slate-700 dark:bg-slate-900/50 lg:w-72">
      {/* New Chat button */}
      <div className="border-b border-slate-200 px-4 py-4 dark:border-slate-700">
        <button
          onClick={onNewChat}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-[13px] font-medium text-slate-700 transition-colors hover:bg-slate-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          <PlusIcon />
          New Chat
        </button>
      </div>

      {/* Search bar */}
      <div className="px-3 pt-3 pb-1">
        <div className="relative">
          <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 dark:text-slate-500">
            <SearchIcon />
          </span>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search chats..."
            className="h-8 w-full rounded-lg border border-slate-200 bg-white pl-8 pr-8 text-[12px] text-slate-700 placeholder:text-slate-400 transition-all focus:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-300 dark:placeholder:text-slate-500 dark:focus:border-blue-600 dark:focus:ring-blue-900/30"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
            >
              <ClearIcon />
            </button>
          )}
        </div>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-3 py-2">
        {filtered.length > 0 ? (
          <div className="space-y-0.5">
            {filtered.map((convo) => (
              <button
                key={convo.convo_id}
                onClick={() => onSelectConversation(convo.convo_id)}
                className={`flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-left text-[13px] transition-colors ${
                  convo.active
                    ? 'bg-blue-50 font-medium text-blue-700 dark:bg-blue-900/20 dark:text-blue-400'
                    : 'text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800'
                }`}
              >
                <ChatBubbleIcon />
                <span className="truncate">{convo.title}</span>
              </button>
            ))}
          </div>
        ) : search ? (
          <div className="flex flex-col items-center gap-1 py-8 text-center">
            <p className="text-[13px] text-slate-400 dark:text-slate-500">No results</p>
            <p className="text-[11px] text-slate-300 dark:text-slate-600">
              No chats matching "{search}"
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 py-12 text-center">
            <p className="text-[13px] text-slate-400 dark:text-slate-500">No conversations yet</p>
            <p className="text-[11px] text-slate-300 dark:text-slate-600">
              Start a new chat to get help with inventory
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-slate-200 px-4 py-3 dark:border-slate-700">
        <p className="text-[10px] text-slate-400 dark:text-slate-600">
          Stratos AI Assistant
        </p>
      </div>
    </div>
  );
}
