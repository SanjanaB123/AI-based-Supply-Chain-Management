import type { ChatMessage } from '../../types/chat';

interface ChatMessageBubbleProps {
  message: ChatMessage;
}

export function ChatMessageBubble({ message }: ChatMessageBubbleProps) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-blue-600 px-3.5 py-2.5 text-[13px] leading-relaxed text-white">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm bg-slate-100 dark:bg-slate-800 px-3.5 py-2.5 text-[13px] leading-relaxed text-slate-800 dark:text-slate-200 whitespace-pre-wrap">
        {message.content}
      </div>
    </div>
  );
}
