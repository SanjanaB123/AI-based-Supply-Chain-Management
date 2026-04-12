import { useState, useEffect } from 'react';
import { fetchAiStatus } from '../../lib/chat';
import type { AiStatusResponse } from '../../types/chat';

type LoadState = 'loading' | 'ok' | 'error';

interface AiStatusIndicatorProps {
  /** When true, uses text colors that are readable on the inverted (opposite-theme) footer surface */
  inverted?: boolean;
}

export function AiStatusIndicator({ inverted = false }: AiStatusIndicatorProps) {
  const [data, setData] = useState<AiStatusResponse | null>(null);
  const [loadState, setLoadState] = useState<LoadState>('loading');

  useEffect(() => {
    let cancelled = false;

    async function refresh() {
      try {
        const result = await fetchAiStatus();
        if (!cancelled) {
          setData(result);
          setLoadState('ok');
        }
      } catch {
        if (!cancelled) {
          setLoadState('error');
        }
      }
    }

    refresh();
    const interval = setInterval(refresh, 30_000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // Text classes for the two contexts:
  //   normal  : slate-500 on white/light surface (light mode) | slate-400 on dark surface (dark mode)
  //   inverted: slate-300 on slate-900 footer (light mode)    | slate-600 on slate-100 footer (dark mode)
  const textClass = inverted
    ? 'text-slate-300 dark:text-slate-600'
    : 'text-slate-500 dark:text-slate-400';

  const dimTextClass = inverted
    ? 'text-slate-400 dark:text-slate-500'
    : 'text-slate-400 dark:text-slate-500';

  const separatorClass = inverted
    ? 'bg-slate-600 dark:bg-slate-400'
    : 'bg-slate-200 dark:bg-slate-700';

  const loadingDotClass = inverted
    ? 'bg-slate-500 dark:bg-slate-400'
    : 'bg-slate-300 dark:bg-slate-600';

  if (loadState === 'loading') {
    return (
      <div className="flex items-center gap-1.5" aria-label="Checking AI status">
        <span className={`h-1.5 w-1.5 rounded-full ${loadingDotClass}`} />
        <span className={`text-[11px] select-none ${dimTextClass}`}>
          Connecting…
        </span>
      </div>
    );
  }

  if (loadState === 'error' || !data) {
    return (
      <div className="flex items-center gap-1.5" aria-label="AI unavailable">
        <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
        <span className={`text-[11px] select-none ${dimTextClass}`}>
          AI unavailable
        </span>
      </div>
    );
  }

  const isOnline = data.status === 'online' && data.ai_enabled;

  return (
    <div className="flex items-center gap-2" aria-label={isOnline ? 'AI online' : 'AI limited'}>
      <div className="flex items-center gap-1.5">
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            isOnline ? 'bg-emerald-400' : 'bg-amber-400'
          }`}
        />
        <span className={`text-[11px] select-none ${textClass}`}>
          {isOnline ? 'AI online' : 'AI limited'}
        </span>
      </div>

      {isOnline && data.model_loaded && (
        <>
          <span className={`h-3 w-px ${separatorClass}`} aria-hidden="true" />
          <span className={`text-[10px] select-none ${dimTextClass}`}>
            Model loaded
          </span>
        </>
      )}
    </div>
  );
}
