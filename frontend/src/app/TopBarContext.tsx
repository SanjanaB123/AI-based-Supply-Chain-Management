import { createContext, useCallback, useContext, useState } from 'react';
import type { ReactNode } from 'react';

// ── Context ───────────────────────────────────────────────────────────────────
// Allows pages to inject content into the desktop top bar.
// - topBarSlot: arbitrary node (store selector, etc.)
// - pageTitle / pageSubtitle: page header metadata rendered left of the store slot

interface TopBarContextValue {
  topBarSlot: ReactNode;
  setTopBarSlot: (node: ReactNode) => void;
  pageTitle: string;
  pageSubtitle: string;
  setPageMeta: (title: string, subtitle: string) => void;
}

const TopBarContext = createContext<TopBarContextValue | null>(null);

export function TopBarProvider({ children }: { children: ReactNode }) {
  const [topBarSlot, setSlot]       = useState<ReactNode>(null);
  const [pageTitle, setTitle]       = useState('');
  const [pageSubtitle, setSubtitle] = useState('');

  const setTopBarSlot = useCallback((node: ReactNode) => setSlot(node), []);
  const setPageMeta   = useCallback((title: string, subtitle: string) => {
    setTitle(title);
    setSubtitle(subtitle);
  }, []);

  return (
    <TopBarContext.Provider value={{ topBarSlot, setTopBarSlot, pageTitle, pageSubtitle, setPageMeta }}>
      {children}
    </TopBarContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTopBar() {
  const ctx = useContext(TopBarContext);
  if (!ctx) throw new Error('useTopBar must be used within TopBarProvider');
  return ctx;
}
