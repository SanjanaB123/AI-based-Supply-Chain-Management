import { createContext, useCallback, useContext, useState } from 'react';
import type { ReactNode } from 'react';

// ── Context ───────────────────────────────────────────────────────────────────
// Allows pages to inject a slot element into the desktop top bar.
// The top bar renders whatever is in `topBarSlot`; pages call `setTopBarSlot`
// in a useEffect to populate it (and clear it on unmount).

interface TopBarContextValue {
  topBarSlot: ReactNode;
  setTopBarSlot: (node: ReactNode) => void;
}

const TopBarContext = createContext<TopBarContextValue | null>(null);

export function TopBarProvider({ children }: { children: ReactNode }) {
  const [topBarSlot, setSlot] = useState<ReactNode>(null);
  const setTopBarSlot = useCallback((node: ReactNode) => setSlot(node), []);
  return (
    <TopBarContext.Provider value={{ topBarSlot, setTopBarSlot }}>
      {children}
    </TopBarContext.Provider>
  );
}

export function useTopBar() {
  const ctx = useContext(TopBarContext);
  if (!ctx) throw new Error('useTopBar must be used within TopBarProvider');
  return ctx;
}
