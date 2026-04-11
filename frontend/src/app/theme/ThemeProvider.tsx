import { createContext, useEffect, useState, type ReactNode } from 'react';

// ── Types ─────────────────────────────────────────────────────────────────────

export type Theme = 'light' | 'dark';

interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
}

// ── Context ───────────────────────────────────────────────────────────────────

// eslint-disable-next-line react-refresh/only-export-components
export const ThemeContext = createContext<ThemeContextValue>({
  theme: 'light',
  toggleTheme: () => {},
});

// ── Provider ──────────────────────────────────────────────────────────────────

export default function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    // Read persisted preference; default to light
    const saved = localStorage.getItem('stratos-theme');
    return saved === 'dark' ? 'dark' : 'light';
  });

  // Apply / remove the 'dark' class on <html> whenever theme changes
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('stratos-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}
