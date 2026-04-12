import { createContext, useEffect, type ReactNode } from 'react';

export type Theme = 'light' | 'dark';

interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
}

// eslint-disable-next-line react-refresh/only-export-components
export const ThemeContext = createContext<ThemeContextValue>({
  theme: 'light',
  toggleTheme: () => {},
});

export default function ThemeProvider({ children }: { children: ReactNode }) {
  // Always light mode
  useEffect(() => {
    document.documentElement.classList.remove('dark');
    localStorage.setItem('stratos-theme', 'light');
  }, []);

  return (
    <ThemeContext.Provider value={{ theme: 'light', toggleTheme: () => {} }}>
      {children}
    </ThemeContext.Provider>
  );
}
