import { useContext } from 'react';
import { ThemeContext } from './ThemeProvider';
import type { Theme } from './ThemeProvider';

export type { Theme };

export function useTheme() {
  return useContext(ThemeContext);
}
