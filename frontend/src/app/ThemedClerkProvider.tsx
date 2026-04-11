import { ClerkProvider } from '@clerk/clerk-react';
import type { ReactNode } from 'react';
import { useTheme } from './theme/useTheme';
import { clerkLightTheme, clerkDarkTheme } from '../lib/clerkTheme';

// ── Props ──────────────────────────────────────────────────────────────────────

interface Props {
  children: ReactNode;
  publishableKey: string;
}

// ── Component ──────────────────────────────────────────────────────────────────

/**
 * Reads the current app theme from ThemeContext and passes the matching
 * Clerk appearance object to ClerkProvider. Because this component sits
 * inside <ThemeProvider>, it re-renders whenever the theme toggles,
 * causing Clerk to instantly re-apply the correct appearance.
 */
export default function ThemedClerkProvider({ children, publishableKey }: Props) {
  const { theme } = useTheme();

  return (
    <ClerkProvider
      publishableKey={publishableKey}
      signInUrl="/sign-in"
      signUpUrl="/sign-up"
      appearance={theme === 'dark' ? clerkDarkTheme : clerkLightTheme}
    >
      {children}
    </ClerkProvider>
  );
}
