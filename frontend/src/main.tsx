import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';
import ThemeProvider from './app/theme/ThemeProvider';
import ThemedClerkProvider from './app/ThemedClerkProvider';

const publishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY as string;

if (!publishableKey) {
  throw new Error('Missing VITE_CLERK_PUBLISHABLE_KEY environment variable.');
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <ThemedClerkProvider publishableKey={publishableKey}>
        <App />
      </ThemedClerkProvider>
    </ThemeProvider>
  </StrictMode>,
);
