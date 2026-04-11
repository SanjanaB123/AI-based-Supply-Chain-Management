import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';
import ThemeProvider from './app/theme/ThemeProvider';
import ThemedClerkProvider from './app/ThemedClerkProvider';
import { CLERK_PUBLISHABLE_KEY } from './lib/config';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <ThemedClerkProvider publishableKey={CLERK_PUBLISHABLE_KEY}>
        <App />
      </ThemedClerkProvider>
    </ThemeProvider>
  </StrictMode>,
);
