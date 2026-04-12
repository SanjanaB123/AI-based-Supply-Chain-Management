import { useAuth } from '@clerk/clerk-react';

/**
 * Returns the current Clerk user's ID and a token getter.
 *
 * getToken() resolves to a short-lived JWT suitable for use as a Bearer
 * token in API requests. Pass the result directly to apiFetch():
 *
 *   const { userId, getToken } = useCurrentUser();
 *   const data = await apiFetch('/inventory', {}, await getToken());
 *
 * For /api/chat: pass userId as the thread identifier in the request body
 * so the backend can maintain per-user conversation history.
 */
export function useCurrentUser() {
  const { userId, getToken } = useAuth();
  return { userId, getToken };
}
