import { API_BASE_URL } from './config';

/**
 * Authenticated fetch helper for backend API calls.
 *
 * Usage in a component:
 *   const { getToken } = useCurrentUser();
 *   const data = await apiFetch('/inventory', {}, await getToken());
 *
 * Chat requests (/api/chat):
 *   Use the Clerk userId as the thread/username identifier in the request body.
 *   Example body: { userId, message }
 *   The userId is available via useCurrentUser() → userId.
 */
export async function apiFetch<T = unknown>(
  path: string,
  options: RequestInit = {},
  token?: string | null,
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }

  return res.json() as Promise<T>;
}
