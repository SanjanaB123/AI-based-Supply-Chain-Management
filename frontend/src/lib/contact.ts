import { apiFetch } from './api';
import type { ContactFormData, ContactResponse } from '../types/gemini-chat';

export async function sendContactMessage(
  data: ContactFormData,
  token: string | null,
): Promise<ContactResponse> {
  return apiFetch<ContactResponse>(
    '/api/contact',
    {
      method: 'POST',
      body: JSON.stringify(data),
    },
    token,
  );
}
