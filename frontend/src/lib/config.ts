const clerkPublishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

if (!clerkPublishableKey) {
  throw new Error("Missing VITE_CLERK_PUBLISHABLE_KEY");
}

if (!apiBaseUrl) {
  throw new Error("Missing VITE_API_BASE_URL");
}

export const CLERK_PUBLISHABLE_KEY = clerkPublishableKey;
export const API_BASE_URL = apiBaseUrl;
