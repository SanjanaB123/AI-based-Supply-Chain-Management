# Frontend — AI-based Supply Chain Management

## Step 1 — Frontend Scaffolding

### What was done

1. Created a Vite + React + TypeScript app inside `frontend/` using the `react-ts` template.
2. Installed and configured **Tailwind CSS v4** using `@tailwindcss/vite` plugin (no `tailwind.config.js` needed; uses `@import "tailwindcss"` in `index.css`).
3. Installed and wired **`@clerk/clerk-react`** — `ClerkProvider` wraps the entire app in `main.tsx` using `import.meta.env.VITE_CLERK_PUBLISHABLE_KEY`.
4. Installed **`react-router-dom`** and set up `BrowserRouter` routing in `src/routes/AppRouter.tsx`.
5. Created clean folder structure inside `src/`:
   - `app/` — root layout (`RootLayout.tsx`)
   - `components/` — shared UI components (empty, ready to fill)
   - `pages/` — page-level components (`HomePage`, `SignInPage`, `SignUpPage`)
   - `lib/` — utilities (`config.ts` reads `VITE_API_BASE_URL`)
   - `routes/` — router definition (`AppRouter.tsx`)
   - `types/` — shared TypeScript types (empty, ready to fill)
6. Set up minimal routes:
   - `/` → `HomePage` (placeholder)
   - `/sign-in/*` → `SignInPage` (Clerk `<SignIn>` component)
   - `/sign-up/*` → `SignUpPage` (Clerk `<SignUp>` component)
7. Created `frontend/.env.example` with required env var keys.
8. Added `src/lib/config.ts` — exports `API_BASE_URL` from `import.meta.env.VITE_API_BASE_URL`.
9. Removed unused Vite template boilerplate (`App.css`, template assets).

---

### File tree

```
frontend/
├── .env.example
├── README.md
├── eslint.config.js
├── index.html
├── package.json
├── package-lock.json
├── public/
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── vite.config.ts
└── src/
    ├── App.tsx
    ├── index.css
    ├── main.tsx
    ├── app/
    │   └── RootLayout.tsx
    ├── components/       (empty)
    ├── lib/
    │   └── config.ts
    ├── pages/
    │   ├── HomePage.tsx
    │   ├── SignInPage.tsx
    │   └── SignUpPage.tsx
    ├── routes/
    │   └── AppRouter.tsx
    └── types/            (empty)
```

---

### Installed packages

| Package | Version | Type |
|---|---|---|
| `react` | ^19.2.4 | dependency |
| `react-dom` | ^19.2.4 | dependency |
| `@clerk/clerk-react` | ^5.x | dependency |
| `react-router-dom` | ^7.x | dependency |
| `tailwindcss` | ^4.x | devDependency |
| `@tailwindcss/vite` | ^4.x | devDependency |
| `vite` | ^8.x | devDependency |
| `typescript` | ~6.x | devDependency |
| `@vitejs/plugin-react` | ^6.x | devDependency |

---

### Before running: create `frontend/.env.local`

```
VITE_CLERK_PUBLISHABLE_KEY=pk_test_your_key_here
VITE_API_BASE_URL=http://localhost:8000
```

> Get your publishable key from the Clerk dashboard under your application's API Keys section.

---

## Step 2 — Fix Clerk Auth Routing (keep auth inside the Vite app)

### What was fixed

Clicking "Don't have an account? Sign up" inside the Sign In page (and vice versa) was navigating users to the **hosted Clerk domain** instead of staying inside the local app. This happened because `ClerkProvider` had no `signInUrl`/`signUpUrl` props, and the `<SignIn>`/`<SignUp>` components had no cross-link URLs — so Clerk's internal links fell back to the hosted Accounts Portal.

The fix wires all three places so every Clerk-generated navigation link resolves to a local path.

### Files changed

| File | Change |
|---|---|
| `src/main.tsx` | Added `signInUrl="/sign-in"` and `signUpUrl="/sign-up"` to `<ClerkProvider>` |
| `src/pages/SignInPage.tsx` | Added `signUpUrl="/sign-up"` to `<SignIn>` component |
| `src/pages/SignUpPage.tsx` | Added `signInUrl="/sign-in"` to `<SignUp>` component |

### Manual steps required

None — no new env vars, no Clerk dashboard changes. Your existing `VITE_CLERK_PUBLISHABLE_KEY` in `.env.local` is unchanged.

### How to verify the fix

1. Run `npm run dev` and open `http://localhost:5173/sign-in`
2. Click "Don't have an account? Sign up" — the URL should change to `http://localhost:5173/sign-up`, **not** redirect to `accounts.clerk.dev` or any external domain
3. On the Sign Up page, click "Already have an account? Sign in" — the URL should change to `http://localhost:5173/sign-in`
4. Complete a sign-in or sign-up flow — you should remain on `localhost:5173` throughout

---

### Command to start the frontend

```bash
cd frontend
npm run dev
```

App will be available at `http://localhost:5173`.
