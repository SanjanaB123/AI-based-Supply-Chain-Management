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

## Step 3 — Auth/App-Shell Foundation

### What was done

1. **Protected route guard** (`src/components/ProtectedRoute.tsx`) — uses `useAuth()` to redirect unauthenticated users to `/sign-in`. Returns `null` while Clerk is loading to prevent flash.
2. **AppShell layout** (`src/app/AppShell.tsx`) — authenticated shell with a top bar containing the Stratos brand link, a placeholder nav, and Clerk's `<UserButton>` on the right.
3. **AuthLayout** (`src/app/AuthLayout.tsx`) — full-page wrapper for sign-in/sign-up routes; renders "Stratos" + tagline above the Clerk card.
4. **DashboardPage** (`src/pages/DashboardPage.tsx`) — placeholder page at `/dashboard`.
5. **API helper** (`src/lib/api.ts`) — `apiFetch(path, options?, token?)` attaches a Bearer token when provided; throws on non-OK responses. Comments document how `/api/chat` should use the Clerk `userId` as the thread identifier.
6. **useCurrentUser hook** (`src/hooks/useCurrentUser.ts`) — thin wrapper over `useAuth()` that exposes `{ userId, getToken }` for use in feature pages.
7. **AppRouter restructured** (`src/routes/AppRouter.tsx`) — three route groups: public (RootLayout), auth (AuthLayout), and protected (ProtectedRoute → AppShell).
8. **SignInPage / SignUpPage simplified** — centering removed; `AuthLayout` owns the page layout now. Both components include `fallbackRedirectUrl="/dashboard"` so successful auth always lands on the dashboard.
9. **HomePage updated** — shows "Stratos" brand, AI supply chain tagline, and context-aware CTAs: signed-out users see Sign In + Sign Up buttons; signed-in users see Go to Dashboard.

### Files created

| File | Purpose |
|---|---|
| `src/components/ProtectedRoute.tsx` | Auth guard — redirects to /sign-in if not signed in |
| `src/app/AppShell.tsx` | Authenticated layout with top bar and UserButton |
| `src/app/AuthLayout.tsx` | Page wrapper for sign-in/sign-up with Stratos branding |
| `src/pages/DashboardPage.tsx` | Placeholder dashboard at /dashboard |
| `src/lib/api.ts` | Authenticated fetch helper with Bearer token support |
| `src/hooks/useCurrentUser.ts` | Hook returning `{ userId, getToken }` from Clerk |

### Files modified

| File | Change |
|---|---|
| `src/routes/AppRouter.tsx` | Restructured into public / auth / protected route groups |
| `src/pages/SignInPage.tsx` | Removed self-centering wrapper; added `fallbackRedirectUrl="/dashboard"` |
| `src/pages/SignUpPage.tsx` | Removed self-centering wrapper; added `fallbackRedirectUrl="/dashboard"` |
| `src/pages/HomePage.tsx` | Added Stratos brand, tagline, and auth-aware CTA buttons |

### No new packages required

All functionality uses existing dependencies: `@clerk/clerk-react`, `react-router-dom`, `react`, Tailwind.

### No env changes required

`VITE_CLERK_PUBLISHABLE_KEY` and `VITE_API_BASE_URL` are unchanged.

### Routes

| Path | Access | Layout |
|---|---|---|
| `/` | Public | RootLayout |
| `/sign-in/*` | Public | AuthLayout + Clerk SignIn card |
| `/sign-up/*` | Public | AuthLayout + Clerk SignUp card |
| `/dashboard` | Protected (requires Clerk session) | AppShell |

### How to verify this step

1. `cd frontend && npm run dev`
2. Open `http://localhost:5173` — see Stratos landing page with Sign In / Sign Up buttons
3. Go to `http://localhost:5173/dashboard` while signed out — should redirect to `/sign-in`
4. Sign in via `/sign-in` — should redirect to `/dashboard` with the AppShell top bar visible
4a. Sign up via `/sign-up` — should also redirect to `/dashboard` after completing registration
5. The top bar should show "Stratos" (left), "Dashboard" nav link, and Clerk `UserButton` (right)
6. Click the `UserButton` — account menu and sign-out should work
7. After signing out, navigating to `/dashboard` should redirect to `/sign-in` again
8. On the landing page after signing in, the button should say "Go to Dashboard" (not Sign In/Sign Up)

### /api/chat future compatibility

The `src/lib/api.ts` and `src/hooks/useCurrentUser.ts` files include inline comments:
- `apiFetch` accepts an optional Bearer token — call `await getToken()` from `useCurrentUser()` before each request
- For `/api/chat`: pass `userId` in the request body as the thread/username identifier so the backend maintains per-user conversation history

---

## Step 4 — Live Dashboard Integration + Dashboard UI Foundation

### What was done

1. **Installed Recharts** (`recharts@3.8.1`) — chart library used for all charts going forward.
2. **Typed API responses** in `src/types/inventory.ts` — exact TypeScript types derived from the live backend (`StoresResponse`, `StockLevelsResponse`, `StockHealthResponse`, `StockStatus`).
3. **Inventory data layer** in `src/lib/inventory.ts` — `fetchStores()`, `fetchStockLevels(storeId)`, `fetchStockHealth(storeId)` each pass a Clerk Bearer token through the existing `apiFetch` helper.
4. **Rebuilt DashboardPage** with live backend data:
   - Fetches store list on mount, auto-selects the first store
   - Store selector re-triggers data fetch when changed
   - Previous data is cleared immediately on store switch (prevents stale display)
   - Parallel fetch of `stock-levels` and `stock-health` via `Promise.all`
   - Cleanup flag prevents stale async state when store changes mid-flight
5. **KPI cards section** (4 cards — Critical, Low, Healthy, Total Units):
   - Each card has a colored top-bar accent (red / amber / emerald / indigo)
   - Large colored numeric value, label, and subtext
   - `grid-cols-2 md:grid-cols-4` — 4 columns on desktop, 2 on iPad
6. **Stock Health donut chart** using Recharts `PieChart + Pie + Cell`:
   - Donut with DOM-overlay center label (total product count)
   - Right-side breakdown legend with color dots, counts, and percentages
   - Hover tooltip showing product count per status
7. **Skeleton loaders** — `KpiSkeleton` and `ChartSkeleton` mirror the real card layout and show during every loading phase (initial load and store switches)
8. **Error banners** — visible inline banners for store-list errors and store-data errors, styled cleanly without blocking the layout
9. **Reusable dashboard components** created in `src/components/dashboard/`:
   - `StoreSelector` — `<select>` with Indigo focus ring
   - `KpiCard` / `KpiSkeleton`
   - `SectionContainer` — consistent section title + children wrapper
   - `StockHealthChart` / `ChartSkeleton`

### Files created

| File | Purpose |
|---|---|
| `src/types/inventory.ts` | TypeScript types for all three backend API responses |
| `src/lib/inventory.ts` | `fetchStores`, `fetchStockLevels`, `fetchStockHealth` |
| `src/components/dashboard/StoreSelector.tsx` | Store dropdown control |
| `src/components/dashboard/KpiCard.tsx` | Colored KPI metric card |
| `src/components/dashboard/KpiSkeleton.tsx` | Animated skeleton for KPI cards |
| `src/components/dashboard/SectionContainer.tsx` | Section title wrapper |
| `src/components/dashboard/StockHealthChart.tsx` | Recharts donut chart + legend panel |
| `src/components/dashboard/ChartSkeleton.tsx` | Animated skeleton for chart section |

### Files modified

| File | Change |
|---|---|
| `src/pages/DashboardPage.tsx` | Full rewrite — live data, store switching, KPI cards, chart |
| `package.json` | Added `recharts` dependency |

### Packages added

| Package | Version |
|---|---|
| `recharts` | ^3.8.1 |

> **Install note:** This project's global npm config has `omit=dev`. Always install with `npm install --include=dev` to get devDependencies (TypeScript, Vite, ESLint, etc.).

### Backend routes used

| Route | Query param | Purpose |
|---|---|---|
| `GET /api/stores` | — | List of store IDs |
| `GET /api/stock-levels` | `store=S001` | Summary + per-product stock status |
| `GET /api/stock-health` | `store=S001` | Breakdown with counts and percentages |

All three routes require a Clerk JWT Bearer token.

### Loading / skeleton states

| Phase | What shows |
|---|---|
| Initial store list load | Skeleton store selector + 4 KPI skeletons + chart skeleton |
| Store data loading (initial + switch) | 4 KPI skeletons + chart skeleton |
| Data loaded | Real KPI cards + donut chart with legend |
| Store list error | Error banner, no selector, no data section |
| Store data error | Error banner below header, data section stays empty |

### Updated file tree

```
frontend/src/
├── App.tsx
├── index.css
├── main.tsx
├── app/
│   ├── AppShell.tsx
│   ├── AuthLayout.tsx
│   └── RootLayout.tsx
├── components/
│   ├── ProtectedRoute.tsx
│   └── dashboard/
│       ├── ChartSkeleton.tsx
│       ├── KpiCard.tsx
│       ├── KpiSkeleton.tsx
│       ├── SectionContainer.tsx
│       ├── StockHealthChart.tsx
│       └── StoreSelector.tsx
├── hooks/
│   └── useCurrentUser.ts
├── lib/
│   ├── api.ts
│   ├── config.ts
│   └── inventory.ts
├── pages/
│   ├── DashboardPage.tsx
│   ├── HomePage.tsx
│   ├── SignInPage.tsx
│   └── SignUpPage.tsx
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts
```

### How to verify this step

1. `cd frontend && npm install --include=dev && npm run dev`
2. Sign in → should land on `/dashboard`
3. **KPI cards** — 4 cards load with colored accents and real counts from the backend
4. **Donut chart** — loads with the health breakdown; hovering a slice shows tooltip
5. **Store switch** — change store in the top-right dropdown; KPI cards and chart should reload with new data immediately (skeletons show briefly during fetch)
6. **Skeleton states** — hard-refresh the page and observe skeleton cards before data arrives
7. **Error state** — temporarily set `VITE_API_BASE_URL` to an invalid URL and reload; error banners should appear without a blank screen

---

### Command to start the frontend

```bash
cd frontend
npm install --include=dev
npm run dev
```

App will be available at `http://localhost:5173`.
