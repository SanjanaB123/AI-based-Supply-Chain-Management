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

---

## Step 5 — Dashboard UI Redesign (Premium Foundation)

### What was done

1. **Introduced a dark left sidebar** (`src/app/AppShell.tsx`) replacing the plain top bar:
   - `bg-slate-900` sidebar, 240px wide, visible at `md` breakpoint and above
   - Stratos brand at top with "AI" micro-label in muted slate
   - Single nav item (Dashboard) with active state (`bg-white/10`) and hover state (`bg-white/5`)
   - Clerk `UserButton` + "Account" label anchored to the sidebar bottom
   - Mobile fallback: `md:hidden` top bar with logo and UserButton for screens below 768px
   - Main content area is `h-screen overflow-hidden` → `flex-1 overflow-auto` so only the page scrolls, sidebar stays fixed

2. **Redesigned DashboardPage layout** (`src/pages/DashboardPage.tsx`):
   - Page header uses `text-xl font-semibold tracking-tight text-slate-900` — confident but not oversized
   - Store selector moved into the header row as an integrated pill control (not a detached form)
   - Removed `SectionContainer` dependency — sections are inlined with `text-[11px] font-semibold uppercase tracking-widest text-slate-400` labels
   - KPI grid changed to `grid-cols-2 lg:grid-cols-4` (2-column at tablet, 4-column at desktop)
   - Error states upgraded: now include an `AlertIcon` SVG and use `rounded-xl` with `border border-red-100`
   - Explicit `{/* Future modules */}` comment block reserving space for tables, analytics tabs, chat

3. **Redesigned KpiCard** (`src/components/dashboard/KpiCard.tsx`):
   - Removed colored top bar; replaced with a 2×2 semantic color dot (`h-2 w-2 rounded-full`) in the card header
   - Value is now `text-3xl font-bold tracking-tight text-slate-900` — unified, readable, not colored
   - Card frame: `border border-slate-100 bg-white shadow-sm rounded-xl`
   - Subtext: `text-xs text-slate-400` — subdued, hierarchy preserved

4. **Redesigned KpiSkeleton** (`src/components/dashboard/KpiSkeleton.tsx`):
   - Matches new KpiCard layout exactly (dot placeholder, number block, subtext block)
   - Uses `bg-slate-100` (was `bg-gray-200`) for consistency with redesigned palette

5. **Redesigned StoreSelector** (`src/components/dashboard/StoreSelector.tsx`):
   - Wrapped in a bordered pill container: `rounded-lg border border-slate-200 bg-white px-3 py-2 shadow-sm`
   - Added a `PinIcon` SVG for context
   - "Store" label and dropdown sit inline within the pill — looks integrated, not form-like
   - Native `<select>` with no border, transparent background — system-native, accessible

6. **Redesigned StockHealthChart** (`src/components/dashboard/StockHealthChart.tsx`):
   - Card changed to `rounded-xl border border-slate-100 bg-white shadow-sm` (from bare `shadow-sm`)
   - Donut shrunk slightly (innerRadius 62, outerRadius 88) for better proportion in the smaller layout
   - Center label: `text-3xl font-bold tracking-tight` with `text-[11px] uppercase tracking-widest` subtext
   - Legend rows now include a two-line description (label + subtext per status)
   - Legend dividers use `divide-slate-50` — near-invisible, keeps it airy
   - Added a **health score footer**: a `bg-emerald-500` progress bar showing the healthy-stock percentage as a proportion — gives at-a-glance context without extra API calls
   - Tooltip styling refined: smaller font, softer shadow

7. **Redesigned ChartSkeleton** (`src/components/dashboard/ChartSkeleton.tsx`):
   - Matches new chart layout — donut circle, three legend rows with two-line placeholders, footer bar placeholder
   - Uses `border border-slate-100` to match the live card

8. **Improved base styles** (`src/index.css`):
   - Font stack: `-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif`
   - Added `-webkit-font-smoothing: antialiased` and `text-rendering: optimizeLegibility` for crisper type

### Files modified

| File | Change |
|---|---|
| `src/app/AppShell.tsx` | Full redesign — dark sidebar, mobile top bar, fixed-height shell |
| `src/pages/DashboardPage.tsx` | Layout upgrade — integrated store selector, inline sections, icon error states |
| `src/components/dashboard/KpiCard.tsx` | Premium redesign — dot indicator, unified number color, slate palette |
| `src/components/dashboard/KpiSkeleton.tsx` | Matches new KpiCard layout and palette |
| `src/components/dashboard/StoreSelector.tsx` | Pill container with PinIcon, integrated look |
| `src/components/dashboard/StockHealthChart.tsx` | Polished legend with subtexts, health score footer, refined chart sizing |
| `src/components/dashboard/ChartSkeleton.tsx` | Matches new chart card layout including footer |
| `src/index.css` | Better font stack + antialiasing |

### Files not modified

`SectionContainer.tsx` — kept as-is, no longer used by DashboardPage but available for future modules.

### Packages added

None. No new dependencies.

### Sidebar introduced

**Yes.** A `bg-slate-900` dark sidebar at 240px, visible at `md` (768px) and above. Mobile (below 768px) falls back to a compact top bar. The sidebar is intentionally minimal — only the Dashboard link is present. Future nav items (Analytics, Alerts, Settings) can be added to `NAV_ITEMS` in `AppShell.tsx`.

### Visual / layout changes summary

| Area | Before | After |
|---|---|---|
| Shell | White top bar, flat | Dark slate sidebar with brand, nav, user |
| Background | `bg-gray-50` | `bg-slate-50` (same visual weight, consistent palette) |
| KPI cards | Colored top bar + colored large number | White card + status dot + unified slate number |
| Store selector | Bare label + select | Pinned pill container — integrated into header |
| Chart card | No border, just shadow | `border border-slate-100` + footer health bar |
| Chart legend | Single-line rows | Two-line rows (label + subtext per status) |
| Skeletons | `bg-gray-200` blocks | `bg-slate-100` blocks matching new layout exactly |
| Error banners | Simple text | Icon + rounded-xl, softer styling |
| Section headers | `SectionContainer` title | Inline `text-[11px] uppercase tracking-widest` |

### Updated file tree

```
frontend/src/
├── App.tsx
├── index.css                        ← updated base font + antialiasing
├── main.tsx
├── app/
│   ├── AppShell.tsx                 ← redesigned: dark sidebar + mobile top bar
│   ├── AuthLayout.tsx
│   └── RootLayout.tsx
├── components/
│   ├── ProtectedRoute.tsx
│   └── dashboard/
│       ├── ChartSkeleton.tsx        ← matches new chart layout
│       ├── KpiCard.tsx              ← dot indicator, unified number color
│       ├── KpiSkeleton.tsx          ← matches new KpiCard
│       ├── SectionContainer.tsx     ← unchanged, available for future use
│       ├── StockHealthChart.tsx     ← polished legend, health score footer
│       └── StoreSelector.tsx        ← pill container, integrated look
├── hooks/
│   └── useCurrentUser.ts
├── lib/
│   ├── api.ts
│   ├── config.ts
│   └── inventory.ts
├── pages/
│   ├── DashboardPage.tsx            ← layout upgrade, inline sections
│   ├── HomePage.tsx
│   ├── SignInPage.tsx
│   └── SignUpPage.tsx
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts
```

### Manual steps required

None. No new env vars, no new packages, no Clerk dashboard changes.

### How to verify this step

1. `cd frontend && npm run dev`
2. Sign in → land on `/dashboard`
3. **Sidebar** — should see dark `slate-900` sidebar on the left with "Stratos AI" brand, "Dashboard" nav link highlighted, and UserButton at the bottom
4. **KPI cards** — 4 white cards in a 2-column (tablet) or 4-column (desktop) grid, each with a small colored dot (red/amber/green/indigo) and a large slate-900 number
5. **Store selector** — should appear in the page header as a bordered pill with a pin icon and the store name
6. **Chart card** — should have a bordered card, refined legend with two-line rows, and a health score progress bar at the bottom
7. **Skeleton states** — hard-refresh and observe skeletons before data arrives; layout should match the live card shapes
8. **Store switch** — change the store; KPI cards and chart reload with skeletons during fetch
9. **iPad mini width (768px)** — resize browser to 768px: sidebar stays visible, KPI cards go 2-column, chart stacks donut above legend
10. **Below 768px** — sidebar hides, a compact top bar appears with logo and UserButton
11. `npx tsc --noEmit` — zero TypeScript errors

---

## Step 6 — Full Dashboard Expansion + Live Analytics Modules

### What was done

1. **Expanded sidebar navigation** (`src/app/AppShell.tsx`):
   - Full nav structure with three sections: primary (Dashboard), Analytics (Analytics, Inventory, Risk, Variance), Intelligence (AI Assistant)
   - Settings item pinned above account at the bottom
   - Non-functional items visually marked with a "Soon" badge and `opacity-40 cursor-not-allowed` — believable product nav, no fake clutter
   - SVG icons for every nav item drawn inline (no icon library added)
   - Sidebar section labels (`ANALYTICS`, `INTELLIGENCE`) in micro-caps above each group

2. **Full-width dashboard layout** (`src/pages/DashboardPage.tsx`):
   - Removed `max-w-6xl` constraint — content now uses the full available width from AppShell padding
   - Below the KPI row: `xl:grid-cols-3` two-panel layout — health chart (`xl:col-span-1`, ~1/3 width) and analytics panel (`xl:col-span-2`, ~2/3 width) side by side at 1280px+
   - Below XL both panels stack full-width gracefully
   - All 6 backend routes fetched in a single `Promise.all` on store change

3. **New TypeScript types** (`src/types/inventory.ts`):
   - `SellThroughProduct`, `SellThroughResponse`
   - `DaysOfSupplyProduct`, `DaysOfSupplyThresholds`, `DaysOfSupplyResponse`
   - `LeadTimeRiskProduct`, `LeadTimeRiskResponse`
   - `ShrinkageProduct`, `ShrinkageResponse`

4. **New fetch functions** (`src/lib/inventory.ts`):
   - `fetchSellThrough(storeId, token)`
   - `fetchDaysOfSupply(storeId, token)`
   - `fetchLeadTimeRisk(storeId, token)`
   - `fetchShrinkage(storeId, token)`

5. **Analytics tab panel** — polished tab switcher (Sell-Through · Days of Supply · Lead-Time Risk · Variance) with `bg-slate-900` active state; switching tabs never triggers a refetch

6. **SellThroughModule** (`src/components/dashboard/SellThroughModule.tsx`):
   - Top/bottom performer callout cards (emerald / red)
   - Recharts horizontal `BarChart` (layout="vertical") with per-bar semantic colors (green ≥70%, amber ≥40%, red <40%)
   - `LabelList` renders the % value to the right of each bar
   - Custom tooltip showing product, category, sold/received counts
   - Capped at 12 products to keep the chart readable; footnote shows full count

7. **DaysOfSupplyModule** (`src/components/dashboard/DaysOfSupplyModule.tsx`):
   - Summary row showing critical/low counts with semantic dot indicators
   - Scrollable table (max-h-72) with per-row color banding (red/amber/neutral)
   - Inline progress bar per row scaled to max days of supply in dataset
   - Days remaining shown prominently in health-colored text
   - Threshold legend inline in the summary row

8. **LeadTimeRiskModule** (`src/components/dashboard/LeadTimeRiskModule.tsx`):
   - At-risk banner when any product has days_of_supply ≤ lead_time_days
   - Recharts `ScatterChart` with three separate `Scatter` series (critical/low/healthy)
   - `ReferenceLine` at 14d (critical) and 45d (low) with dash pattern and color labels
   - Built-in `Legend` auto-generated from series names
   - Custom tooltip shows product ID, category, both metrics, and stockout warning flag

9. **ShrinkageModule** (`src/components/dashboard/ShrinkageModule.tsx`):
   - Total inventory variance summary card at the top — framed as "variance", not "loss"
   - Positive = unaccounted units (amber); negative = data anomaly (muted); zero = emerald
   - Scrollable table (max-h-64) with received / sold / on-hand / variance columns
   - Footnote explains the variance formula clearly

10. **ModuleSkeleton** (`src/components/dashboard/ModuleSkeleton.tsx`):
    - Generic animated skeleton matching a bar-chart layout (callout row + bars + footer note)
    - Used as the placeholder for all 4 analytics modules while loading

11. **StockHealthChart updated** (`src/components/dashboard/StockHealthChart.tsx`):
    - Removed deprecated `Cell` import — recharts v3 reads `fill` from data objects directly
    - Layout changed from `md:flex-row` to always `flex-col` so it works cleanly in the narrow `xl:col-span-1` column
    - Donut sizing adjusted (innerRadius 58, outerRadius 84) for the compact vertical layout

12. **ChartSkeleton updated** (`src/components/dashboard/ChartSkeleton.tsx`):
    - Matches new always-vertical `StockHealthChart` layout (donut above, legend below)

### Files created

| File | Purpose |
|---|---|
| `src/components/dashboard/ModuleSkeleton.tsx` | Generic bar-chart-style skeleton for analytics modules |
| `src/components/dashboard/SellThroughModule.tsx` | Ranked horizontal bar chart with performer callouts |
| `src/components/dashboard/DaysOfSupplyModule.tsx` | Color-coded table with inline progress bars |
| `src/components/dashboard/LeadTimeRiskModule.tsx` | Scatter plot with threshold lines and at-risk banner |
| `src/components/dashboard/ShrinkageModule.tsx` | Inventory variance table with summary card |

### Files modified

| File | Change |
|---|---|
| `src/types/inventory.ts` | Added 4 new response types (SellThrough, DaysOfSupply, LeadTimeRisk, Shrinkage) |
| `src/lib/inventory.ts` | Added `fetchSellThrough`, `fetchDaysOfSupply`, `fetchLeadTimeRisk`, `fetchShrinkage` |
| `src/app/AppShell.tsx` | Full nav expansion — 3 sections, Soon badges, section labels, Settings bottom item |
| `src/pages/DashboardPage.tsx` | Full rewrite — removed max-width, 6-route parallel fetch, xl two-panel layout, analytics tabs |
| `src/components/dashboard/StockHealthChart.tsx` | Removed deprecated Cell, always flex-col, compact sizing |
| `src/components/dashboard/ChartSkeleton.tsx` | Matches new vertical-only StockHealthChart layout |

### Packages added

None. All new modules use Recharts (already installed).

### Backend routes now used

| Route | Query param | Purpose |
|---|---|---|
| `GET /api/stores` | — | Store list |
| `GET /api/stock-levels` | `store=S001` | KPI card counts |
| `GET /api/stock-health` | `store=S001` | Donut chart data |
| `GET /api/sell-through` | `store=S001` | Ranked sell-through bar chart |
| `GET /api/days-of-supply` | `store=S001` | Days-remaining table with thresholds |
| `GET /api/lead-time-risk` | `store=S001` | Scatter plot (lead time vs supply) |
| `GET /api/shrinkage` | `store=S001` | Inventory variance table |

All routes require Clerk JWT Bearer token.

### New dashboard modules

| Module | Tab label | Chart type | Key insight |
|---|---|---|---|
| SellThroughModule | Sell-Through | Horizontal bar chart | Ranks products by sell-through rate; flags top/bottom performers |
| DaysOfSupplyModule | Days of Supply | Color-coded table | Highlights imminent stockouts; threshold legend inline |
| LeadTimeRiskModule | Lead-Time Risk | Scatter chart | Identifies items that will stock out before restock arrives |
| ShrinkageModule | Variance | Data table | Surfaces unaccounted inventory; framed neutrally as variance |

### Loading states added

| State | What shows |
|---|---|
| Store list loading | Skeleton store selector pill |
| Store data loading (any fetch) | 4 KPI skeletons + ChartSkeleton + ModuleSkeleton in analytics panel |
| Store switch | All data cleared immediately; skeletons fill in until new data arrives |
| Data loaded | All live modules active; tab switching is instant (no re-fetch) |
| Data error | Red error banner; modules show `EmptyModule` fallback |

### Sidebar/navigation changes

- Section 1 (no label): Dashboard (active, functional)
- Section 2 (label: ANALYTICS): Analytics, Inventory, Risk, Variance — all marked "Soon"
- Section 3 (label: INTELLIGENCE): AI Assistant — marked "Soon"
- Bottom: Settings ("Soon") above Account (UserButton)

### Updated file tree

```
frontend/src/
├── App.tsx
├── index.css
├── main.tsx
├── app/
│   ├── AppShell.tsx                      ← expanded sidebar nav, section labels, Soon badges
│   ├── AuthLayout.tsx
│   └── RootLayout.tsx
├── components/
│   ├── ProtectedRoute.tsx
│   └── dashboard/
│       ├── ChartSkeleton.tsx             ← updated for vertical-only layout
│       ├── DaysOfSupplyModule.tsx        ← NEW: days-remaining table
│       ├── KpiCard.tsx
│       ├── KpiSkeleton.tsx
│       ├── LeadTimeRiskModule.tsx        ← NEW: scatter chart + at-risk banner
│       ├── ModuleSkeleton.tsx            ← NEW: generic analytics skeleton
│       ├── SectionContainer.tsx
│       ├── SellThroughModule.tsx         ← NEW: horizontal bar chart + callouts
│       ├── ShrinkageModule.tsx           ← NEW: variance table + summary card
│       ├── StockHealthChart.tsx          ← updated: no Cell, always flex-col
│       └── StoreSelector.tsx
├── hooks/
│   └── useCurrentUser.ts
├── lib/
│   ├── api.ts
│   ├── config.ts
│   └── inventory.ts                      ← added 4 new fetch functions
├── pages/
│   ├── DashboardPage.tsx                 ← full rewrite: wide layout, 6-route fetch, tabs
│   ├── HomePage.tsx
│   ├── SignInPage.tsx
│   └── SignUpPage.tsx
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts                      ← added 4 new response types
```

### Manual steps required

None. No new env vars, no new npm packages, no Clerk dashboard changes.

### How to verify this step

1. `cd frontend && npm install --include=dev && npm run dev`
2. Sign in → land on `/dashboard`
3. **Sidebar** — verify 3 nav sections with section labels; "Soon" badge on all non-Dashboard items; Settings above Account at bottom
4. **Desktop layout (≥1280px)** — health chart and analytics panel should appear side-by-side (1/3 and 2/3 width); KPI cards 4-across
5. **KPI cards** — all 4 load with live counts; store switch reloads them
6. **Health chart** — donut above legend (always vertical); health score footer shows correct %
7. **Analytics tabs** — switch between Sell-Through / Days of Supply / Lead-Time Risk / Variance; each should load live data
8. **Sell-Through tab** — horizontal bar chart with green/amber/red bars; top/bottom performer callouts visible
9. **Days of Supply tab** — sorted table (lowest days first); row colors match health status; progress bars present
10. **Lead-Time Risk tab** — scatter plot with critical/low/healthy series; red and amber reference lines at 14d and 45d; at-risk banner if applicable
11. **Variance tab** — total variance summary card at top; table with Received/Sold/On Hand/Variance columns; footnote explains the formula
12. **Loading skeletons** — hard-refresh; all sections show skeletons before data arrives
13. **Store switch** — change store; all data reloads simultaneously; no stale data visible during load
14. **Below 1280px (xl)** — health chart and analytics panel stack vertically (full width each)
15. **iPad mini (768px)** — sidebar visible; KPI cards 2-column; everything usable

---

## Step 7 — Dashboard Densification & Shell Refinement

### What was done

**Shell refinements:**
- Added a dedicated desktop top bar (`bg-slate-800`) — one tone lighter than the sidebar (`bg-slate-900`) to give the shell visual depth
- Top bar contains the store selector (injected via context), notification bell, inbox icon, and UserButton — all on the right
- Sidebar section label text updated from near-invisible `text-slate-600` to `text-slate-400` for legibility
- Active sidebar nav item changed from `bg-white/10` to `bg-blue-500/20 text-blue-100` (icon: `text-blue-300`) for a lighter blue selected state
- Inactive nav item text changed from `text-slate-400` to `text-slate-300` for better contrast
- Added **Help & Support** section to sidebar: User Guide, FAQs, Contact Support — all marked "Soon"

**Color system hardened:**
- Critical: `#ef4444` / `bg-red-500` / `text-red-*` — applied consistently in chips, bars, dots, row tints, banners
- Low/Warning: `#f59e0b` / `bg-amber-500` / `text-amber-*` — applied consistently
- Healthy/Good: `#10b981` / `bg-emerald-500` / `text-emerald-*` — applied consistently
- All new components follow the same semantic color mapping as existing modules

**Store selector moved to desktop top bar:**
- `TopBarContext.tsx` — new React context with a slot pattern; pages inject elements into the top bar
- `DashboardPage` injects a dark-variant `StoreSelector` into the top bar via `useTopBar()` and a `useEffect` that tracks store state
- Mobile fallback: a `md:hidden` store selector in the page header ensures mobile users still see it

**New lower-grid modules (all use existing fetched data — no new API calls):**

| Module | Data source | What it shows |
|---|---|---|
| `CriticalItemsTable` | `DaysOfSupply` + `LeadTimeRisk` | All critical/low products sorted by days remaining; includes lead time and stockout risk flag |
| `InsightCards` | `SellThrough` + `DaysOfSupply` + `Shrinkage` | 4 derived insights: top performer, urgent stockout, variance anomaly, best stock position |
| `CategoryBreakdown` | `StockLevels` | Per-category stacked health bar (critical/low/healthy) with counts |
| `RiskSpotlightPanel` | `LeadTimeRisk` | Products where days_of_supply ≤ lead_time_days; shows supply/lead time gap |
| `VarianceHighlights` | `Shrinkage` | Top 8 products by absolute variance magnitude; positive surplus vs. anomaly distinguished |

**Dashboard layout:**
- Page now has 4 sections: KPI row → Health chart + Analytics tabs → Urgent Items + Insights → Risk + Category + Variance
- Lower grid uses `xl:grid-cols-3` matching the existing analytics section for visual consistency
- All new modules have dedicated loading skeletons: `SmallPanelSkeleton`, `InsightSkeleton`, `CategorySkeleton`

### Files created

| File | Description |
|---|---|
| `src/app/TopBarContext.tsx` | React context for injecting elements into the desktop top bar |
| `src/components/dashboard/CriticalItemsTable.tsx` | Urgent products table (critical + low) |
| `src/components/dashboard/InsightCards.tsx` | 4-card derived insight stack |
| `src/components/dashboard/CategoryBreakdown.tsx` | Per-category stacked health bar |
| `src/components/dashboard/RiskSpotlightPanel.tsx` | Stockout-before-restock risk list |
| `src/components/dashboard/VarianceHighlights.tsx` | Extreme variance item list |

### Files modified

| File | Change |
|---|---|
| `src/app/AppShell.tsx` | Desktop top bar, sidebar active state, section labels, Help & Support section, TopBarProvider wrapper |
| `src/components/dashboard/StoreSelector.tsx` | Added `variant` prop (`'light' \| 'dark'`) |
| `src/pages/DashboardPage.tsx` | Top bar slot injection, lower grid, inline skeletons, SectionLabel/ModuleCard helpers |

### Updated file tree

```
frontend/src/
├── App.tsx
├── index.css
├── main.tsx
├── app/
│   ├── AppShell.tsx                           ← top bar, sidebar refinements, TopBarProvider
│   ├── AuthLayout.tsx
│   ├── RootLayout.tsx
│   └── TopBarContext.tsx                      ← NEW: slot context for top bar injection
├── components/
│   ├── ProtectedRoute.tsx
│   └── dashboard/
│       ├── CategoryBreakdown.tsx              ← NEW: stacked category health bar
│       ├── ChartSkeleton.tsx
│       ├── CriticalItemsTable.tsx             ← NEW: urgent items table (DOS + LTR data)
│       ├── DaysOfSupplyModule.tsx
│       ├── InsightCards.tsx                   ← NEW: derived insight cards
│       ├── KpiCard.tsx
│       ├── KpiSkeleton.tsx
│       ├── LeadTimeRiskModule.tsx
│       ├── ModuleSkeleton.tsx
│       ├── RiskSpotlightPanel.tsx             ← NEW: stockout-before-restock spotlight
│       ├── SellThroughModule.tsx
│       ├── ShrinkageModule.tsx
│       ├── StockHealthChart.tsx
│       ├── StoreSelector.tsx                  ← added dark variant
│       └── VarianceHighlights.tsx             ← NEW: extreme variance list
├── hooks/
│   └── useCurrentUser.ts
├── lib/
│   ├── api.ts
│   ├── config.ts
│   └── inventory.ts
├── pages/
│   ├── DashboardPage.tsx                      ← lower grid, top bar slot, new sections
│   ├── HomePage.tsx
│   ├── SignInPage.tsx
│   └── SignUpPage.tsx
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts
```

### Packages added

None. All new functionality uses existing dependencies (React, Tailwind, Recharts, Clerk).

### Manual steps required

None. No new env vars, no Clerk changes, no new npm packages.

### How to verify this step

1. `cd frontend && npm run dev` → open `/dashboard`
2. **Desktop top bar** — should be `bg-slate-800` (visibly lighter than the sidebar); store selector appears on the left, bell + inbox + user button on the right
3. **Sidebar active state** — Dashboard item should have a soft blue highlight (not white); section labels should be legible
4. **Help & Support section** — should appear in sidebar with User Guide, FAQs, Contact Support items (all "Soon")
5. **Store selector in top bar** — changing the store via the top bar store selector should trigger all data to reload
6. **Mobile (< 768px)** — store selector appears in page header; desktop top bar is hidden; mobile top bar shows brand + user button
7. **KPI row + health chart + analytics tabs** — all still function as before
8. **Urgent Items section** — table shows critical/low products sorted by days remaining; Lead Time and Exposure columns visible on wide screens
9. **Operational Insights** — 4 insight cards (Top Performer, Urgent Stockout, Variance Anomaly, Best Stock Position) derived from live data
10. **Risk Spotlight** — if any products have days_of_supply ≤ lead_time_days, they appear here with gap calculation; otherwise shows green "no risk" state
11. **Category Breakdown** — each product category shows a stacked health bar with red/amber/green segments
12. **Variance Highlights** — top 8 products by absolute variance magnitude, positive values shown in amber, negative in muted gray
13. **Loading skeletons** — hard-refresh; all 5 new lower panels show skeleton placeholders before data arrives
14. **Store switch** — switching stores clears and reloads all panels; no stale data visible
15. **Zero TypeScript errors** — `npx tsc --noEmit` exits 0
16. `npx tsc --noEmit` — zero TypeScript errors

---

## Step 8 — Dashboard Polish & Chart Refinement Pass

### What was done

1. **Shell redesign — light-mode system**
   - Sidebar switched from dark (`bg-slate-900`) to a cool light-gray surface (`bg-slate-100`) with a `border-r border-slate-200`
   - Top bar switched from dark (`bg-slate-800`) to white (`bg-white`) with `border-b border-slate-200`
   - Main content area retains `bg-slate-50`; cards remain `bg-white` — clear three-level depth
   - All icon/text colors in the sidebar flipped to dark-on-light

2. **Top bar redesign — true page header**
   - Page title ("Inventory Overview") and subtitle now live in the top bar left side (injected via extended `TopBarContext`)
   - Store selector moved inline with the title (still injected by `DashboardPage` via slot)
   - Search bar added (non-functional UI, ready for wiring)
   - Notification bell + inbox icons retained with light-mode hover states
   - `UserButton` stays at the far right
   - Page body now contains only dashboard module rows — no duplicate header

3. **Sidebar polish**
   - Active nav item: `bg-blue-100 text-blue-700` (lighter, cleaner selection vs old `bg-blue-500/20`)
   - Inactive: `text-slate-600 hover:bg-slate-200/60` — legible on light surface
   - Section labels: same `text-slate-400` uppercase
   - Brand "AI" badge changed from `text-slate-500` → `text-blue-500`
   - "Soon" badge updated to `ring-slate-300` (appropriate for light bg)

4. **Sidebar account section**
   - Replaced generic "Account" label with real user data via Clerk `useUser()`
   - Shows full name (bold, `text-slate-800`) + email (smaller, `text-slate-400`)
   - Falls back to username or "User" if name is not set

5. **Semantic palette applied consistently**
   - Critical = `red-500`, Low = `amber-500`, Healthy = `emerald-500`
   - KPI cards now have a colored accent bar across the top (`h-0.5`) + colored label text matching the variant
   - InsightCards borders upgraded from `*-100` to `*-200` for better depth
   - All existing chart colors remain correct

6. **Responsive typography scaling**
   - KPI values: `text-3xl xl:text-4xl`
   - KPI subtext: `text-xs xl:text-[13px]`
   - KPI label: `text-[10px] xl:text-[11px]`
   - Section labels: `xl:text-[11px]`
   - Top bar title: `xl:text-base`; subtitle: `xl:text-[12px]`
   - StockHealthChart center count: `xl:text-4xl`

7. **Analytics tab bar — premium pill style**
   - Active tab: `bg-blue-600 text-white shadow-sm` (solid blue pill)
   - Inactive: `text-slate-500 hover:bg-slate-200/70 hover:text-slate-800`
   - Tab bar row: `bg-slate-50/60` tint to differentiate from card body
   - Removed old `bg-slate-900` active style

8. **SellThroughModule — ranked performance list (full redesign)**
   - Replaced Recharts `BarChart` entirely
   - Now: rank number + product + category + progress track + % label
   - Health color logic identical: emerald ≥70%, amber 40–70%, red <40%
   - Top Performer / Needs Attention callout cards retained and refined
   - Column headers added; rows have hover state
   - Cleaner, faster to scan, no chart rendering overhead

9. **LeadTimeRiskModule — dual-bar risk timeline (full redesign)**
   - Replaced Recharts `ScatterChart` entirely
   - Now: per-product rows sorted by risk (supply − lead_time gap ascending)
   - Each row: product name + category chip, risk badge (AT RISK / +Nd buffer), dual bars
   - Supply bar colored by health (red/amber/emerald); lead time bar always blue
   - At-risk rows have `bg-red-100/60 border-red-200` background; borderline rows amber tint
   - Legend: supply bar color + lead time bar color
   - Footer: explains when supply bar is shorter than lead time = at risk
   - Shows top 8 products by risk priority; all live data preserved

10. **Card border polish**
    - All cards switched from `border-slate-100` → `border-slate-200/80` for better definition against the `bg-white` surface
    - KpiCard: added overflow-hidden for the accent bar clip

11. **Layout consistency**
    - Row sections use `xl:items-start` to prevent forced equal heights across uneven modules
    - `gap-6` → `gap-5` in section grids for tighter professional spacing
    - All `space-y-7` → `space-y-6` in page-level container

12. **TopBarContext extended**
    - Added `pageTitle`, `pageSubtitle`, `setPageMeta` to the context interface
    - Pages call `setPageMeta(title, subtitle)` in `useEffect` with cleanup
    - AppShell reads these and renders in the top bar left

---

### Files created / modified

| File | Change |
|---|---|
| `src/app/TopBarContext.tsx` | Added `pageTitle`, `pageSubtitle`, `setPageMeta` to context |
| `src/app/AppShell.tsx` | Full shell redesign: light sidebar, new top bar layout, user info via `useUser()` |
| `src/pages/DashboardPage.tsx` | Removed page header; added `setPageMeta`; updated store selector to `variant="light"`; premium tab bar |
| `src/components/dashboard/KpiCard.tsx` | Accent bar, colored label, responsive typography |
| `src/components/dashboard/SellThroughModule.tsx` | Full redesign: ranked list with progress tracks |
| `src/components/dashboard/LeadTimeRiskModule.tsx` | Full redesign: dual-bar risk timeline |
| `src/components/dashboard/StockHealthChart.tsx` | Border polish, responsive center count, spacing tweaks |
| `src/components/dashboard/InsightCards.tsx` | Border upgrade (`*-200`), tighter spacing |
| `src/components/dashboard/StoreSelector.tsx` | Light variant updated for new top bar |

---

### Updated frontend tree (Step 4 state)

```
frontend/src/
├── App.tsx
├── main.tsx
├── index.css
├── app/
│   ├── AppShell.tsx                   ← redesigned: light shell, real user info
│   ├── AuthLayout.tsx
│   ├── RootLayout.tsx
│   └── TopBarContext.tsx              ← extended: pageTitle, pageSubtitle, setPageMeta
├── components/
│   ├── ProtectedRoute.tsx
│   └── dashboard/
│       ├── CategoryBreakdown.tsx
│       ├── ChartSkeleton.tsx
│       ├── CriticalItemsTable.tsx
│       ├── DaysOfSupplyModule.tsx
│       ├── InsightCards.tsx           ← border polish
│       ├── KpiCard.tsx                ← accent bar, responsive type
│       ├── KpiSkeleton.tsx
│       ├── LeadTimeRiskModule.tsx     ← REDESIGNED: dual-bar risk timeline
│       ├── ModuleSkeleton.tsx
│       ├── RiskSpotlightPanel.tsx
│       ├── SectionContainer.tsx
│       ├── SellThroughModule.tsx      ← REDESIGNED: ranked performance list
│       ├── ShrinkageModule.tsx
│       ├── StockHealthChart.tsx       ← border polish, responsive
│       ├── StoreSelector.tsx          ← light variant updated
│       └── VarianceHighlights.tsx
├── hooks/
│   └── useCurrentUser.ts
├── lib/
│   ├── api.ts
│   ├── config.ts
│   └── inventory.ts
├── pages/
│   ├── DashboardPage.tsx              ← no page header; setPageMeta; light tab bar
│   ├── HomePage.tsx
│   ├── SignInPage.tsx
│   └── SignUpPage.tsx
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts
```

---

### Packages added

None. All changes use existing dependencies (React, Tailwind CSS, Recharts, `@clerk/clerk-react`).

---

### Manual steps required

None. No new env vars, no new npm packages, no Clerk configuration changes.

---

### How to verify this step

1. `cd frontend && npm run dev` → open `/dashboard`
2. **Shell** — sidebar is light cool-gray (`bg-slate-100`), not dark; top bar is white; main content is `bg-slate-50`
3. **Top bar** — left side shows "Inventory Overview" (bold) + subtitle; store selector appears inline; search bar in center-right; bell + inbox + user on far right
4. **Sidebar active state** — Dashboard has a soft blue pill (`bg-blue-100 text-blue-700`), not dark
5. **Sidebar user section** — shows your Clerk display name (bold) + email address (smaller, gray)
6. **KPI cards** — each has a colored horizontal accent bar at the top matching its semantic color; values scale up at `xl:` viewport
7. **Analytics tabs** — active tab is a solid blue pill (`bg-blue-600`); tab bar has a slate-50 tint background
8. **Sell-Through tab** — shows a ranked list with numbered rows, progress bars, and color-coded `%` labels; no Recharts bar chart
9. **Lead-Time Risk tab** — shows per-product rows sorted by risk, each with two horizontal bars (supply in green/amber/red, lead time in blue); at-risk rows are red-tinted; "AT RISK" badge visible for stockout items
10. **Semantic colors** — critical=red, low=amber, healthy=emerald, applied in KPI accents, chips, banners, progress bars
11. **Responsive typography** — at viewport ≥ 1280px (`xl`), KPI values, top bar title, and section labels visibly increase in size
12. **Store switching** — changing store reloads all data; no stale state
13. **Mobile (< 768px)** — layout degrades gracefully; store selector visible in page area; top bar shows brand + user
14. **Zero TypeScript errors** — `npx tsc --noEmit` exits 0

---

## Step 9 — Strategic Analytics Upgrade: Heatmap + Trend Chart

### What was done

1. **Replaced `Operational Insights` module with `InventoryHeatmapModule`**
   - The previous four-card `InsightCards` panel was removed from the dashboard layout.
   - A new **GitHub-style inventory health density grid** (`InventoryHeatmapModule.tsx`) takes its place in the same `xl:col-span-1` slot.
   - `InsightCards.tsx` remains on disk but is no longer imported or rendered.

2. **Added a new full-width `InventoryTrendChart` section**
   - Inserted as a standalone full-width section under KPI cards section, improving page rhythm and information density.
   - Uses a Recharts `AreaChart` with three stacked areas (critical / low / healthy).
   - Five timeframe filter pills: **Last week · Last 30 days · Last 3 mos · Last 6 mos · Last 12 mos**.

3. **Preserved all manual developer improvements**
   - Same-row equal heights (`h-86` fixed on heatmap) — unchanged.
   - Sidebar background matching topbar — unchanged.
   - Active nav state (`bg-blue-200 text-blue-700`) — unchanged.
   - All `*-100` tint upgrades from previous step — unchanged.
   - Light shell and semantic color palette — unchanged.

---

### Files created

| File | Description |
|---|---|
| `src/components/dashboard/InventoryHeatmapModule.tsx` | GitHub-style product health density grid |
| `src/components/dashboard/InventoryTrendChart.tsx` | Full-width stacked area chart with timeframe filters |

### Files modified

| File | Change |
|---|---|
| `src/pages/DashboardPage.tsx` | Replaced `InsightCards` import + usage with `InventoryHeatmapModule`; added full-width `InventoryTrendChart` section with matching skeleton; updated section label from "Operational Insights" → "Inventory Density" |

---

### Packages added

None. Both new components use Recharts (already installed at `^3.8.1`) and Tailwind CSS.

---

### Module: InventoryHeatmapModule — how it works

- **Data source:** `DaysOfSupplyResponse` (already fetched on store load, no new API call).
- **Layout:** Products are rendered as small 14 × 14 px colored tiles, grouped by category into rows. Category labels appear to the left of each row.
- **Color encoding:**
  | Color | Days of supply remaining |
  |---|---|
  | `bg-red-600` (deep red) | ≤ 7 days |
  | `bg-red-400` (medium red) | 8 – 14 days |
  | `bg-amber-400` (amber) | 15 – 30 days |
  | `bg-emerald-300` (light green) | 31 – 60 days |
  | `bg-emerald-500` (green) | > 60 days |
- **Sorting:** Categories are sorted by descending critical-item count (most urgent categories appear first). Within each row, products are sorted by ascending days of supply (most critical tiles appear leftmost).
- **Interactivity:** Hovering a tile shows an info bar above the grid with the product ID, days remaining, and category.
- **Legend:** A compact "Supply" legend at the bottom maps each color to a day-range bucket.
- **Summary counters:** Critical / low / healthy product counts displayed in the header.

---

### Module: InventoryTrendChart — how it works

- **Data source:** `DaysOfSupplyResponse` (same fetch, no new API call).
- **Derivation logic:** Since the backend does not expose time-series history, the chart projects backwards from the current snapshot using a constant-velocity assumption:
  - For each historical point *T* days ago, each product's estimated days-of-supply is `current_dos + T`.
  - Products are re-classified (critical / low / healthy) at each point using the live `thresholds.critical_below` and `thresholds.low_below` values.
  - Aggregated counts per status are plotted over time.
  - **Narrative:** Further in the past → more products were healthy. Approaching today → more products are low or critical. This reflects real inventory drawdown.
- **Chart type:** Recharts `AreaChart` with `stackId="stack"` — three stacked semi-transparent fills (critical at the bottom, then low, then healthy at the top). Total height = constant product count; the composition shifts over time.
- **Timeframe filters:**
  | Label | Range | Data points |
  |---|---|---|
  | Last week | 7 days | 7 daily points |
  | Last 30 days | 30 days | 10 evenly spaced |
  | Last 3 mos | 90 days | 13 evenly spaced |
  | Last 6 mos | 180 days | 12 evenly spaced |
  | Last 12 mos | 365 days | 13 evenly spaced |
- **Labeling:** Chart is clearly labelled "Stock Health Over Time · Projected from current daily sales velocity" so the derived nature of the data is transparent.
- **Custom tooltip:** Shows the breakdown of critical / low / healthy counts plus total for each hovered data point.

---

### Dashboard composition after this step

| # | Section | Width | Module(s) |
|---|---|---|---|
| 1 | Stock Status | Full | KPI row (4 cards) |
| 2 | **Inventory Trend** | **Full (new)** | **InventoryTrendChart with 5 timeframe filters** |
| 3 | Health + Analytics | 1/3 + 2/3 | StockHealthChart + tabbed analytics panel |
| 4 | Urgent + Density | 2/3 + 1/3 | CriticalItemsTable + **InventoryHeatmapModule** |
| 5 | Lower grid | 1/3 each | RiskSpotlightPanel + CategoryBreakdown + VarianceHighlights |

---

### Updated frontend tree

```
frontend/src/
├── App.tsx
├── main.tsx
├── index.css
├── app/
│   ├── AppShell.tsx
│   ├── AuthLayout.tsx
│   ├── RootLayout.tsx
│   └── TopBarContext.tsx
├── components/
│   ├── ProtectedRoute.tsx
│   └── dashboard/
│       ├── CategoryBreakdown.tsx
│       ├── ChartSkeleton.tsx
│       ├── CriticalItemsTable.tsx
│       ├── DaysOfSupplyModule.tsx
│       ├── InsightCards.tsx              ← retained on disk, no longer rendered
│       ├── InventoryHeatmapModule.tsx    ← NEW: GitHub-style health density grid
│       ├── InventoryTrendChart.tsx       ← NEW: full-width stacked area chart
│       ├── KpiCard.tsx
│       ├── KpiSkeleton.tsx
│       ├── LeadTimeRiskModule.tsx
│       ├── ModuleSkeleton.tsx
│       ├── RiskSpotlightPanel.tsx
│       ├── SectionContainer.tsx
│       ├── SellThroughModule.tsx
│       ├── ShrinkageModule.tsx
│       ├── StockHealthChart.tsx
│       ├── StoreSelector.tsx
│       └── VarianceHighlights.tsx
├── hooks/
│   └── useCurrentUser.ts
├── lib/
│   ├── api.ts
│   ├── config.ts
│   └── inventory.ts
├── pages/
│   ├── DashboardPage.tsx                ← heatmap wired; trend section added
│   ├── HomePage.tsx
│   ├── SignInPage.tsx
│   └── SignUpPage.tsx
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts
```

---

### Manual steps required

None. No new env vars, no new npm packages, no Clerk configuration changes.

---

### How to verify this step

1. `cd frontend && npm run dev` → open `/dashboard`
2. **Inventory Trend section** — a full-width card labelled "Inventory Trend" should appear under the KPI cards row.
3. **Trend chart** — shows a stacked area chart with three color bands (red = critical at bottom, amber = low, green = healthy at top).
4. **Timeframe filters** — five pill buttons: "Last week · Last 30 days · Last 3 mos · Last 6 mos · Last 12 mos". Clicking each changes the X-axis range and label format.
5. **Trend direction** — further left (older) should show more green (healthy); rightmost point (today) matches current critical/low/healthy counts from the KPI cards.
6. **Trend tooltip** — hover over any data point; tooltip shows critical / low / healthy breakdown + total.
7. **Inventory Density section** — in the 2/3 + 1/3 row (right column), the old "Operational Insights" card stack is gone; a labeled "Inventory Density" heatmap grid appears.
8. **Heatmap tiles** — small colored squares grouped by category; each row has a category label on the left; deep red tiles appear in categories with critical items.
9. **Heatmap hover** — hovering a tile shows the product ID, days remaining, and category in the info bar above the grid.
10. **Heatmap legend** — bottom of the card shows a color key: ≤7d, ≤14d, ≤30d, ≤60d, 60d+.
11. **Heatmap summary** — header shows total critical / low / healthy counts for quick orientation.
12. **Loading skeletons** — hard-refresh; the trend card shows an animated multi-element skeleton; the density card shows the existing `InsightSkeleton` placeholder.
13. **Store switch** — changing the store reloads all panels including heatmap and trend chart.
14. **Shell unchanged** — light sidebar, white topbar, `bg-blue-200 text-blue-700` active nav state, equal row heights all preserved.
15. **Zero TypeScript errors** — `npx tsc --noEmit` exits 0.

---

## Step 10 — Premium Split-Layout Auth Redesign

### What was done

Redesigned the Sign-In and Sign-Up pages from a bare centered-card layout into a polished, branded split-screen auth experience.

1. **`AuthLayout.tsx` — complete redesign (split-screen shell)**
   - Left panel (`md:w-[46%] lg:w-[48%] xl:w-[50%]`, hidden below `md`):
     - `auth-bg.png` used as a full-bleed background via CSS `background-image`
     - Two layered overlays: a dark base (`bg-slate-950/70`) + a blue-pooling gradient (`bg-gradient-to-t from-blue-950/55`) for depth and text legibility
     - Stratos wordmark (white, with blue "AI" micro-label) anchored to the top-left — links back to `/` via `tabIndex={-1}` (kept out of keyboard focus while in auth flow)
     - Value proposition block: small-caps category label, headline in `text-[28px] xl:text-[34px]`, body copy, and three stat callouts (94% forecast accuracy, 12ms alert latency, 3.4× faster decisions)
     - Copyright footer anchored to the bottom-left
   - Right panel (`flex-1`, full-width on mobile):
     - `bg-white` — clean, high-contrast against the dark left side
     - Clerk form constrained to `max-w-[400px]` and centered
     - Mobile-only Stratos brand header (visible only below `md` breakpoint) with the same wordmark + tagline
     - Bottom copyright footnote
   - Entire layout is `min-h-screen` and `flex` — no scrollbar on the auth pages

2. **`src/lib/clerkAppearance.ts` — new shared Clerk appearance config**
   - `variables`: colorPrimary `#1e40af` (blue-800), matching `bg-slate-50` input background, `#0f172a` text, `borderRadius: 0.5rem`
   - `elements`: card shadow/border stripped so the form sits flush on the white panel; social buttons, divider, form field labels/inputs, primary button, footer links, identity preview, and alert text all aligned to the Stratos design language
   - Exported as a single default object consumed by both page components

3. **`SignInPage.tsx` and `SignUpPage.tsx` — appearance prop added**
   - Both components import `clerkAppearance` and pass it as `appearance={clerkAppearance}`
   - All existing props preserved: `routing="path"`, `path`, `signUpUrl`/`signInUrl`, `fallbackRedirectUrl="/dashboard"`

### Files created

| File | Purpose |
|---|---|
| `src/lib/clerkAppearance.ts` | Shared Clerk `appearance` config object — variables + element class overrides |

### Files modified

| File | Change |
|---|---|
| `src/app/AuthLayout.tsx` | Full redesign — split-screen shell with branded left panel and clean right form panel |
| `src/pages/SignInPage.tsx` | Added `appearance={clerkAppearance}` prop |
| `src/pages/SignUpPage.tsx` | Added `appearance={clerkAppearance}` prop |

### Packages added

None. No new dependencies.

### Auth layout changes

| Aspect | Before | After |
|---|---|---|
| Layout | Centered column on `bg-gray-50` | Full-screen split: dark branded left / white form right |
| Brand | Inline `<h1>Stratos</h1>` above Clerk card | Wordmark + "AI" badge on left panel; mobile header on right |
| Background | Flat gray | `auth-bg.png` behind a dark overlay + blue gradient on left |
| Value prop | Two-line tagline | Headline + body copy + three stat callouts |
| Clerk card | Default floating card | Shadow/border stripped; form sits flush on white panel |
| Responsiveness | Always centered | Left panel hidden below `md`; right panel full-width on mobile |

### How Clerk styling was preserved

- `<SignIn>` and `<SignUp>` remain the actual auth engine — no custom form logic
- `routing`, `path`, `signUpUrl`/`signInUrl`, and `fallbackRedirectUrl` props are unchanged
- `ClerkProvider` in `main.tsx` is untouched
- `ProtectedRoute` and all redirect behavior are untouched
- The `appearance` prop only adds Tailwind classes and CSS variables on top of Clerk's built-in structure

### Responsiveness

| Breakpoint | Behavior |
|---|---|
| `< 768px` (mobile) | Left panel hidden; right panel full-width; mobile Stratos header visible |
| `≥ 768px` (md — tablet/iPad mini) | Split layout active; left panel at 46% width |
| `≥ 1024px` (lg) | Left panel 48% |
| `≥ 1280px` (xl) | Left panel 50%; headline scales to `text-[34px]` |

### Updated frontend tree

```
frontend/src/
├── App.tsx
├── main.tsx
├── index.css
├── app/
│   ├── AppShell.tsx
│   ├── AuthLayout.tsx                ← REDESIGNED: split-screen branded layout
│   ├── RootLayout.tsx
│   └── TopBarContext.tsx
├── components/
│   ├── ProtectedRoute.tsx
│   └── dashboard/
│       ├── CategoryBreakdown.tsx
│       ├── ChartSkeleton.tsx
│       ├── CriticalItemsTable.tsx
│       ├── DaysOfSupplyModule.tsx
│       ├── InsightCards.tsx
│       ├── InventoryHeatmapModule.tsx
│       ├── InventoryTrendChart.tsx
│       ├── KpiCard.tsx
│       ├── KpiSkeleton.tsx
│       ├── LeadTimeRiskModule.tsx
│       ├── ModuleSkeleton.tsx
│       ├── RiskSpotlightPanel.tsx
│       ├── SectionContainer.tsx
│       ├── SellThroughModule.tsx
│       ├── ShrinkageModule.tsx
│       ├── StockHealthChart.tsx
│       ├── StoreSelector.tsx
│       └── VarianceHighlights.tsx
├── hooks/
│   └── useCurrentUser.ts
├── lib/
│   ├── api.ts
│   ├── clerkAppearance.ts            ← NEW: shared Clerk appearance config
│   ├── config.ts
│   └── inventory.ts
├── pages/
│   ├── DashboardPage.tsx
│   ├── HomePage.tsx
│   ├── SignInPage.tsx                ← appearance prop added
│   └── SignUpPage.tsx                ← appearance prop added
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts
```

### Manual steps required

None. No new env vars, no new npm packages, no Clerk dashboard changes required.

### How to verify this step

1. `cd frontend && npm run dev`
2. Open `http://localhost:5173/sign-in`
3. **Desktop (≥ 768px)** — page should be a true split: dark left panel with the supply chain image, Stratos wordmark, headline, stats; white right panel with the Clerk sign-in form
4. **Left panel image** — `auth-bg.png` should be visible behind overlays; text should be legible; overall feel should be dark/premium, not raw-stretched
5. **Left panel stats** — three callouts (94%, 12ms, 3.4×) visible above the copyright footer
6. **Right panel** — Clerk form rendered cleanly, no extra card shadow/border, fits within white panel; buttons and inputs should feel styled (blue-800 primary, slate-50 input background)
7. **Sign-up page** — go to `http://localhost:5173/sign-up`; same split layout, same styled Clerk card
8. **Cross-links** — "Don't have an account? Sign up" and "Already have an account? Sign in" should still navigate between pages within the app (not to clerk.com)
9. **Auth redirects** — completing sign-in or sign-up should redirect to `/dashboard`
10. **Mobile (< 768px)** — left panel hidden; Stratos brand header appears above the Clerk form; layout does not break
11. **Dashboard unchanged** — `/dashboard` layout, sidebar, modules, and nav should be exactly as before
12. **Zero TypeScript errors** — `npx tsc --noEmit` exits 0

---

## Step 8 — Premium Dark Landing Page

### What was done

1. **Redesigned `HomePage.tsx`** — full rewrite; now composes `LandingNav` and `HeroSection` on a dark `bg-slate-950` base with a dot-grid overlay and radial blue glow.

2. **Created `LandingNav`** (`src/components/landing/LandingNav.tsx`):
   - Fixed to top of viewport (`position: fixed`), `h-16`, `backdrop-blur-md` frost-glass treatment
   - Stratos + AI wordmark on the left (same visual weight as auth and dashboard sidebar)
   - Placeholder center links: Platform · Solutions · Pricing
   - Right side is **auth-aware**:
     - Loading: animated skeleton placeholder (no CTA flicker)
     - Signed out: "Sign In" text link + "Get Started" blue pill CTA
     - Signed in: "Go to Dashboard" blue pill with arrow icon
   - Collapses "Sign In" on mobile (`hidden sm:block`) so the primary CTA is always visible

3. **Created `HeroSection`** (`src/components/landing/HeroSection.tsx`):
   - Nav spacer (`h-16`) so fixed nav doesn't overlap content
   - **Eyebrow badge** — rounded-full pill with pulsing blue dot and "Supply chain intelligence · Powered by AI" label
   - **Headline** — `text-[40px]` → `text-[56px]` at md; "Know every risk / before it arrives." with the second line in `text-blue-400`
   - **Subheadline** — one sentence, `text-slate-400`, `max-w-xl`
   - **CTA group** — auth-aware (same Clerk `useAuth` check):
     - Signed out: "Start for free" (blue, glowing box-shadow) + "Sign in" (frosted glass border)
     - Signed in: "Go to Dashboard" (blue, glowing box-shadow)
     - Loading: pulse skeleton so there is no layout shift
   - **Trust strip** — "No credit card required · 14-day free trial · Cancel anytime" in muted slate-600
   - **`DashboardPreview`** sub-component (internal to `HeroSection.tsx`) — a browser-chrome-framed abstract mock of the Stratos dashboard:
     - Browser chrome with traffic-light dots and URL bar showing `app.stratos.ai/dashboard`
     - Left sidebar with brand wordmark + nav items (active state on Dashboard)
     - Page header row with title, store context, and export action
     - 4 KPI cards (Forecast Acc., Days of Supply, Stock at Risk, Lead Time) with delta indicators
     - Inventory Trend bar chart (12 bars, recent bars in full blue)
     - Stock Health mini-donut with Healthy / Low / Critical legend
     - 3-row data table hint (SKU · Stock · Forecast · Status)
     - Soft radial glow beneath the frame
     - Bottom fade mask (`linear-gradient` overlay) for a "peek" depth effect

4. **Updated `RootLayout.tsx`** — removed hardcoded `bg-white` so each route can control its own background; landing page sets `bg-slate-950`, auth pages keep their own styling.

---

### Files created

| File | Role |
|---|---|
| `src/components/landing/LandingNav.tsx` | Premium fixed navbar for the public landing page |
| `src/components/landing/HeroSection.tsx` | Full hero section + internal `DashboardPreview` mock |

### Files modified

| File | Change |
|---|---|
| `src/pages/HomePage.tsx` | Full rewrite — now a dark landing page using the new components |
| `src/app/RootLayout.tsx` | Removed `bg-white` to allow per-route background control |

---

### Updated file tree

```
frontend/src/
├── App.tsx
├── index.css
├── main.tsx
├── app/
│   ├── AppShell.tsx
│   ├── AuthLayout.tsx
│   ├── RootLayout.tsx          ← updated (removed bg-white)
│   └── TopBarContext.tsx
├── components/
│   ├── ProtectedRoute.tsx
│   ├── dashboard/
│   │   ├── CategoryBreakdown.tsx
│   │   ├── ChartSkeleton.tsx
│   │   ├── CriticalItemsTable.tsx
│   │   ├── DaysOfSupplyModule.tsx
│   │   ├── InsightCards.tsx
│   │   ├── InventoryHeatmapModule.tsx
│   │   ├── InventoryTrendChart.tsx
│   │   ├── KpiCard.tsx
│   │   ├── KpiSkeleton.tsx
│   │   ├── LeadTimeRiskModule.tsx
│   │   ├── ModuleSkeleton.tsx
│   │   ├── RiskSpotlightPanel.tsx
│   │   ├── SectionContainer.tsx
│   │   ├── SellThroughModule.tsx
│   │   ├── ShrinkageModule.tsx
│   │   ├── StockHealthChart.tsx
│   │   ├── StoreSelector.tsx
│   │   └── VarianceHighlights.tsx
│   └── landing/                ← new
│       ├── HeroSection.tsx     ← new (includes DashboardPreview)
│       └── LandingNav.tsx      ← new
├── lib/
│   ├── api.ts
│   ├── clerkAppearance.ts
│   ├── config.ts
│   └── inventory.ts
├── pages/
│   ├── DashboardPage.tsx
│   ├── HomePage.tsx            ← rewritten
│   ├── SignInPage.tsx
│   └── SignUpPage.tsx
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts
```

---

### Packages added

None. No new dependencies were required.

---

### Manual steps required

None. No new env vars, no Clerk dashboard changes, no npm installs needed.

---

### Auth-aware CTA logic

`LandingNav` and `HeroSection` both call `useAuth()` from `@clerk/clerk-react`:

```tsx
const { isSignedIn, isLoaded } = useAuth();
```

| State | Navbar shows | Hero CTA shows |
|---|---|---|
| Loading (`!isLoaded`) | Pulse skeleton | Pulse skeleton |
| Signed out | "Sign In" + "Get Started" | "Start for free" + "Sign in" |
| Signed in | "Go to Dashboard →" | "Go to Dashboard →" |

No double render or layout shift — the skeleton fills the exact button width while Clerk resolves.

---

### Responsiveness

| Breakpoint | Behavior |
|---|---|
| `lg` (1024px+) | Full layout: nav center links + hero + wide dashboard preview |
| `md` (768px–1023px) | Same layout; center nav links visible; preview sidebar visible |
| `sm` (640px–767px) | "Sign In" nav link hidden (space); CTA and headline readable; sidebar visible in preview |
| Mobile (< 640px) | Headline stacks; trust strip dots hidden; dashboard preview has no sidebar; still reads cleanly |

Desktop/laptop quality is the priority — all proportions and visual depth are tuned for 1280px–1440px wide screens.

---

### How to verify this step

1. `cd frontend && npm run dev`
2. Open `http://localhost:5173/` (while **signed out**)
   - Page should be **dark slate-950** background with a faint dot grid
   - Fixed frosted navbar: Stratos AI wordmark left, Platform/Solutions/Pricing center, "Sign In" + "Get Started" right
   - Below nav: pulsing blue eyebrow badge → bold headline "Know every risk / **before it arrives.**" (blue accent line) → slate subheadline → blue "Start for free" CTA + bordered "Sign in" → trust strip
   - Below text: browser-framed dashboard mock with sidebar, KPI cards, bar chart, donut, table rows
   - Bottom of preview fades to dark (depth mask)
3. Click **"Get Started"** → should route to `/sign-up`
4. Click **"Sign In"** (nav or hero) → should route to `/sign-in`
5. Sign in, then return to `http://localhost:5173/`
   - Navbar right side: "Go to Dashboard →" (blue)
   - Hero CTA: "Go to Dashboard →" (blue, glowing)
   - Both links should navigate to `/dashboard`
6. Navigate to `/dashboard` → layout, sidebar, and all modules should be **unchanged**
7. Navigate to `/sign-in` and `/sign-up` → auth split layout should be **unchanged**
8. Resize to tablet width (768px) — layout should remain composed, no broken overflow
9. Resize to mobile (375px) — page should be readable though not pixel-perfect

---

## Step 6 — Premium Light-Mode Landing Page with GSAP Motion

### What was done

Redesigned the public landing page (`/`) from a single dark-mode hero screen into a full premium **light-mode SaaS landing page** with tasteful GSAP motion, multiple polished sections, a compact team credits module, and a proper SaaS footer.

Key changes:
1. **Converted to light-mode** — full `bg-white` / `bg-slate-50` / `bg-slate-100` surface system; dark sections (`bg-slate-950`) used only for contrast impact (metrics, footer).
2. **GSAP installed and wired** — staggered hero entrance, scroll-triggered section reveal, hover-lift on feature cards.
3. **Navbar redesigned** — light background, scroll-aware shadow, same auth-aware CTA logic.
4. **Hero section redesigned** — light browser chrome frame around the dark product preview; radial blue tint gradient; faint dot-grid overlay; large responsive headline up to 76px on XL.
5. **Five page sections added** — Features, Metrics, Workflow, Team, Footer (each its own component).
6. **Team section** — compact card grid with initials avatars, name, and role tag for all 5 listed team members.
7. **Footer** — premium dark (`bg-slate-950`) SaaS footer with brand, tagline, nav columns, status pill, and copyright.
8. **Auth-aware CTAs preserved** — signed-out shows "Start for free" + "Sign in"; signed-in shows "Go to Dashboard".
9. **TypeScript stays clean** — zero `tsc` errors; all icon types use `FC` from react to avoid global `JSX` namespace.

---

### Files created

| File | Notes |
|---|---|
| `src/components/landing/FeaturesSection.tsx` | 6 capability cards — GSAP scroll trigger + hover-lift per card |
| `src/components/landing/MetricsSection.tsx` | 4 key outcome stats on dark (`bg-slate-950`) background |
| `src/components/landing/WorkflowSection.tsx` | 3-step "How it works" section on `bg-slate-50` |
| `src/components/landing/TeamSection.tsx` | Compact 5-member team credits with initials avatars |
| `src/components/landing/LandingFooter.tsx` | Premium dark SaaS footer with status pill and nav columns |

### Files modified

| File | Notes |
|---|---|
| `src/pages/HomePage.tsx` | Complete rewrite — light-mode, composes all 5 sections + footer |
| `src/components/landing/LandingNav.tsx` | Complete rewrite — light mode, scroll-aware shadow, GSAP entrance |
| `src/components/landing/HeroSection.tsx` | Complete rewrite — light mode, light browser chrome, GSAP stagger |

---

### Packages added

| Package | Version | Purpose |
|---|---|---|
| `gsap` | ^3.x | GSAP core + ScrollTrigger for motion |

---

### Updated frontend tree

```
frontend/src/
├── App.tsx
├── index.css
├── main.tsx
├── app/
│   ├── AppShell.tsx
│   ├── AuthLayout.tsx
│   ├── RootLayout.tsx
│   └── TopBarContext.tsx
├── components/
│   ├── dashboard/            (18 components, unchanged)
│   │   ├── CategoryBreakdown.tsx
│   │   ├── ChartSkeleton.tsx
│   │   ├── CriticalItemsTable.tsx
│   │   ├── DaysOfSupplyModule.tsx
│   │   ├── InsightCards.tsx
│   │   ├── InventoryHeatmapModule.tsx
│   │   ├── InventoryTrendChart.tsx
│   │   ├── KpiCard.tsx
│   │   ├── KpiSkeleton.tsx
│   │   ├── LeadTimeRiskModule.tsx
│   │   ├── ModuleSkeleton.tsx
│   │   ├── RiskSpotlightPanel.tsx
│   │   ├── SectionContainer.tsx
│   │   ├── SellThroughModule.tsx
│   │   ├── ShrinkageModule.tsx
│   │   ├── StockHealthChart.tsx
│   │   ├── StoreSelector.tsx
│   │   └── VarianceHighlights.tsx
│   └── landing/              (7 components — 5 new, 2 rewritten)
│       ├── FeaturesSection.tsx   ← NEW
│       ├── HeroSection.tsx       ← REWRITTEN
│       ├── LandingFooter.tsx     ← NEW
│       ├── LandingNav.tsx        ← REWRITTEN
│       ├── MetricsSection.tsx    ← NEW
│       ├── TeamSection.tsx       ← NEW
│       └── WorkflowSection.tsx   ← NEW
├── hooks/
│   └── useCurrentUser.ts
├── lib/
│   ├── api.ts
│   ├── clerkAppearance.ts
│   ├── config.ts
│   └── inventory.ts
├── pages/
│   ├── DashboardPage.tsx         (unchanged)
│   ├── HomePage.tsx              ← REWRITTEN
│   ├── SignInPage.tsx            (unchanged)
│   └── SignUpPage.tsx            (unchanged)
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts
```

---

### GSAP interactions added

| Location | Interaction | Description |
|---|---|---|
| `LandingNav` | Mount entrance | Slide down from y:-20 + fade in, 0.65s ease power2.out |
| `HeroSection` | Staggered entrance | Each element (eyebrow → headline → sub → CTA → trust → preview) fades up with 90ms stagger |
| `FeaturesSection` | Scroll trigger | All 6 cards fade up from y:28 with 90ms stagger on ScrollTrigger `top 78%` |
| `FeaturesSection` | Card hover-lift | `gsap.to` on mouseenter/mouseleave — y:-5 + shadow enhancement, 0.25s / 0.3s |
| `MetricsSection` | Scroll trigger | 4 metric items fade up with 100ms stagger |
| `WorkflowSection` | Scroll trigger | 3 step cards fade up with 130ms stagger |
| `TeamSection` | Scroll trigger | 5 team cards fade up with 80ms stagger |

---

### Landing page sections

| Section | Background | Notes |
|---|---|---|
| Hero | `bg-white` + radial blue tint | Large headline, dashboard preview in light browser chrome |
| Features | `bg-white` | 6 capability cards in 3-column grid |
| Metrics | `bg-slate-950` (dark) | 4 key outcome stats in large bold type |
| Workflow | `bg-slate-50` | 3-step "How it works" process cards |
| Team | `bg-white` | 5 team member cards with initials avatars |
| Footer | `bg-slate-950` (dark) | Brand, tagline, nav columns, status pill, copyright |

---

### Team members displayed

Cards with initials avatar (colored bg), name, and role tag:

| Member | Role shown | Avatar accent |
|---|---|---|
| Sanjana Brahmbhatt | MLOps · ETL | blue |
| Rohit Prabu | MLOps · ETL | indigo |
| Vedashree Bane | Containerization · ML CI/CD | emerald |
| Somya Padhy | Backend · Deployment | amber |
| Ghanashyam | Documentation | slate |

(Aryan Mehta intentionally omitted per project instructions.)

---

### Light/dark architecture

- Landing page root div: `bg-white` — light by default with no dark-variant CSS
- Dark sections (`bg-slate-950`) are **intentional high-contrast surfaces** (Metrics, Footer), not a dark mode
- No theme toggle was added in this step (classified as optional; can be layered later with class-based Tailwind dark mode)
- Dashboard (`bg-slate-900` sidebar + explicit colors) is completely unaffected
- Auth pages are completely unaffected

---

### Auth-aware CTA behavior

| State | Navbar | Hero CTA |
|---|---|---|
| `isLoaded = false` | Pulse skeleton | Pulse skeleton |
| Signed out | "Sign in" text + "Get Started" button | "Start for free" (primary) + "Sign in" (secondary) |
| Signed in | "Go to Dashboard →" button | "Go to Dashboard →" (primary) |

---

### Responsiveness

| Breakpoint | Layout |
|---|---|
| `xl` (1280px+) | Hero headline 76px; features 3-col; metrics 4-col; team 5-col wrap |
| `lg` (1024px–1279px) | Hero headline 68px; features 3-col; metrics 4-col |
| `md` (768px–1023px) | Hero headline 58px; features 2-col; metrics 2-col; workflow 3-col; nav links visible |
| `sm` (640px–767px) | Team 2-col; trust strip dots hidden; nav "Sign in" hidden |
| Mobile (<640px) | Features 1-col; metrics 2-col; workflow 1-col; team 1-col |

---

### Manual steps required

None. GSAP was added to `dependencies` via `npm install gsap`. Run `npm install` and `npm run dev`.

---

### How to verify this step

1. `cd frontend && npm run dev`
2. Open `http://localhost:5173/` while **signed out**:
   - Page background should be **white** (not dark)
   - Navbar: white/frosted; "Stratos AI" left; "Platform / How it works / Team" center; "Sign in" + "Get Started" right
   - Hero: blue eyebrow badge → large dark headline "Know every risk / **before it arrives.**" (blue accent) → gray subheadline → dark "Start for free" CTA + bordered "Sign in" → trust strip → **light browser chrome** around the dashboard preview (dark product UI inside)
   - Scroll down: Features section (6 cards) → cards animate in on scroll; hover a card to see the lift effect
   - Metrics section: **dark background** with large white numbers (94%, 12ms, 3.4×, 50+)
   - Workflow section: light gray bg with 3 step cards
   - Team section: 5 compact cards with colored initials avatars
   - Footer: dark bg with brand, nav columns, status pill, copyright
3. Resize to 1280px wide — content should fill the viewport confidently, no boxed/narrow feel
4. Sign in, return to `http://localhost:5173/` — CTAs should change to "Go to Dashboard →"
5. Visit `/dashboard` — unchanged; dark sidebar and all data modules work as before
6. Visit `/sign-in` and `/sign-up` — auth layout unchanged
7. `npm run build` — should complete with zero TypeScript errors

---

## Step 11 — Landing Page Polish Pass

### What was done

A targeted polish pass over the landing page only. Dashboard, auth pages, and nav were not touched.

#### 1. Hero preview seam fix (`HeroSection.tsx`)
- Added `pb-20 md:pb-28` to the dashboard preview wrapper so the browser-chrome frame now floats with breathing room above the next section — the hard edge of the frame no longer sits flush at the section boundary.
- Increased the bottom fade overlay from `h-12` to `h-28` and changed the end stop from `rgba(248,250,252,0.5)` (slate-50 at 50%) to `rgba(255,255,255,0.92)` — the frame now fades cleanly into the white page background with no residual slate tint.
- Removed `border-t border-slate-100` from `FeaturesSection` — redundant divider that created a double-line seam where the preview ended and features began.

#### 2. Workflow section — connected process rebuild (`WorkflowSection.tsx`)
- Replaced the disconnected card grid with a **flex-row process layout**: cards sit side-by-side with arrow connectors between them on desktop; a thin vertical line connects them on mobile.
- Each card now has a **consistent top structure**: step badge (01/02/03 in a filled circle) at the top-left, followed by a `bg-linear-to-r` rule to the card edge, then the icon container, then title, then body copy — all in the same vertical order in every card.
- Arrow connector SVG (`ConnectorArrow`) sits between cards in the flex row (desktop), hidden on mobile.
- Vertical hairline connector shown on mobile between stacked cards.
- Cards use `flex-1` to grow equally across the row; gap is removed on desktop so cards and connectors pack tightly.
- Fixed Tailwind v4 warning: `bg-gradient-to-r` → `bg-linear-to-r`.

#### 3. Team section — 3×2 grid with vertical cards (`TeamSection.tsx`)
- Replaced the cramped flex-wrap horizontal card layout with a proper **CSS grid**: `grid-cols-1 sm:grid-cols-2 lg:grid-cols-3` — 3 columns on desktop, 2 on tablet, 1 on mobile.
- Cards are now **vertical and centered**: large `h-36 w-36` avatar at the top, name below (semibold), contribution text below that (allowed to wrap — no `truncate`).
- Each card has a **per-member tinted background** (`memberBg` hex field on the data object) so the grid reads as a cohesive colour-coded set rather than uniform white tiles.
- Avatar circles updated to `bg-*-200` tones (one step deeper than the previous `bg-*-100`) to contrast against the tinted card background.
- **Photo support** baked in: each member has an `img` path (`/images/<name>.png`). The avatar renders both layers — initials div always present underneath (`absolute inset-0`), photo stacked on top; `onError` handler sets `display:none` on the image if the file is missing, revealing the initials automatically. No state management needed.
- Updated contribution text strings:
  - Aryan Mehta: `Frontend · Frontend CI/CD · Designing · Branding`
  - Vedashree Bane: `Containerization · ML Models CI/CD`
  - Somya Padhy: `Backend · Backend Deployment`
- All 6 members display without truncation; consistent `px-6 py-9` internal padding across all cards.

#### 4. Footer — light mode premium SaaS footer (`LandingFooter.tsx`)
- Switched from dark (`bg-slate-950`) to light (`bg-slate-50`) to match the overall landing page palette.
- Grid layout: `1col` mobile → `2col sm` → `4col md`, brand block spans `col-span-2`.
- Brand block: Stratos wordmark, 2-line tagline, project attribution pill (Northeastern University · MLOps · Spring 2026 inside a white bordered pill with an emerald dot).
- Link columns renamed from "Company" to "Project"; footer now has "Product" and "Project" groups.
- Product links (`Platform`, `How it works`, `Features`) use real `href` anchors pointing to `#features` / `#workflow`.
- Project links: `Team` → `#team`, `Documentation` → `#`. **GitHub** is a hardcoded `<a>` outside the links constant, pointing to the real repo URL (`https://github.com/SanjanaB123/AI-based-Supply-Chain-Management`) with `target="_blank" rel="noreferrer noopener"`, an underline style, and a `↗` indicator.
- Section headers: `text-slate-400 uppercase tracking-widest text-[11px]`.
- Link text: `text-slate-500 hover:text-slate-900` for clean light-mode readability.
- Bottom bar: `border-t border-slate-200`, copyright left, project credit right — both in `text-slate-400`.

#### 5. Features section light improvements (`FeaturesSection.tsx`)
- Icon container: `h-10 w-10` → `h-11 w-11` for slightly more visual presence.
- Hover shadow: `0 16px 40px -8px rgba(0,0,0,0.10)` → `0 20px 48px -8px rgba(0,0,0,0.11)` — marginally more elevated lift.
- Removed `border-t border-slate-100` (moved boundary responsibility to the hero's new bottom padding).

---

### Files modified

| File | Change |
|---|---|
| `src/components/landing/HeroSection.tsx` | Preview wrapper gets `pb-20 md:pb-28`; bottom fade enlarged and goes to white |
| `src/components/landing/FeaturesSection.tsx` | Removed `border-t`; icon container `h-11 w-11`; hover shadow bump |
| `src/components/landing/WorkflowSection.tsx` | Full layout rebuild — flex-row, arrow connectors, consistent card structure |
| `src/components/landing/TeamSection.tsx` | Full rebuild — 3×2 grid, vertical cards, larger avatars, updated role strings |
| `src/components/landing/LandingFooter.tsx` | Full rebuild — light mode, premium SaaS footer layout |

### Files created

None.

### Packages added

None.

---

### How the hero border issue was fixed

The visible seam was caused by two adjacent lines: the bottom border of the browser-chrome preview frame, and the `border-t border-slate-100` at the top of `FeaturesSection`. Both appeared at the same vertical position because the preview wrapper had `pb-0`.

Fix: added `pb-20 md:pb-28` to the preview wrapper (space below the frame), extended the bottom fade overlay to `h-28` going to near-opaque white, and removed the redundant `border-t` from `FeaturesSection`. The frame now floats in white space and its edge is softened by the fade.

---

### Updated frontend tree

```
frontend/src/
├── App.tsx
├── index.css
├── main.tsx
├── app/
│   ├── AppShell.tsx
│   ├── AuthLayout.tsx
│   ├── RootLayout.tsx
│   └── TopBarContext.tsx
├── components/
│   ├── ProtectedRoute.tsx
│   ├── dashboard/            (18 components, unchanged)
│   └── landing/
│       ├── FeaturesSection.tsx   ← icon size, hover shadow, border-t removed
│       ├── HeroSection.tsx       ← preview padding + fade improved
│       ├── LandingFooter.tsx     ← REBUILT: light mode premium footer
│       ├── LandingNav.tsx        (unchanged)
│       ├── MetricsSection.tsx    (unchanged)
│       ├── TeamSection.tsx       ← REBUILT: 3×2 grid, vertical cards
│       └── WorkflowSection.tsx   ← REBUILT: flex-row process, arrow connectors
├── hooks/
│   └── useCurrentUser.ts
├── lib/
│   ├── api.ts
│   ├── clerkAppearance.ts
│   ├── config.ts
│   └── inventory.ts
├── pages/
│   ├── DashboardPage.tsx         (unchanged)
│   ├── HomePage.tsx              (unchanged)
│   ├── SignInPage.tsx            (unchanged)
│   └── SignUpPage.tsx            (unchanged)
├── routes/
│   └── AppRouter.tsx
└── types/
    └── inventory.ts
```

---

### Manual steps required

None. No new env vars, no new packages, no Clerk changes.

---

### How to verify this step

1. `cd frontend && npm run dev`
2. Open `http://localhost:5173/` (signed out)
3. **Hero border** — scroll slowly; the dashboard preview should float cleanly above the Features section with no hard seam or double-line border visible at the transition
4. **Workflow section** — three cards sit in a row separated by `→` arrow connectors (desktop); each card has the step number badge (01/02/03) at top-left followed by the icon, title, and body in consistent alignment; on mobile cards stack with a vertical line connector between them
5. **Team section** — six cards in a 3×2 grid (desktop), 2×3 (tablet), 1×6 (mobile); each card has a distinct tinted background colour; where a photo exists at `/images/<name>.png` the photo renders, otherwise the initials circle shows (test by renaming/removing the image file — the card should fall back gracefully); Aryan Mehta appears first
6. **Footer** — light gray background (`bg-slate-50`), not dark; brand wordmark top-left; attribution pill below tagline; "Product" and "Project" link columns; GitHub link opens the real repo in a new tab with an `↗` indicator; clean bottom bar with copyright
7. **Features cards** — hover a card; the lift shadow should be slightly more elevated; icon containers are slightly larger
8. **Dashboard unchanged** — `/dashboard` layout, sidebar, modules, and nav are exactly as before
9. **Auth pages unchanged** — `/sign-in` and `/sign-up` split layout unchanged
10. `npx tsc --noEmit` — zero TypeScript errors

---

## Step 12 — Dark / Light Theme System

### What was done

Implemented a complete, production-grade dark/light theme system across the entire frontend.

**Theme infrastructure (`src/app/theme/`)**

- `ThemeProvider.tsx` — React context provider that reads `localStorage` (`stratos-theme` key), applies/removes the `.dark` class on `<html>`, and persists the user's choice.
- `useTheme.ts` — `useTheme()` hook that returns `{ theme, toggleTheme }`.
- Both are exported from the `src/app/theme/` directory.

**Flash-of-wrong-theme prevention**

An inline `<script>` block is injected in `index.html`'s `<head>` that runs synchronously before React hydrates:

```html
<script>
  (function () {
    var saved = localStorage.getItem('stratos-theme');
    if (saved === 'dark') { document.documentElement.classList.add('dark'); }
  })();
</script>
```

**Tailwind v4 dark mode configuration (`src/index.css`)**

```css
@import "tailwindcss";
@variant dark (&:where(.dark, .dark *));
```

This configures class-based dark mode — `dark:` utilities activate when any ancestor has `.dark`. Light mode is the default; there are **no `light:` utilities** anywhere in the codebase.

**Theme toggle button**

A sun/moon SVG icon toggle appears in:
- The desktop/mobile top bar (`AppShell.tsx`)
- The landing page nav (`LandingNav.tsx`)

**Design tokens**

| Surface | Light | Dark |
|---|---|---|
| Page background | `bg-white` / `bg-slate-50` | `dark:bg-slate-950` / `dark:bg-slate-900` |
| Card / module | `bg-white` | `dark:bg-slate-800` |
| Sidebar | `bg-white` | `dark:bg-slate-950` |
| Active nav item | `bg-blue-200 text-blue-700` | `dark:bg-blue-900/40 dark:text-blue-400` |
| Skeleton fills | `bg-slate-100` / `bg-slate-200` | `dark:bg-slate-700` |
| Dividers / borders | `border-slate-100/200` | `dark:border-slate-700` |
| Semantic colors | `red-500`, `amber-500`, `emerald-500` (preserved in both modes) |

**Chart theming (Recharts)**

`StockHealthChart` and `InventoryTrendChart` use `useTheme()` to pass dynamic `stroke`, `fill`, and `contentStyle` props directly to Recharts primitives (`CartesianGrid`, `XAxis`, `YAxis`, `Tooltip`, `Cursor`). This avoids CSS hacks and keeps chart elements fully responsive to theme changes.

**Coverage**

Every component in the app now carries `dark:` variants:

- Landing: `LandingNav`, `HeroSection`, `FeaturesSection`, `MetricsSection`, `WorkflowSection`, `TeamSection`, `LandingFooter`
- Auth: `AuthLayout`, `RootLayout`
- Dashboard shell: `AppShell`, `DashboardPage`
- KPI / Skeleton: `KpiCard`, `KpiSkeleton`, `ChartSkeleton`, `ModuleSkeleton`, `SectionContainer`, `StoreSelector`
- Charts: `StockHealthChart`, `InventoryTrendChart`
- Modules: `CriticalItemsTable`, `DaysOfSupplyModule`, `SellThroughModule`, `LeadTimeRiskModule`, `ShrinkageModule`, `RiskSpotlightPanel`, `CategoryBreakdown`, `VarianceHighlights`, `InventoryHeatmapModule`

**Notes**

- `MetricsSection` intentionally stays `bg-slate-950` in both modes (it's a brand-dark section by design).
- `HeroSection` dashboard preview stays dark in both modes (it's a product UI mockup).
- Team card pastel backgrounds are applied via conditional inline style only in light mode; `dark:bg-slate-800` handles dark mode without needing `!important`.
- `clerkAppearance.ts` was superseded in Step 13 — see below.

---

### New files

| File | Purpose |
|---|---|
| `src/app/theme/ThemeProvider.tsx` | Context provider + localStorage persistence |
| `src/app/theme/useTheme.ts` | `useTheme()` hook |

---

### How to verify this step

1. `cd frontend && npm run dev`
2. Open `http://localhost:5173/` — default is **light mode**
3. Click the sun/moon icon in the nav — page switches to **dark mode** immediately with no flash
4. Refresh the page — dark mode persists (localStorage `stratos-theme = "dark"`)
5. Navigate to `/dashboard` — sidebar, topbar, cards, charts, and all modules are themed correctly
6. Toggle again — back to light; all pastel team card backgrounds return
7. `npx tsc --noEmit` — zero TypeScript errors

---

## Step 13 — Clerk Theming + Sidebar Appearance Toggle + Logo

### What was done

Completed the theme system by making all Clerk UI surfaces (auth cards, UserButton popup) fully responsive to the app's light/dark theme, added a second theme toggle in the sidebar's Settings area, and replaced all text wordmarks with theme-aware logo images.

**How Clerk theming works**

Clerk does not read Tailwind's `.dark` class automatically — its internal styles are isolated from the DOM class. The solution is to pass an explicit `appearance` object to `ClerkProvider` that matches the current theme. When the user toggles the theme, the appearance object updates and Clerk re-renders with the new styles.

Architecture:
- `src/lib/clerkTheme.ts` — defines `clerkLightTheme` and `clerkDarkTheme`, each containing a `variables` block (CSS custom properties that Clerk reads internally) and an `elements` block (Tailwind class names applied directly to Clerk DOM nodes).
- `src/app/ThemedClerkProvider.tsx` — sits inside `<ThemeProvider>`, uses `useTheme()`, and passes the correct appearance object to `<ClerkProvider>` based on the active theme.
- `src/main.tsx` — updated to use `<ThemedClerkProvider>` in place of `<ClerkProvider>` directly.
- `src/pages/SignInPage.tsx` and `SignUpPage.tsx` — removed per-component `appearance` overrides; they now inherit from the provider.
- `src/lib/clerkAppearance.ts` — deleted (superseded by `clerkTheme.ts`).

**Specificity overrides for Clerk elements**

Some Clerk elements apply their own hardcoded color styles with enough specificity to override plain Tailwind classes. These elements use the Tailwind v4 `!` important suffix (`text-slate-200!`, `border-slate-700!`, etc.) in the dark theme to ensure correct rendering:
- `socialButtonsBlockButton` and `socialButtonsBlockButtonText` — "Login with Google" border and text color in dark mode
- `userButtonPopoverActionButtonText` — "Manage Account" and "Sign Out" text in the UserButton popup

**Clerk appearance design**

| Element | Light | Dark |
|---|---|---|
| Auth card | `bg-transparent` (layout shows through) | `bg-transparent` (layout's `dark:bg-slate-950` shows through) |
| Inputs | `bg-slate-50 border-slate-200 text-slate-900` | `bg-slate-800 border-slate-700 text-slate-100` |
| Primary button | `bg-blue-700 text-white hover:bg-blue-800` | `bg-blue-600 text-white hover:bg-blue-500` |
| Header title | `text-slate-900` | `text-slate-100` |
| Footer links | `text-blue-700 hover:text-blue-800` | `text-blue-400 hover:text-blue-300` |
| Divider | `bg-slate-200` | `bg-slate-700` |
| Alerts | `text-red-700` | `text-red-400` |
| OAuth button | `bg-white border-slate-200 text-slate-700` | `bg-slate-800 border-slate-700! text-slate-200!` |
| UserButton popup card | `bg-white border-slate-200/80 shadow-xl` | `bg-slate-800 border-slate-700/80 shadow-xl` |
| Popup action buttons | `hover:bg-slate-50 text-slate-700` | `hover:bg-slate-700 text-slate-200!` |
| Popup user name | `text-slate-900` | `text-slate-100` |
| Popup email | `text-slate-500` | `text-slate-400` |

**Sidebar theme toggle**

A compact "Appearance" row was added to the sidebar's bottom section, between the Settings nav item and the user account row. It shows:
- Label: `Appearance` (muted)
- Button: sun/moon icon + `Light` / `Dark` text label

It calls the same `toggleTheme()` from `useTheme()` as the top-bar button — both are wired to the same context state, so they stay in sync automatically.

**Logo images**

Two logo files placed in `public/icons/`:
- `logo-light.png` — used when the app is in light mode
- `logo-dark.png` — used when the app is in dark mode

The text "Stratos" + "AI" wordmarks in all nav locations have been replaced with `<img>` elements that switch source based on `theme` from `useTheme()`:

| Location | Light | Dark | Notes |
|---|---|---|---|
| Dashboard sidebar | `logo-light.png` | `logo-dark.png` | `h-7` |
| Dashboard mobile header | `logo-light.png` | `logo-dark.png` | `h-7` |
| Landing nav | `logo-light.png` | `logo-dark.png` | `h-7` |
| Auth mobile header | `logo-light.png` | `logo-dark.png` | `h-7` |
| Auth left panel | always `logo-dark.png` | always `logo-dark.png` | Left panel always has `bg-slate-950/70` overlay; `h-8 md:h-9 xl:h-12` |

---

### Files created

| File | Purpose |
|---|---|
| `src/lib/clerkTheme.ts` | Light + dark Clerk appearance objects |
| `src/app/ThemedClerkProvider.tsx` | Theme-aware ClerkProvider wrapper |

### Files modified

| File | Change |
|---|---|
| `src/main.tsx` | Replaced `<ClerkProvider>` with `<ThemedClerkProvider>` |
| `src/pages/SignInPage.tsx` | Removed static `appearance` prop (inherits from provider) |
| `src/pages/SignUpPage.tsx` | Removed static `appearance` prop (inherits from provider) |
| `src/app/AppShell.tsx` | Added sidebar Appearance toggle row; replaced wordmarks with logo images |
| `src/app/AuthLayout.tsx` | Added `useTheme`; replaced wordmarks with logo images |
| `src/components/landing/LandingNav.tsx` | Replaced wordmark with logo image |
| `src/lib/clerkTheme.ts` | Added `!` important suffix to social button and popup action text in dark theme |

### Files deleted

| File | Reason |
|---|---|
| `src/lib/clerkAppearance.ts` | Superseded by `clerkTheme.ts` |

---

### How to verify this step

1. `cd frontend && npm run dev`
2. **Logo** — in light mode the light logo appears in the navbar, sidebar, and auth pages; toggle to dark mode and the dark logo replaces it everywhere instantly
3. Open `http://localhost:5173/sign-in`
   - **Light mode**: white card, slate text, white inputs, blue submit button, "Login with Google" has dark border and text
   - Toggle to **dark mode**: `slate-950` background, light text, dark slate inputs, "Login with Google" shows light border and light text
4. Open `http://localhost:5173/sign-up` — same theming as sign-in
5. Sign in and go to `/dashboard`
6. Click the `UserButton` avatar (top-right or top-bar):
   - Light: white popup, dark text on "Manage Account" / "Sign Out"
   - Dark: slate-800 popup, light text on "Manage Account" / "Sign Out"
7. Toggle theme from the **top-bar** sun/moon button — entire app (including logo) updates instantly
8. Toggle theme from the **sidebar** Appearance row — same result, synced with top-bar
9. Refresh — theme persists from localStorage
10. `npx tsc --noEmit` — zero TypeScript errors

---

## Step 14 — Dashboard Footer Bar & Stratos AI Assistant Chat UI

### What was done

1. Built a premium **dashboard footer bar** that spans the bottom of the authenticated shell with three sections: AI status (left), copyright (center), and the AI assistant launcher (right).
2. Integrated `GET /api/ai-status` into an `AiStatusIndicator` component that polls every 30 seconds and shows a live dot + label.
3. Built a fully functional **Stratos AI Assistant** chat panel backed by `POST /api/chat`.
4. Implemented per-user conversation persistence via `localStorage` keyed by Clerk `userId`.
5. Mounted everything inside `AppShell` so the footer and chat are scoped exclusively to the authenticated dashboard.

---

### Architecture

#### Footer bar

```
DashboardFooterBar (h-10 shrink-0 strip at the bottom of the right column)
├── Left  → AiStatusIndicator  (polls /api/ai-status every 30 s)
├── Center → "© 2026 Stratos. All rights reserved." (muted, centered)
└── Right  → ChatLauncher button (opens / closes ChatPanel)
```

The footer is a `shrink-0` flex row appended below `<main>` inside the right column of `AppShell`. It inherits the same `bg-white dark:bg-slate-950` and `border-t border-slate-200 dark:border-slate-800` visual language as the top bar.

#### Chat panel

```
ChatPanel (fixed right-4 bottom-12 z-50, CSS slide-up/fade transition)
├── ChatHeader   — title, "Inventory copilot" subtitle, online dot, clear & close buttons
├── ChatMessageList  — scrollable, auto-scrolls to latest message
│   ├── ChatMessageBubble (user = blue right, assistant = slate left)
│   ├── TypingIndicator   — three staggered bounce dots while awaiting response
│   └── inline error banner on API failure
├── ChatEmptyState — sparkle icon + 5 suggested inventory prompts (shown when no messages)
└── ChatComposer  — auto-resizing textarea, send button, Enter-to-send / Shift+Enter newline
```

---

### Files created

| File | Purpose |
|------|---------|
| `src/types/chat.ts` | `ChatMessage`, `ChatRequest`, `ChatResponse`, `AiStatusResponse` types |
| `src/lib/chat.ts` | `sendChatMessage()` and `fetchAiStatus()` API functions |
| `src/hooks/useChatAssistant.ts` | All chat state: open/close, messages, send, clear, localStorage persistence |
| `src/components/chat/AiStatusIndicator.tsx` | Polls `/api/ai-status`, shows dot + label |
| `src/components/chat/DashboardFooterBar.tsx` | Footer bar shell (3-section flex layout) |
| `src/components/chat/ChatLauncher.tsx` | Footer-right pill button that toggles the panel |
| `src/components/chat/ChatPanel.tsx` | Fixed floating panel with CSS open/close transition |
| `src/components/chat/ChatHeader.tsx` | Panel header with title, clear, close actions |
| `src/components/chat/ChatMessageList.tsx` | Scrollable message list, auto-scrolls to bottom |
| `src/components/chat/ChatMessageBubble.tsx` | Individual user / assistant message bubble |
| `src/components/chat/ChatComposer.tsx` | Auto-resizing textarea + send button |
| `src/components/chat/ChatEmptyState.tsx` | Empty-state with sparkle icon + 5 prompt chips |
| `src/components/chat/TypingIndicator.tsx` | Three staggered bouncing dots while awaiting reply |

### Files modified

| File | Change |
|------|--------|
| `src/app/AppShell.tsx` | Imported chat components, wired `useChatAssistant`, added `DashboardFooterBar` and `ChatPanel` |

---

### API routes used

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/chat` | `POST` | Send user message, receive assistant response |
| `/api/ai-status` | `GET` | Poll AI backend health for footer indicator |

**`/api/chat` request body:**
```json
{ "message": "What are the low stock items?", "username": "<clerk-userId>" }
```

**`/api/chat` response:**
```json
{ "response": "...", "agent": "ai-assistant" }
```

**`/api/ai-status` response:**
```json
{ "status": "online", "ai_enabled": true, "model_loaded": true, "mcp_server": "https://..." }
```

---

### localStorage persistence strategy

- **Key format:** `stratos-chat:<userId>` where `userId` is the Clerk `userId` string.
- **On mount:** `useChatAssistant` reads from `localStorage` the first time a `userId` becomes available and restores the message array.
- **On every message change:** the full message array is serialised and written back to `localStorage`.
- **On clear chat:** `localStorage.removeItem(key)` removes only this user's history; other users are unaffected.
- **On sign-out / different user:** a different `userId` produces a different key, so conversation history is never shared between accounts.
- Storage errors (quota exceeded, private-browsing restrictions) are caught and silenced — the chat remains functional without persistence.

---

### How to verify footer + chat in light and dark mode

```
1. npm run dev  (inside frontend/)
2. Sign in and navigate to /dashboard
```

**Footer bar:**
- [ ] Footer appears as a slim bar at the very bottom of the dashboard shell
- [ ] Left section: dot + "AI online" (green) / "AI unavailable" (red) / "Connecting…" (grey) — updates within 30 s
- [ ] When AI is online and model is loaded: secondary label "Model loaded" appears after the separator
- [ ] Center: "© 2026 Stratos. All rights reserved." is centered
- [ ] Right: "Ask Stratos AI" pill button is visible
- [ ] In **dark mode**: footer uses `slate-950` background, `slate-800` border — matches sidebar/top bar

**Chat panel:**
- [ ] Clicking "Ask Stratos AI" slides up the chat panel from above the footer
- [ ] Panel shows the empty state with sparkle icon and 5 suggested prompts
- [ ] Clicking a prompt chip populates and sends it immediately
- [ ] User bubble appears right-aligned in blue; assistant response appears left-aligned in slate
- [ ] Three-dot typing indicator shows while the request is in flight
- [ ] Send button and textarea are disabled while sending
- [ ] After response: message list auto-scrolls to the latest message
- [ ] If API fails: red inline error message appears below the last message
- [ ] Clicking the trash icon clears the conversation (also removes from localStorage)
- [ ] Clicking × (or clicking "Ask Stratos AI" again) closes the panel with a fade/slide transition
- [ ] Refresh the page — previous conversation reloads from localStorage
- [ ] In **dark mode**: panel uses `slate-900` background, `slate-800` surfaces, correct text colours throughout

**TypeScript:**
```bash
npx tsc --noEmit   # must exit 0 with no output
```

---

## Step 15 — Dashboard Footer Bar & Chat Panel Polish Pass

### What was done

A targeted visual-interaction polish pass over the dashboard footer bar and Stratos AI chat panel. No business logic, API contracts, or authentication behaviour was changed.

---

### A — Footer bar: opposite-theme "control dock"

**Before:** `h-10`, same surface as app shell (`bg-white dark:bg-slate-950`).

**After:** `h-14`, inverted-theme surface — dark in light mode, light in dark mode.

| Mode | Background | Border |
|---|---|---|
| Light (default) | `bg-slate-900` | `border-slate-700` |
| Dark | `dark:bg-slate-100` | `dark:border-slate-300` |

This makes the footer read as a distinct "control dock" with strong visual separation from the main dashboard canvas, and vertically matches the weight of the sidebar user-account panel (which is also ~56 px tall with `py-3` padding).

**How the opposite-theme is implemented:**
Tailwind dark-mode variant is class-based (`.dark` on `<html>`). The footer simply swaps its surface tokens:
- `bg-slate-900` is the base (dark dock) — active in light mode where `.dark` is absent.
- `dark:bg-slate-100` overrides it to a light dock — active in dark mode where `.dark` is present.
No JavaScript, no theme listener, no inline style — purely two Tailwind classes.

**`AiStatusIndicator` — `inverted` prop added:**
The status text classes are adjusted for the reversed surface so WCAG AA contrast is met in both modes:

| Surface | Normal text class | Inverted text class |
|---|---|---|
| Light mode (dark footer) | `text-slate-500` (5.5:1 on white) | `text-slate-300` (9:1 on `slate-900`) |
| Dark mode (light footer) | `dark:text-slate-400` (6.5:1 on `slate-950`) | `dark:text-slate-600` (7.5:1 on `slate-100`) |

The separator and loading-dot colours are similarly swapped.

---

### B — Chat launcher button: always blue

**Before:** conditional — blue when open, slate when closed (dark mode adaptive).

**After:** `bg-blue-600 text-white hover:bg-blue-700` always, in both light and dark mode.

This makes it a stable branded primary CTA that reads clearly on both the dark footer (light mode) and the light footer (dark mode) without any theme-dependent logic. The `isOpen` state no longer changes the button style.

---

### C — Backdrop blur overlay when chat is open

A `fixed inset-0 z-40` `<div>` was added to `AppShell` between the page content and the chat panel (`z-50`):

```
bg-slate-900/20 backdrop-blur-[2px]
transition-opacity duration-300
pointer-events-none  ← never blocks dashboard interaction
```

- `opacity-100` when `chat.isOpen`, `opacity-0` otherwise — smooth 300ms CSS fade.
- `pointer-events-none` always: the dashboard remains interactive while the panel is open; clicking the overlay does **not** close the chat (preserving the existing UX contract).
- The blur is intentionally subtle (`2px`) — enough to create depth without making data illegible.

---

### D — Expand / collapse chat panel

**State:** `isExpanded: boolean` is held inside `ChatPanel` via `useState`. It starts as `false` and persists across open/close cycles (intentional — user's panel size preference is remembered for the session).

**Normal → Expanded dimensions:**

| Dimension | Normal | Expanded |
|---|---|---|
| Width | `w-[420px]` | `w-[680px]` |
| Max width | `max-w-[calc(100vw-2rem)]` | `max-w-[calc(100vw-2rem)]` |
| Height | `min(600px, calc(100vh-80px))` | `min(760px, calc(100vh-80px))` |

`min()` ensures the panel never overflows the viewport vertically, and `max-w-[calc(100vw-2rem)]` prevents horizontal overflow on narrow screens.

**Expand icon:** placed in `ChatHeader` to the **left** of the trash/clear icon, right side of the header action group:

```
[ ExpandIcon ] [ TrashIcon? ] [ XIcon ]
```

- `ExpandIcon` — outward-pointing corner arrows (SVG).
- `CollapseIcon` — inward-pointing corner arrows (SVG).
- `aria-label` switches between "Expand chat panel" and "Collapse chat panel".
- Always visible (not conditional on message count).

**Transition:** `transition-all duration-300 ease-out` on the panel `<div>` — Tailwind applies this to both `width` and `height` changes, producing a smooth resize animation.

**Footer offset:** Chat panel repositioned from `bottom-12` (48 px) to `bottom-14` (56 px) to sit flush above the taller `h-14` footer.

---

### E — Preserved functionality

All of the following remain unchanged:
- `GET /api/ai-status` polling (30-second interval, `AiStatusIndicator`)
- `POST /api/chat` message send with Clerk Bearer token
- `localStorage` persistence keyed by Clerk `userId`
- Open / close, send / typing indicator / clear / close behaviours
- Dark / light theme support across all chat components

---

### Files modified

| File | Change |
|---|---|
| `src/components/chat/AiStatusIndicator.tsx` | Added `inverted?: boolean` prop; text, separator, and dot colours adapt for reversed surface |
| `src/components/chat/DashboardFooterBar.tsx` | `h-10` → `h-14`; `bg-white dark:bg-slate-950` → `bg-slate-900 dark:bg-slate-100`; border tokens inverted; passes `inverted` to `AiStatusIndicator` |
| `src/components/chat/ChatLauncher.tsx` | Always `bg-blue-600 text-white hover:bg-blue-700`; removed theme-conditional styling |
| `src/components/chat/ChatHeader.tsx` | Added `isExpanded` + `onExpand` props; `ExpandIcon` / `CollapseIcon` SVGs; button placed left of trash icon; shared `iconBtnClass` constant |
| `src/components/chat/ChatPanel.tsx` | `useState(isExpanded)`; `handleToggleExpand`; dynamic `widthClass` + `panelHeight`; repositioned from `bottom-12` → `bottom-14`; `transition-all duration-300`; passes `isExpanded`/`onExpand` to `ChatHeader` |
| `src/app/AppShell.tsx` | Backdrop blur overlay (`fixed inset-0 z-40`, `pointer-events-none`, fade transition) added between content and `ChatPanel` |

### Files created

None.

---

### How to verify

#### Footer

1. `cd frontend && npm run dev` → open `/dashboard`
2. **Light mode** — footer should be visibly **dark** (`slate-900` navy/near-black) with a `slate-700` top border. It should contrast sharply against the light main canvas.
3. **Dark mode** (toggle theme) — footer should be visibly **light** (`slate-100` off-white) with a `slate-300` top border. It should contrast sharply against the dark main canvas.
4. In both modes, footer height should visually match the sidebar user-account panel height (~56 px).
5. Left section: AI status dot + label readable in both modes (check contrast manually).
6. Center: copyright text readable in both modes.

#### Chat launcher button

7. In **light mode**: "Ask Stratos AI" button is **solid blue** (`bg-blue-600`). Hover makes it slightly darker (`bg-blue-700`).
8. In **dark mode**: button remains **solid blue** (not inverted or transparent). Hover behaviour identical.
9. Open the panel → button stays blue (no mode change on open).

#### Backdrop blur overlay

10. Click "Ask Stratos AI" — a subtle blur/dim overlay should appear over the dashboard canvas behind the chat panel, fading in smoothly (~300ms).
11. Close the panel — overlay fades out.
12. While overlay is visible, click on a KPI card or dashboard element — it should remain interactive (pointer-events-none).

#### Expand / collapse

13. Open the chat panel — in the header, an expand icon (four outward-corner arrows) appears to the **left of the trash icon** (when messages exist) or left of the close button (when empty).
14. Click the expand icon — panel widens to ~680 px and grows taller; transition is smooth (~300ms).
15. Click again — panel returns to normal size (~420 px, 600 px max height); smooth transition.
16. Test on a laptop screen (≤1280 px): panel should not overflow viewport horizontally or vertically in either state.
17. In expanded mode, header stays pinned, messages scroll, composer stays pinned at bottom.
18. Close panel, reopen → expanded state persists for the session.

#### TypeScript

```bash
npx tsc --noEmit   # must exit 0 with zero output
```
