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
