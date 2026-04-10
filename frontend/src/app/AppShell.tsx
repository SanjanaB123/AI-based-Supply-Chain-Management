import { UserButton, useUser } from '@clerk/clerk-react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { TopBarProvider, useTopBar } from './TopBarContext';

// ── Icons ─────────────────────────────────────────────────────────────────────

function DashboardIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <rect x="1" y="1" width="5.5" height="5.5" rx="1" fill="currentColor" />
      <rect x="8.5" y="1" width="5.5" height="5.5" rx="1" fill="currentColor" />
      <rect x="1" y="8.5" width="5.5" height="5.5" rx="1" fill="currentColor" />
      <rect x="8.5" y="8.5" width="5.5" height="5.5" rx="1" fill="currentColor" />
    </svg>
  );
}

function BarChartIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <rect x="1" y="7" width="3" height="7" rx="0.5" fill="currentColor" />
      <rect x="6" y="4" width="3" height="10" rx="0.5" fill="currentColor" />
      <rect x="11" y="1" width="3" height="13" rx="0.5" fill="currentColor" />
    </svg>
  );
}

function BoxIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path d="M7.5 1L13 4v7L7.5 14 2 11V4L7.5 1z" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinejoin="round" />
      <path d="M7.5 1v13M2 4l5.5 3.5L13 4" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
    </svg>
  );
}

function RiskIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path d="M7.5 1.5L13.5 12.5H1.5L7.5 1.5z" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinejoin="round" />
      <path d="M7.5 6v3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <circle cx="7.5" cy="10.5" r="0.6" fill="currentColor" />
    </svg>
  );
}

function VarianceIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path d="M1.5 11L5 7l3 3 5-6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SparkleIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path d="M7.5 1v2M7.5 12v2M1 7.5h2M12 7.5h2M3.05 3.05l1.42 1.42M10.53 10.53l1.42 1.42M10.53 4.47l1.42-1.42M3.05 11.95l1.42-1.42" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <circle cx="7.5" cy="7.5" r="2" fill="currentColor" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <circle cx="7.5" cy="7.5" r="2" stroke="currentColor" strokeWidth="1.3" />
      <path d="M7.5 1.5v1.25M7.5 12.25v1.25M1.5 7.5h1.25M12.25 7.5H13.5M3.34 3.34l.88.88M10.78 10.78l.88.88M10.78 4.22l.88-.88M3.34 11.66l.88-.88" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

function BookIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <rect x="2" y="1" width="10" height="13" rx="1" stroke="currentColor" strokeWidth="1.3" />
      <path d="M5 5h5M5 7.5h5M5 10h3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

function HelpCircleIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <circle cx="7.5" cy="7.5" r="6" stroke="currentColor" strokeWidth="1.3" />
      <path d="M5.5 6a2 2 0 0 1 4 0c0 1.5-2 1.5-2 3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <circle cx="7.5" cy="11" r="0.6" fill="currentColor" />
    </svg>
  );
}

function HeadsetIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path d="M2.5 8V7a5 5 0 0 1 10 0v1" stroke="currentColor" strokeWidth="1.3" />
      <rect x="1.5" y="8" width="2.5" height="4" rx="1" stroke="currentColor" strokeWidth="1.3" />
      <rect x="11" y="8" width="2.5" height="4" rx="1" stroke="currentColor" strokeWidth="1.3" />
    </svg>
  );
}

function BellIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M8 1.5a4.5 4.5 0 0 0-4.5 4.5V9L2 11h12l-1.5-2V6A4.5 4.5 0 0 0 8 1.5z" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinejoin="round" />
      <path d="M6.5 12.5a1.5 1.5 0 0 0 3 0" stroke="currentColor" strokeWidth="1.3" fill="none" />
    </svg>
  );
}

function InboxIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="1.5" y="3.5" width="13" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.3" />
      <path d="M1.5 9.5h3l1.5 2h4l1.5-2h3" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.3" />
      <path d="M9.5 9.5L12.5 12.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

// ── Nav structure ─────────────────────────────────────────────────────────────

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
  soon?: boolean;
}

interface NavSection {
  label?: string;
  items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    items: [
      { label: 'Dashboard', path: '/dashboard', icon: <DashboardIcon /> },
    ],
  },
  {
    label: 'Analytics',
    items: [
      { label: 'Analytics',  path: '/analytics', icon: <BarChartIcon />, soon: true },
      { label: 'Inventory',  path: '/inventory', icon: <BoxIcon />,      soon: true },
      { label: 'Risk',       path: '/risk',       icon: <RiskIcon />,    soon: true },
      { label: 'Variance',   path: '/variance',   icon: <VarianceIcon />, soon: true },
    ],
  },
  {
    label: 'Intelligence',
    items: [
      { label: 'AI Assistant', path: '/ai', icon: <SparkleIcon />, soon: true },
    ],
  },
  {
    label: 'Help & Support',
    items: [
      { label: 'User Guide',       path: '/guide',   icon: <BookIcon />,       soon: true },
      { label: 'FAQs',             path: '/faqs',    icon: <HelpCircleIcon />, soon: true },
      { label: 'Contact Support',  path: '/support', icon: <HeadsetIcon />,    soon: true },
    ],
  },
];

// ── NavLink ────────────────────────────────────────────────────────────────────

function NavLink({ item, active }: { item: NavItem; active: boolean }) {
  if (item.soon) {
    return (
      <div className="flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] font-medium opacity-40 cursor-not-allowed select-none">
        <span className="text-slate-800">{item.icon}</span>
        <span className="text-slate-800">{item.label}</span>
        <span className="ml-auto rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-widest text-slate-400 ring-1 ring-slate-300">
          Soon
        </span>
      </div>
    );
  }

  return (
    <Link
      to={item.path}
      className={`flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors ${
        active
          ? 'bg-blue-200 text-blue-700'
          : 'text-slate-600 hover:bg-slate-200/60 hover:text-slate-900'
      }`}
    >
      <span className={active ? 'text-blue-500' : 'text-slate-400'}>{item.icon}</span>
      {item.label}
    </Link>
  );
}

// ── Inner shell ───────────────────────────────────────────────────────────────

function AppShellInner() {
  const location = useLocation();
  const { topBarSlot, pageTitle, pageSubtitle } = useTopBar();
  const { user, isLoaded } = useUser();

  const displayName = isLoaded
    ? user?.firstName
      ? `${user.firstName}${user.lastName ? ' ' + user.lastName : ''}`
      : user?.username ?? user?.fullName ?? 'User'
    : '';
  const userEmail = isLoaded ? (user?.primaryEmailAddress?.emailAddress ?? '') : '';

  return (
    <div className="flex h-screen overflow-hidden bg-slate-100">

      {/* ── Sidebar (md+) ──────────────────────────────────────────────────── */}
      <aside className="hidden md:flex md:w-60 md:shrink-0 md:flex-col bg-white border-r border-slate-200">

        {/* Brand */}
        <div className="flex h-14 shrink-0 items-center px-5 border-b border-slate-200">
          <Link to="/" className="flex items-baseline gap-2 focus:outline-none">
            <span className="text-[15px] font-bold tracking-tight text-slate-900">Stratos</span>
            <span className="text-[10px] font-semibold uppercase tracking-widest text-blue-500">
              AI
            </span>
          </Link>
        </div>

        {/* Nav sections */}
        <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-5">
          {NAV_SECTIONS.map((section, si) => (
            <div key={si}>
              {section.label && (
                <p className="mb-1 px-3 text-[9px] font-semibold uppercase tracking-widest text-slate-900">
                  {section.label}
                </p>
              )}
              <div className="space-y-0.5">
                {section.items.map(item => (
                  <NavLink
                    key={item.path}
                    item={item}
                    active={!item.soon && location.pathname.startsWith(item.path)}
                  />
                ))}
              </div>
            </div>
          ))}
        </nav>

        {/* Bottom: Settings + user account */}
        <div className="shrink-0 border-t border-slate-200">
          <div className="px-3 pt-2 pb-1">
            <NavLink
              item={{ label: 'Settings', path: '/settings', icon: <GearIcon />, soon: true }}
              active={false}
            />
          </div>
          <div className="flex items-center gap-3 border-t border-slate-200 px-4 py-3">
            <UserButton />
            <div className="min-w-0 flex-1">
              <p className="truncate text-[13px] font-semibold leading-tight text-slate-800">
                {displayName}
              </p>
              {userEmail && (
                <p className="truncate text-[11px] leading-tight text-slate-400">{userEmail}</p>
              )}
            </div>
          </div>
        </div>
      </aside>

      {/* ── Right side ─────────────────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col overflow-hidden">

        {/* Desktop top bar */}
        <header className="hidden md:flex h-14 shrink-0 items-center gap-3 border-b border-slate-200 bg-white px-5">

          {/* Page title + subtitle */}
          <div className="flex min-w-0 flex-col justify-center">
            <h1 className="text-[15px] font-semibold leading-tight tracking-tight text-slate-900 xl:text-base">
              {pageTitle || 'Dashboard'}
            </h1>
            {pageSubtitle && (
              <p className="text-[11px] leading-tight text-slate-400 xl:text-[12px]">
                {pageSubtitle}
              </p>
            )}
          </div>

          {/* Divider before store selector */}
          {topBarSlot && <div className="h-5 w-px shrink-0 bg-slate-200" />}

          {/* Store selector slot */}
          {topBarSlot && (
            <div className="flex shrink-0 items-center gap-3">
              {topBarSlot}
            </div>
          )}

          {/* Push utility actions to the right */}
          <div className="flex-1" />

          {/* Search bar */}
          <div className="relative hidden lg:flex items-center">
            <span className="pointer-events-none absolute left-2.5 text-slate-400">
              <SearchIcon />
            </span>
            <input
              type="text"
              placeholder="Search products, stores…"
              className="h-8 w-48 rounded-lg border border-slate-200 bg-slate-50 pl-8 pr-3 text-[12px] text-slate-700 placeholder:text-slate-400 transition-all focus:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-100 xl:w-60"
              readOnly
            />
          </div>

          {/* Notification + inbox + user */}
          <div className="flex items-center gap-0.5">
            <button
              className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
              aria-label="Notifications"
            >
              <BellIcon />
            </button>
            <button
              className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
              aria-label="Inbox"
            >
              <InboxIcon />
            </button>
            <div className="mx-2 h-5 w-px bg-slate-200" />
            <UserButton />
          </div>
        </header>

        {/* Mobile top bar */}
        <header className="flex md:hidden h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4">
          <Link to="/" className="text-[15px] font-bold tracking-tight text-slate-900">
            Stratos
          </Link>
          <UserButton />
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto bg-slate-50 px-6 py-7 lg:px-10 lg:py-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

// ── AppShell (exported) ───────────────────────────────────────────────────────

export default function AppShell() {
  return (
    <TopBarProvider>
      <AppShellInner />
    </TopBarProvider>
  );
}
