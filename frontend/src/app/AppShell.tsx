import { UserButton } from '@clerk/clerk-react';
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

// Top bar utility icons
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
        <span className="text-slate-500">{item.icon}</span>
        <span className="text-slate-300">{item.label}</span>
        <span className="ml-auto rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-widest text-slate-500 ring-1 ring-white/10">
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
          ? 'bg-blue-500/20 text-blue-50'
          : 'text-slate-300 hover:bg-white/5 hover:text-slate-100'
      }`}
    >
      <span className={active ? 'text-blue-300' : 'text-slate-500'}>{item.icon}</span>
      {item.label}
    </Link>
  );
}

// ── Inner shell (uses context) ────────────────────────────────────────────────

function AppShellInner() {
  const location = useLocation();
  const { topBarSlot } = useTopBar();

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">

      {/* ── Sidebar (md+) ──────────────────────────────────────────────────── */}
      <aside className="hidden md:flex md:w-60 md:shrink-0 md:flex-col bg-slate-900">

        {/* Brand */}
        <div className="flex h-14 shrink-0 items-center px-5 border-b border-white/5">
          <Link to="/" className="flex items-baseline gap-2 focus:outline-none">
            <span className="text-[15px] font-bold tracking-tight text-white">Stratos</span>
            <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              AI
            </span>
          </Link>
        </div>

        {/* Nav sections */}
        <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-5">
          {NAV_SECTIONS.map((section, si) => (
            <div key={si}>
              {section.label && (
                <p className="mb-1 px-3 text-[9px] font-semibold uppercase tracking-widest text-slate-400">
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

        {/* Bottom: Settings + Account */}
        <div className="shrink-0 border-t border-white/5">
          <div className="px-3 pt-2 pb-1">
            <NavLink
              item={{ label: 'Settings', path: '/settings', icon: <GearIcon />, soon: true }}
              active={false}
            />
          </div>
          <div className="flex items-center gap-3 px-5 py-3.5">
            <UserButton />
            <span className="text-[12px] text-slate-400 truncate select-none">Account</span>
          </div>
        </div>
      </aside>

      {/* ── Right side ─────────────────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col overflow-hidden">

        {/* Desktop top bar — hidden on mobile */}
        <header className="hidden md:flex h-14 shrink-0 items-center justify-between border-b border-white/5 bg-slate-800 px-5">
          {/* Left: page slot (store selector injected by page) */}
          <div className="flex items-center gap-3">
            {topBarSlot}
          </div>

          {/* Right: utility actions + user */}
          <div className="flex items-center gap-1">
            <button
              className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-white/5 hover:text-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
              aria-label="Notifications"
            >
              <BellIcon />
            </button>
            <button
              className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-white/5 hover:text-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
              aria-label="Inbox"
            >
              <InboxIcon />
            </button>
            <div className="mx-2 h-5 w-px bg-white/10" />
            <UserButton />
          </div>
        </header>

        {/* Mobile top bar — hidden md+ */}
        <header className="flex md:hidden h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4">
          <Link to="/" className="text-[15px] font-bold tracking-tight text-slate-900">
            Stratos
          </Link>
          <UserButton />
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto px-6 py-7 lg:px-10 lg:py-8">
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
