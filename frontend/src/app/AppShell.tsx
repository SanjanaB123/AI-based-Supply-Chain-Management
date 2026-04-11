import { UserButton, useUser } from '@clerk/clerk-react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { TopBarProvider, useTopBar } from './TopBarContext';
import { InventoryDataProvider } from '../hooks/InventoryDataContext';
import { useTheme } from './theme/useTheme';
import { useChatAssistant } from '../hooks/useChatAssistant';
import { DashboardFooterBar } from '../components/chat/DashboardFooterBar';
import { ChatPanel } from '../components/chat/ChatPanel';

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

function HeadsetIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path d="M2.5 8V7a5 5 0 0 1 10 0v1" stroke="currentColor" strokeWidth="1.3" />
      <rect x="1.5" y="8" width="2.5" height="4" rx="1" stroke="currentColor" strokeWidth="1.3" />
      <rect x="11" y="8" width="2.5" height="4" rx="1" stroke="currentColor" strokeWidth="1.3" />
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
      { label: 'Analytics',  path: '/analytics', icon: <BarChartIcon /> },
      { label: 'Inventory',  path: '/inventory', icon: <BoxIcon /> },
      { label: 'Risk',       path: '/risk',       icon: <RiskIcon /> },
      { label: 'Variance',   path: '/variance',   icon: <VarianceIcon /> },
    ],
  },
  {
    label: 'Intelligence',
    items: [
      { label: 'AI Assistant', path: '/ai', icon: <SparkleIcon /> },
    ],
  },
  {
    label: 'Help & Support',
    items: [
      { label: 'Contact Support',  path: '/support', icon: <HeadsetIcon /> },
    ],
  },
];

// ── NavLink ────────────────────────────────────────────────────────────────────

function NavLink({ item, active }: { item: NavItem; active: boolean }) {
  if (item.soon) {
    return (
      <div className="flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] font-medium opacity-40 cursor-not-allowed select-none">
        <span className="text-slate-800 dark:text-slate-300">{item.icon}</span>
        <span className="text-slate-800 dark:text-slate-300">{item.label}</span>
        <span className="ml-auto rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-widest text-slate-400 dark:text-slate-500 ring-1 ring-slate-300 dark:ring-slate-700">
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
          ? 'bg-blue-200 text-blue-700 dark:bg-blue-900/40 dark:text-blue-400'
          : 'text-slate-600 dark:text-slate-400 hover:bg-slate-200/60 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'
      }`}
    >
      <span className={active ? 'text-blue-500 dark:text-blue-400' : 'text-slate-400 dark:text-slate-500'}>
        {item.icon}
      </span>
      {item.label}
    </Link>
  );
}

// ── Inner shell ───────────────────────────────────────────────────────────────

function AppShellInner() {
  const location = useLocation();
  const { topBarSlot, pageTitle, pageSubtitle } = useTopBar();
  const { user, isLoaded } = useUser();
  const { theme } = useTheme();
  const chat = useChatAssistant();

  const displayName = isLoaded
    ? user?.firstName
      ? `${user.firstName}${user.lastName ? ' ' + user.lastName : ''}`
      : user?.username ?? user?.fullName ?? 'User'
    : '';
  const userEmail = isLoaded ? (user?.primaryEmailAddress?.emailAddress ?? '') : '';

  return (
    <div className="flex h-screen overflow-hidden bg-slate-100 dark:bg-slate-900">

      {/* ── Sidebar (md+) ──────────────────────────────────────────────────── */}
      <aside className="hidden md:flex md:w-60 md:shrink-0 md:flex-col bg-white dark:bg-slate-950 border-r border-slate-200 dark:border-slate-800">

        {/* Brand */}
        <div className="flex h-14 shrink-0 items-center px-5 border-b border-slate-200 dark:border-slate-800">
          <Link to="/" className="focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 rounded-sm">
            <img
              src={theme === 'dark' ? '/icons/logo-dark.png' : '/icons/logo-light.png'}
              alt="Stratos"
              className="h-10 w-auto"
            />
          </Link>
        </div>

        {/* Nav sections */}
        <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-5">
          {NAV_SECTIONS.map((section, si) => (
            <div key={si}>
              {section.label && (
                <p className="mb-1 px-3 text-[9px] font-semibold uppercase tracking-widest text-slate-900 dark:text-slate-400">
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

        {/* Bottom: Settings + theme toggle + user account */}
        <div className="shrink-0 border-t border-slate-200 dark:border-slate-800">
          <div className="px-3 pt-2 pb-1">
          </div>


          <div className="flex items-center gap-3 border-t border-slate-200 dark:border-slate-800 px-4 py-3">
            <UserButton />
            <div className="min-w-0 flex-1">
              <p className="truncate text-[13px] font-semibold leading-tight text-slate-800 dark:text-slate-200">
                {displayName}
              </p>
              {userEmail && (
                <p className="truncate text-[11px] leading-tight text-slate-400 dark:text-slate-500">{userEmail}</p>
              )}
            </div>
          </div>
        </div>
      </aside>

      {/* ── Right side ─────────────────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col overflow-hidden">

        {/* Desktop top bar */}
        <header className="hidden md:flex h-14 shrink-0 items-center gap-3 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 px-5">

          {/* Page title + subtitle */}
          <div className="flex min-w-0 flex-col justify-center">
            <h1 className="text-[15px] font-semibold leading-tight tracking-tight text-slate-900 dark:text-slate-100 xl:text-base">
              {pageTitle || 'Dashboard'}
            </h1>
            {pageSubtitle && (
              <p className="text-[11px] leading-tight text-slate-400 dark:text-slate-500 xl:text-[12px]">
                {pageSubtitle}
              </p>
            )}
          </div>

          {/* Divider before store selector */}
          {topBarSlot && <div className="h-5 w-px shrink-0 bg-slate-200 dark:bg-slate-700" />}

          {/* Store selector slot */}
          {topBarSlot && (
            <div className="flex shrink-0 items-center gap-3">
              {topBarSlot}
            </div>
          )}

          {/* Push utility actions to the right */}
          <div className="flex-1" />

          {/* User */}
          <div className="flex items-center gap-0.5">
            <UserButton />
          </div>
        </header>

        {/* Mobile top bar */}
        <header className="flex md:hidden h-14 shrink-0 items-center justify-between border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 px-4">
          <Link to="/" className="focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 rounded-sm">
            <img
              src={theme === 'dark' ? '/icons/logo-dark.png' : '/icons/logo-light.png'}
              alt="Stratos"
              className="h-7 w-auto"
            />
          </Link>
          <div className="flex items-center gap-2">
            <UserButton />
          </div>
        </header>

        {/* Page content */}
        <main className={`flex-1 bg-slate-50 dark:bg-slate-900 ${
          location.pathname === '/ai'
            ? 'overflow-hidden p-0'
            : 'overflow-auto px-6 py-7 lg:px-10 lg:py-8'
        }`}>
          <Outlet />
        </main>

        {/* Footer bar — hidden on /ai page */}
        {location.pathname !== '/ai' && (
          <DashboardFooterBar isOpen={chat.isOpen} onToggle={chat.toggle} />
        )}
      </div>

      {/* Backdrop + floating chat panel — hidden on /ai page */}
      {location.pathname !== '/ai' && (
        <>
          <div
            aria-hidden="true"
            className={`fixed inset-0 z-40 bg-slate-900/20 backdrop-blur-[2px] pointer-events-none transition-opacity duration-300 ${
              chat.isOpen ? 'opacity-100' : 'opacity-0'
            }`}
          />
          <ChatPanel
            isOpen={chat.isOpen}
            messages={chat.messages}
            isSending={chat.isSending}
            error={chat.error}
            onClose={chat.close}
            onClear={chat.clearChat}
            onSend={chat.sendMessage}
          />
        </>
      )}
    </div>
  );
}

// ── AppShell (exported) ───────────────────────────────────────────────────────

export default function AppShell() {
  return (
    <TopBarProvider>
      <InventoryDataProvider>
        <AppShellInner />
      </InventoryDataProvider>
    </TopBarProvider>
  );
}
