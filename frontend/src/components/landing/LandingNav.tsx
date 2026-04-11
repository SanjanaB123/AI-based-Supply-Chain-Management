import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@clerk/clerk-react';
import { gsap } from 'gsap';
import { useTheme } from '../../app/theme/useTheme';

// ── Icons ──────────────────────────────────────────────────────────────────────

function ArrowRightIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path
        d="M2.5 6h7M6.5 2.5L10 6l-3.5 3.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <circle cx="7.5" cy="7.5" r="2.5" stroke="currentColor" strokeWidth="1.3" />
      <path d="M7.5 1v1.5M7.5 12.5V14M1 7.5h1.5M12.5 7.5H14M3.05 3.05l1.06 1.06M10.89 10.89l1.06 1.06M10.89 4.11l1.06-1.06M3.05 11.95l1.06-1.06" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path d="M12.5 9A6 6 0 0 1 6 2.5a6 6 0 1 0 6.5 6.5z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
    </svg>
  );
}

// ── Data ───────────────────────────────────────────────────────────────────────

const NAV_LINKS = [
  { label: 'Platform',     href: '#features'  },
  { label: 'How it works', href: '#workflow'  },
] as const;

// ── Component ──────────────────────────────────────────────────────────────────

export default function LandingNav() {
  const { isSignedIn, isLoaded } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [scrolled, setScrolled] = useState(false);
  const navRef = useRef<HTMLElement>(null);

  // Scroll-aware shadow
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // GSAP entrance: slide down + fade in
  useEffect(() => {
    if (!navRef.current) return;
    gsap.fromTo(
      navRef.current,
      { y: -20, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.65, ease: 'power2.out', delay: 0.05 },
    );
  }, []);

  return (
    <nav
      ref={navRef}
      className={[
        'fixed inset-x-0 top-0 z-50 h-16 border-b transition-all duration-300',
        scrolled
          ? 'border-slate-200 dark:border-slate-800 bg-white/96 dark:bg-slate-950/96 shadow-sm shadow-slate-900/6 backdrop-blur-md'
          : 'border-slate-200/70 dark:border-slate-800/70 bg-white/80 dark:bg-slate-950/80 backdrop-blur-md',
      ].join(' ')}
    >
      <div className="mx-auto flex h-full max-w-480 items-center justify-between px-6 lg:px-10">

        {/* ── Brand ─────────────────────────────────────────────────────────── */}
        <Link
          to="/"
          className="rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        >
          <img
            src={theme === 'dark' ? '/icons/logo-dark.png' : '/icons/logo-light.png'}
            alt="Stratos"
            className="h-7 w-auto"
          />
        </Link>

        {/* ── Center nav ────────────────────────────────────────────────────── */}
        <div className="hidden md:flex items-center gap-8">
          {NAV_LINKS.map((item) => (
            <a
              key={item.label}
              href={item.href}
              className="text-[13px] font-medium text-slate-500 dark:text-slate-400 transition-colors duration-150 hover:text-slate-900 dark:hover:text-slate-100"
            >
              {item.label}
            </a>
          ))}
        </div>

        {/* ── Right: auth-aware CTAs ──────────────────────────────────────── */}
        <div className="flex items-center gap-2">
          {!isLoaded ? (
            <div className="h-8 w-28 animate-pulse rounded-lg bg-slate-100 dark:bg-slate-800" />
          ) : isSignedIn ? (
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 dark:bg-slate-100 px-4 py-2 text-[13px] font-semibold text-white dark:text-slate-900 transition-all duration-150 hover:bg-slate-800 dark:hover:bg-slate-200 hover:-translate-y-px"
            >
              Go to Dashboard
              <ArrowRightIcon />
            </Link>
          ) : (
            <>
              <Link
                to="/sign-in"
                className="hidden sm:block text-[13px] font-medium text-slate-500 dark:text-slate-400 transition-colors duration-150 hover:text-slate-900 dark:hover:text-slate-100"
              >
                Sign in
              </Link>
              <Link
                to="/sign-up"
                className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 dark:bg-slate-100 px-4 py-2 text-[13px] font-semibold text-white dark:text-slate-900 transition-all duration-150 hover:bg-slate-800 dark:hover:bg-slate-200 hover:-translate-y-px"
              >
                Get Started
                <ArrowRightIcon />
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
