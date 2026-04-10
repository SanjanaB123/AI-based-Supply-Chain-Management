import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@clerk/clerk-react';
import { gsap } from 'gsap';

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

// ── Data ───────────────────────────────────────────────────────────────────────

const NAV_LINKS = [
  { label: 'Platform',     href: '#features'  },
  { label: 'How it works', href: '#workflow'  },
  { label: 'Team',         href: '#team'      },
] as const;

// ── Component ──────────────────────────────────────────────────────────────────

export default function LandingNav() {
  const { isSignedIn, isLoaded } = useAuth();
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
          ? 'border-slate-200 bg-white/96 shadow-sm shadow-slate-900/6 backdrop-blur-md'
          : 'border-slate-200/70 bg-white/80 backdrop-blur-md',
      ].join(' ')}
    >
      <div className="mx-auto flex h-full max-w-480 items-center justify-between px-6 lg:px-10">

        {/* ── Brand ─────────────────────────────────────────────────────────── */}
        <Link
          to="/"
          className="flex items-baseline gap-1.5 rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        >
          <span className="text-[15px] font-bold tracking-tight text-slate-900">Stratos</span>
          <span className="text-[9px] font-semibold uppercase tracking-widest text-blue-600">AI</span>
        </Link>

        {/* ── Center nav ────────────────────────────────────────────────────── */}
        <div className="hidden md:flex items-center gap-8">
          {NAV_LINKS.map((item) => (
            <a
              key={item.label}
              href={item.href}
              className="text-[13px] font-medium text-slate-500 transition-colors duration-150 hover:text-slate-900"
            >
              {item.label}
            </a>
          ))}
        </div>

        {/* ── Right: auth-aware CTAs ─────────────────────────────────────────── */}
        <div className="flex items-center gap-3">
          {!isLoaded ? (
            <div className="h-8 w-28 animate-pulse rounded-lg bg-slate-100" />
          ) : isSignedIn ? (
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-4 py-2 text-[13px] font-semibold text-white transition-all duration-150 hover:bg-slate-800 hover:-translate-y-px"
            >
              Go to Dashboard
              <ArrowRightIcon />
            </Link>
          ) : (
            <>
              <Link
                to="/sign-in"
                className="hidden sm:block text-[13px] font-medium text-slate-500 transition-colors duration-150 hover:text-slate-900"
              >
                Sign in
              </Link>
              <Link
                to="/sign-up"
                className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-4 py-2 text-[13px] font-semibold text-white transition-all duration-150 hover:bg-slate-800 hover:-translate-y-px"
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
