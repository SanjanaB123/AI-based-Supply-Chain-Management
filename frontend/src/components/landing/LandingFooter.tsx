import { Link } from 'react-router-dom';

// ── Data ───────────────────────────────────────────────────────────────────────

const PRODUCT_LINKS = [
  { label: 'Platform',      href: '#features'  },
  { label: 'How it works',  href: '#workflow'  },
  { label: 'Features',      href: '#features'  },
] as const;

const PROJECT_LINKS = [
  { label: 'Team',          href: '#team'      },
  { label: 'Documentation', href: '#'          },
] as const;

// ── Component ──────────────────────────────────────────────────────────────────

export default function LandingFooter() {
  const year = new Date().getFullYear();

  return (
    <footer className="border-t border-slate-200 bg-slate-50">
      <div className="mx-auto max-w-480 px-6 lg:px-10">

        {/* Main footer grid */}
        <div className="grid grid-cols-1 gap-10 py-16 sm:grid-cols-2 md:grid-cols-4 md:gap-12 md:py-20">

          {/* Brand block — spans 2 cols on md+ */}
          <div className="sm:col-span-2 md:col-span-2">

            {/* Wordmark */}
            <Link
              to="/"
              className="inline-flex items-baseline gap-1.5 rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              <span className="text-[16px] font-bold tracking-tight text-slate-900">Stratos</span>
              <span className="text-[9px] font-semibold uppercase tracking-widest text-blue-600">AI</span>
            </Link>

            {/* Tagline */}
            <p className="mt-4 max-w-xs text-[13.5px] leading-relaxed text-slate-500">
              AI-powered supply chain intelligence. Predict disruption, optimize
              inventory, and make confident decisions — before problems arrive.
            </p>

            {/* Project attribution pill */}
            <div className="mt-6 inline-flex items-center gap-2.5 rounded-full border border-slate-200 bg-white px-4 py-2">
              <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
              <span className="text-[11.5px] font-medium text-slate-600">
                Northeastern University
              </span>
              <span className="h-3.5 w-px bg-slate-200" />
              <span className="text-[11.5px] text-slate-400">
                MLOps · Spring 2026
              </span>
            </div>
          </div>

          {/* Product links */}
          <div>
            <p className="mb-5 text-[11px] font-semibold uppercase tracking-widest text-slate-400">
              Product
            </p>
            <ul className="space-y-3.5">
              {PRODUCT_LINKS.map((item) => (
                <li key={item.label}>
                  <a
                    href={item.href}
                    className="text-[13.5px] text-slate-500 transition-colors duration-150 hover:text-slate-900"
                  >
                    {item.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Project links */}
          <div>
            <p className="mb-5 text-[11px] font-semibold uppercase tracking-widest text-slate-400">
              Project
            </p>
            <ul className="space-y-3.5">
              {PROJECT_LINKS.map((item) => (
                <li key={item.label}>
                  <a
                    href={item.href}
                    className="text-[13.5px] text-slate-500 transition-colors duration-150 hover:text-slate-900"
                  >
                    {item.label}
                  </a>
                </li>
              ))}

              <li>
                <a
                  href="https://github.com/SanjanaB123/AI-based-Supply-Chain-Management"
                  target='_blank'
                  rel='noreferrer noopener'
                  className="text-[13.5px] text-slate-500 transition-colors duration-150 hover:text-slate-900 underline"
                >
                  GitHub ↗
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col gap-2.5 border-t border-slate-200 py-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-[12px] text-slate-400">
            © {year} Stratos AI. All rights reserved.
          </p>
          <p className="text-[12px] text-slate-400">
            MLOps Final Project · Northeastern University · Spring 2026
          </p>
        </div>

      </div>
    </footer>
  );
}
