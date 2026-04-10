import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@clerk/clerk-react';
import { gsap } from 'gsap';

// ── Icons ──────────────────────────────────────────────────────────────────────

function ArrowRightIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path
        d="M3 7h8M8 4l3 3-3 3"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ── Dashboard mock data ────────────────────────────────────────────────────────

interface KpiCardData {
  label: string;
  value: string;
  delta: string;
  positive: boolean;
}

const KPI_CARDS: KpiCardData[] = [
  { label: 'Forecast Acc.',  value: '94.2%', delta: '+1.3%', positive: true  },
  { label: 'Days of Supply', value: '18.4d', delta: '+2.1d', positive: true  },
  { label: 'Stock at Risk',  value: '$84K',  delta: '+$12K', positive: false },
  { label: 'Lead Time',      value: '6.3d',  delta: '−0.4d', positive: true  },
];

const BAR_HEIGHTS = [42, 58, 47, 72, 63, 80, 68, 88, 55, 92, 74, 84];

interface SidebarItemData {
  label: string;
  active: boolean;
  soon: boolean;
}

const SIDEBAR_ITEMS: SidebarItemData[] = [
  { label: 'Dashboard',    active: true,  soon: false },
  { label: 'Analytics',   active: false, soon: true  },
  { label: 'Inventory',   active: false, soon: true  },
  { label: 'Risk',        active: false, soon: true  },
  { label: 'AI Assistant', active: false, soon: true },
];

interface TableRowData {
  sku: string;
  stock: string;
  forecast: string;
  status: string;
  statusColor: string;
}

const TABLE_ROWS: TableRowData[] = [
  { sku: 'SKU-1042-A', stock: '1,240', forecast: '+8%', status: 'Healthy',  statusColor: 'text-emerald-400' },
  { sku: 'SKU-8821-C', stock: '186',   forecast: '−2%', status: 'Low',      statusColor: 'text-amber-400'   },
  { sku: 'SKU-5503-B', stock: '42',    forecast: '−9%', status: 'Critical', statusColor: 'text-red-400'     },
];

// ── Dashboard preview ─────────────────────────────────────────────────────────

function DashboardPreview() {
  return (
    <div className="relative mx-auto w-full max-w-5xl">

      {/* Soft blue glow behind the frame */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -inset-x-8 -bottom-10 top-8"
        style={{
          background:
            'radial-gradient(ellipse 75% 45% at 50% 0%, rgba(59,130,246,0.11) 0%, transparent 70%)',
        }}
      />

      {/* Browser chrome — light frame */}
      <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl shadow-slate-900/[0.12] ring-1 ring-slate-200/80">

        {/* Browser top bar */}
        <div className="flex items-center gap-3 border-b border-slate-200 bg-slate-50/90 px-4 py-2.5">
          <div className="flex gap-1.5 shrink-0">
            <div className="h-2.5 w-2.5 rounded-full bg-red-400/70" />
            <div className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
            <div className="h-2.5 w-2.5 rounded-full bg-green-400/70" />
          </div>
          <div className="flex flex-1 justify-center">
            <div className="flex w-full max-w-xs items-center justify-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1">
              <svg width="9" height="9" viewBox="0 0 9 9" fill="none" aria-hidden="true">
                <path
                  d="M4.5 1.5a3 3 0 100 6 3 3 0 000-6z"
                  stroke="#94a3b8"
                  strokeWidth="1"
                />
                <path d="M3 4.5h3M4.5 3v3" stroke="#94a3b8" strokeWidth="1" strokeLinecap="round" />
              </svg>
              <span className="text-[10px] text-slate-400">app.stratos.ai/dashboard</span>
            </div>
          </div>
          <div className="w-14 shrink-0" />
        </div>

        {/* Product UI — stays dark to represent the actual product */}
        <div className="flex bg-slate-950" style={{ height: '320px' }}>

          {/* Sidebar */}
          <aside className="hidden sm:flex w-40 shrink-0 flex-col border-r border-white/[0.05] bg-slate-950/60 px-2 py-3">
            <div className="mb-4 flex items-center gap-1 px-1.5">
              <span className="text-[11px] font-bold tracking-tight text-white">Stratos</span>
              <span className="text-[7px] font-semibold uppercase tracking-wider text-blue-400">AI</span>
            </div>
            <div className="space-y-px">
              {SIDEBAR_ITEMS.map((item) => (
                <div
                  key={item.label}
                  className={[
                    'flex items-center gap-2 rounded-md px-2 py-1.5 text-[10px] font-medium',
                    item.active ? 'bg-blue-600/15 text-blue-400' : 'text-slate-600',
                  ].join(' ')}
                >
                  <div
                    className={[
                      'h-1.5 w-1.5 shrink-0 rounded-full',
                      item.active ? 'bg-blue-400' : 'bg-slate-700',
                    ].join(' ')}
                  />
                  {item.label}
                  {item.soon && (
                    <span className="ml-auto text-[7px] text-slate-700">Soon</span>
                  )}
                </div>
              ))}
            </div>
          </aside>

          {/* Main content */}
          <div className="flex-1 overflow-hidden p-3.5">

            {/* Page header */}
            <div className="mb-3 flex items-center justify-between">
              <div>
                <p className="text-[11px] font-semibold text-white">Overview</p>
                <p className="text-[9px] text-slate-500">All stores · Live</p>
              </div>
              <div className="flex items-center gap-2">
                <div className="rounded border border-white/[0.07] bg-slate-800/50 px-2 py-1 text-[9px] text-slate-400">
                  Export CSV
                </div>
                <div className="h-5 w-5 rounded-full bg-slate-700/60 ring-1 ring-slate-600/30" />
              </div>
            </div>

            {/* KPI cards */}
            <div className="mb-3 grid grid-cols-4 gap-2">
              {KPI_CARDS.map((kpi) => (
                <div
                  key={kpi.label}
                  className="rounded-lg border border-white/[0.06] bg-slate-800/30 px-2.5 py-2"
                >
                  <p className="mb-0.5 text-[8px] leading-tight text-slate-500">{kpi.label}</p>
                  <p className="text-[13px] font-bold leading-tight text-white">{kpi.value}</p>
                  <p className={['mt-0.5 text-[8px] font-medium', kpi.positive ? 'text-emerald-400' : 'text-red-400'].join(' ')}>
                    {kpi.delta}
                  </p>
                </div>
              ))}
            </div>

            {/* Charts row */}
            <div className="mb-2 grid grid-cols-5 gap-2">

              {/* Bar chart */}
              <div className="col-span-3 rounded-lg border border-white/[0.06] bg-slate-800/20 p-3">
                <p className="mb-2 text-[9px] font-medium text-slate-400">Inventory Trend</p>
                <div className="flex items-end gap-1" style={{ height: '64px' }}>
                  {BAR_HEIGHTS.map((h, i) => (
                    <div
                      key={i}
                      className="flex-1 rounded-sm"
                      style={{
                        height: `${h}%`,
                        background:
                          i >= BAR_HEIGHTS.length - 2
                            ? 'rgba(59,130,246,0.80)'
                            : 'rgba(59,130,246,0.20)',
                      }}
                    />
                  ))}
                </div>
              </div>

              {/* Donut */}
              <div className="col-span-2 rounded-lg border border-white/[0.06] bg-slate-800/20 p-3">
                <p className="mb-2 text-[9px] font-medium text-slate-400">Stock Health</p>
                <div className="flex items-center gap-3">
                  <div className="relative h-11 w-11 shrink-0">
                    <svg viewBox="0 0 36 36" className="h-full w-full -rotate-90">
                      <circle cx="18" cy="18" r="13" fill="none" stroke="rgb(30,41,59)" strokeWidth="6" />
                      <circle cx="18" cy="18" r="13" fill="none" stroke="rgba(52,211,153,0.80)" strokeWidth="6" strokeDasharray="50 50" />
                      <circle cx="18" cy="18" r="13" fill="none" stroke="rgba(251,191,36,0.70)" strokeWidth="6" strokeDasharray="25 75" strokeDashoffset="-50" />
                      <circle cx="18" cy="18" r="13" fill="none" stroke="rgba(248,113,113,0.65)" strokeWidth="6" strokeDasharray="10 90" strokeDashoffset="-75" />
                    </svg>
                    <span className="absolute inset-0 flex items-center justify-center text-[8px] font-bold text-white">60%</span>
                  </div>
                  <div className="space-y-1">
                    {[
                      { label: 'Healthy',  color: 'bg-emerald-400' },
                      { label: 'Low',      color: 'bg-amber-400'   },
                      { label: 'Critical', color: 'bg-red-400'     },
                    ].map((item) => (
                      <div key={item.label} className="flex items-center gap-1.5">
                        <div className={['h-1.5 w-1.5 rounded-full shrink-0', item.color].join(' ')} />
                        <span className="text-[8px] text-slate-400">{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Table */}
            <div className="overflow-hidden rounded-lg border border-white/[0.05] bg-slate-800/15">
              <div className="grid grid-cols-4 gap-2 border-b border-white/[0.05] px-3 py-1.5">
                {['Product', 'Stock', 'Forecast', 'Status'].map((col) => (
                  <span key={col} className="text-[8px] font-medium text-slate-600">{col}</span>
                ))}
              </div>
              {TABLE_ROWS.map((row, i) => (
                <div
                  key={row.sku}
                  className={['grid grid-cols-4 gap-2 px-3 py-1.5', i < TABLE_ROWS.length - 1 ? 'border-b border-white/[0.03]' : ''].join(' ')}
                >
                  <span className="truncate text-[8px] text-slate-300">{row.sku}</span>
                  <span className="text-[8px] text-slate-400">{row.stock}</span>
                  <span className="text-[8px] text-slate-400">{row.forecast}</span>
                  <span className={['text-[8px] font-medium', row.statusColor].join(' ')}>{row.status}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom fade — blends frame edge into white page background */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 bottom-0 h-28 rounded-b-2xl"
        style={{ background: 'linear-gradient(to bottom, transparent 0%, rgba(255,255,255,0.92) 100%)' }}
      />
    </div>
  );
}

// ── Hero section ──────────────────────────────────────────────────────────────

export default function HeroSection() {
  const { isSignedIn, isLoaded } = useAuth();

  const eyebrowRef  = useRef<HTMLDivElement>(null);
  const headlineRef = useRef<HTMLHeadingElement>(null);
  const subRef      = useRef<HTMLParagraphElement>(null);
  const ctaRef      = useRef<HTMLDivElement>(null);
  const trustRef    = useRef<HTMLDivElement>(null);
  const previewRef  = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const elements = [
      eyebrowRef.current,
      headlineRef.current,
      subRef.current,
      ctaRef.current,
      trustRef.current,
    ];

    // Set initial hidden state
    gsap.set(elements, { opacity: 0, y: 22 });
    gsap.set(previewRef.current, { opacity: 0, y: 44 });

    // Staggered entrance timeline
    const tl = gsap.timeline({ defaults: { ease: 'power3.out' }, delay: 0.15 });
    elements.forEach((el, i) => {
      tl.to(el, { opacity: 1, y: 0, duration: 0.6 }, i * 0.09);
    });
    tl.to(previewRef.current, { opacity: 1, y: 0, duration: 0.85, ease: 'power2.out' }, 0.35);

    return () => { tl.kill(); };
  }, []);

  return (
    <section className="relative overflow-hidden bg-white pt-16">

      {/* Subtle radial hero tint */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 h-[680px]"
        style={{
          background:
            'radial-gradient(ellipse 85% 55% at 50% 0%, rgba(59,130,246,0.07) 0%, transparent 70%)',
        }}
      />

      {/* Very faint dot grid */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage:
            'radial-gradient(circle, rgba(148,163,184,0.09) 1px, transparent 1px)',
          backgroundSize: '28px 28px',
        }}
      />

      <div className="relative mx-auto max-w-480 px-6 lg:px-10">

        {/* ── Text block ──────────────────────────────────────────────────── */}
        <div className="flex flex-col items-center pt-20 pb-12 text-center">

          {/* Eyebrow badge */}
          <div
            ref={eyebrowRef}
            className="mb-7 inline-flex items-center gap-2.5 rounded-full border border-blue-200 bg-blue-50 px-4 py-1.5"
          >
            <span className="inline-block h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-blue-500" />
            <span className="text-[12px] font-medium tracking-wide text-blue-700">
              Supply chain intelligence · Powered by AI
            </span>
          </div>

          {/* Headline */}
          <h1
            ref={headlineRef}
            className="max-w-4xl text-[48px] font-bold leading-[1.07] tracking-tight text-slate-900 sm:text-[58px] lg:text-[68px] xl:text-[76px]"
          >
            Know every risk<br />
            <span className="text-blue-600">before it arrives.</span>
          </h1>

          {/* Subheadline */}
          <p
            ref={subRef}
            className="mx-auto mt-6 max-w-2xl text-[17px] leading-relaxed text-slate-500"
          >
            Stratos unifies predictive demand forecasting, real-time inventory
            intelligence, and supplier risk monitoring in one confident platform
            built for modern supply chains.
          </p>

          {/* CTA group */}
          <div
            ref={ctaRef}
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            {!isLoaded ? (
              <div className="h-12 w-44 animate-pulse rounded-xl bg-slate-100" />
            ) : isSignedIn ? (
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-7 py-3.5 text-[15px] font-semibold text-white shadow-lg shadow-slate-900/20 transition-all duration-200 hover:bg-slate-800 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-slate-900/25"
              >
                Go to Dashboard
                <ArrowRightIcon />
              </Link>
            ) : (
              <>
                <Link
                  to="/sign-up"
                  className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-7 py-3.5 text-[15px] font-semibold text-white shadow-lg shadow-slate-900/20 transition-all duration-200 hover:bg-slate-800 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-slate-900/25"
                >
                  Start for free
                  <ArrowRightIcon />
                </Link>
                <Link
                  to="/sign-in"
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-7 py-3.5 text-[15px] font-semibold text-slate-700 shadow-sm transition-all duration-200 hover:border-slate-300 hover:bg-slate-50 hover:-translate-y-0.5"
                >
                  Sign in
                </Link>
              </>
            )}
          </div>

          {/* Trust strip */}
          <div
            ref={trustRef}
            className="mt-5 flex flex-wrap items-center justify-center gap-5 text-[12px] text-slate-400"
          >
            <span>No credit card required</span>
            <span className="hidden sm:block h-1 w-1 rounded-full bg-slate-300" />
            <span>14-day free trial</span>
            <span className="hidden sm:block h-1 w-1 rounded-full bg-slate-300" />
            <span>Cancel anytime</span>
          </div>
        </div>

        {/* ── Dashboard preview ────────────────────────────────────────────── */}
        <div ref={previewRef} className="pb-20 md:pb-28">
          <DashboardPreview />
        </div>
      </div>
    </section>
  );
}
