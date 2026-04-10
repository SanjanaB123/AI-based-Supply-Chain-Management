import { Link, Outlet } from 'react-router-dom';

// ── Stat callouts shown on the branded left panel ─────────────────────────────

const STATS = [
  { value: '94%',  label: 'Forecast accuracy' },
  { value: '12ms', label: 'Alert latency'      },
  { value: '3.4×', label: 'Faster decisions'   },
];

// ── Layout ────────────────────────────────────────────────────────────────────

export default function AuthLayout() {
  return (
    <div className="flex min-h-screen w-full">

      {/* ── Left: branded visual panel (md+) ─────────────────────────────── */}
      <aside
        aria-hidden="true"
        className="relative hidden md:flex md:w-[46%] lg:w-[48%] xl:w-[50%] flex-col overflow-hidden select-none"
      >
        {/* Background image */}
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: "url('/images/auth-bg.png')" }}
        />

        {/* Dark base overlay — ensures text legibility over any image */}
        <div className="absolute inset-0 bg-slate-950/70" />

        {/* Warm blue gradient pooling at the bottom — adds depth */}
        <div className="absolute inset-0 bg-linear-to-t from-blue-950/55 via-transparent to-transparent" />

        {/* Content layer */}
        <div className="relative flex h-full flex-col justify-between px-10 py-11 xl:px-14">

          {/* Brand wordmark */}
          <Link
            to="/"
            tabIndex={-1}
            className="inline-flex items-baseline gap-2 focus:outline-none"
          >
            <span className="text-sm font-extrabold tracking-tight text-white md:text-[17px] xl:text-[33px]">Stratos</span>
            <span className="text-[8px] font-semibold uppercase tracking-widest text-blue-400 md:text-[10px] xl:text-[17px]">
              AI
            </span>
          </Link>

          {/* Value proposition block */}
          <div className="space-y-7">
            <div className="space-y-3">
              <p className="text-[9px] font-semibold uppercase tracking-widest text-blue-400 md:text-[10px] xl:text-[15px]">
                Supply chain intelligence
              </p>
              <h2 className="text-xl font-bold leading-tight tracking-tight text-white italic md:text-[28px] xl:text-[52px]">
                Clarity across your<br />entire supply chain.
              </h2>
              <p className="max-w-xl text-xs leading-relaxed text-slate-300 md:text-[13px] xl:text-[20px]">
                Predictive demand forecasting, real-time inventory intelligence,
                and risk management — unified in one platform.
              </p>
            </div>

            {/* Stats row */}
            <div className="flex gap-8 xl:gap-12">
              {STATS.map((s) => (
                <div key={s.label}>
                  <p className="text-base font-bold text-white md:text-2xl xl:text-[40px]">{s.value}</p>
                  <p className="mt-0.5 text-[10px] leading-tight text-slate-400 md:text-[11px] xl:text-[14px]">{s.label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Footer */}
          <p className="text-[11px] text-slate-600">
            © 2025 Stratos · AI supply chain intelligence
          </p>
        </div>
      </aside>

      {/* ── Right: auth form panel ────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col items-center justify-center bg-white px-6 py-12 sm:px-10">

        {/* Mobile-only brand header (hidden on md+) */}
        <div className="mb-8 text-center md:hidden">
          <Link to="/" className="inline-flex items-baseline gap-2">
            <span className="text-xl font-bold tracking-tight text-slate-900">Stratos</span>
            <span className="text-[10px] font-semibold uppercase tracking-widest text-blue-500">
              AI
            </span>
          </Link>
          <p className="mt-1 text-sm text-slate-500">AI-powered supply chain intelligence</p>
        </div>

        {/* Clerk form — constrained to Clerk's natural card width */}
        <div className="w-full max-w-100">
          <Outlet />
        </div>

        {/* Bottom footnote */}
        <p className="mt-10 text-center text-[11px] text-slate-400">
          © 2025 Stratos · All rights reserved
        </p>
      </div>
    </div>
  );
}
