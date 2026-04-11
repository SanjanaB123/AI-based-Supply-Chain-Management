import type { LeadTimeRiskResponse, LeadTimeRiskProduct } from '../../types/inventory';

// ── Helpers ───────────────────────────────────────────────────────────────────

function supplyBarClass(health: LeadTimeRiskProduct['stock_health']): string {
  if (health === 'critical') return 'bg-red-500';
  if (health === 'low')      return 'bg-amber-500';
  return 'bg-emerald-500';
}

// ── Risk row ──────────────────────────────────────────────────────────────────

function RiskRow({ p, maxDays }: { p: LeadTimeRiskProduct; maxDays: number }) {
  const gap      = Math.round(p.days_of_supply - p.lead_time_days);
  const isAtRisk = gap <= 0;
  const isBorder = gap > 0 && gap <= 7;

  const supplyPct = Math.min(100, (p.days_of_supply / maxDays) * 100);
  const leadPct   = Math.min(100, (p.lead_time_days / maxDays) * 100);

  return (
    <div
      className={`rounded-lg border px-3 py-2.5 transition-colors ${
        isAtRisk
          ? 'border-red-200 dark:border-red-900/50 bg-red-100/60 dark:bg-red-900/20'
          : isBorder
          ? 'border-amber-100 dark:border-amber-900/40 bg-amber-100/40 dark:bg-amber-900/15'
          : 'border-slate-100 dark:border-slate-700 bg-white dark:bg-slate-800/60 hover:bg-slate-50/60 dark:hover:bg-slate-700/60'
      }`}
    >
      {/* Header row */}
      <div className="mb-2 flex items-center justify-between">
        <div className="min-w-0 flex items-baseline gap-2">
          <span className="text-[12px] font-semibold text-slate-800 dark:text-slate-200 leading-tight truncate">
            {p.product_id}
          </span>
          <span className="shrink-0 rounded-full bg-slate-100 dark:bg-slate-700 px-2 py-0.5 text-[10px] text-slate-500 dark:text-slate-400">
            {p.category}
          </span>
        </div>

        {isAtRisk ? (
          <span className="ml-2 shrink-0 rounded-full bg-red-100 dark:bg-red-900/40 px-2.5 py-0.5 text-[10px] font-semibold text-red-600 dark:text-red-400">
            AT RISK
          </span>
        ) : isBorder ? (
          <span className="ml-2 shrink-0 rounded-full bg-amber-100 dark:bg-amber-900/40 px-2.5 py-0.5 text-[10px] font-semibold text-amber-600 dark:text-amber-400">
            +{gap}d buffer
          </span>
        ) : (
          <span className="ml-2 shrink-0 text-[10px] text-slate-400 dark:text-slate-500 tabular-nums">
            +{gap}d
          </span>
        )}
      </div>

      {/* Dual bar comparison */}
      <div className="space-y-1">
        {/* Supply bar */}
        <div className="flex items-center gap-2">
          <span className="w-11 shrink-0 text-[10px] text-slate-400 dark:text-slate-500 text-right">Supply</span>
          <div className="flex-1 h-1.5 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${supplyBarClass(p.stock_health)}`}
              style={{ width: `${supplyPct}%` }}
            />
          </div>
          <span className="w-8 shrink-0 text-right text-[11px] font-semibold tabular-nums text-slate-700 dark:text-slate-300">
            {Math.round(p.days_of_supply)}d
          </span>
        </div>
        {/* Lead time bar */}
        <div className="flex items-center gap-2">
          <span className="w-11 shrink-0 text-[10px] text-slate-400 dark:text-slate-500 text-right">Lead T.</span>
          <div className="flex-1 h-1.5 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden">
            <div
              className="h-full rounded-full bg-blue-400 dark:bg-blue-500 transition-all duration-500"
              style={{ width: `${leadPct}%` }}
            />
          </div>
          <span className="w-8 shrink-0 text-right text-[11px] font-semibold tabular-nums text-slate-700 dark:text-slate-300">
            {p.lead_time_days}d
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data: LeadTimeRiskResponse;
}

const MAX_DISPLAY = 8;

export default function LeadTimeRiskModule({ data }: Props) {
  if (data.products.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-slate-400 dark:text-slate-500">
        No lead-time data available for this store.
      </div>
    );
  }

  const sorted = [...data.products].sort(
    (a, b) => (a.days_of_supply - a.lead_time_days) - (b.days_of_supply - b.lead_time_days),
  );

  const atRisk  = sorted.filter(p => p.days_of_supply <= p.lead_time_days);
  const maxDays = Math.max(...data.products.map(p => Math.max(p.days_of_supply, p.lead_time_days)), 1);

  const displayed = sorted.slice(0, MAX_DISPLAY);

  return (
    <div className="space-y-4">

      {/* ── Alert / all-clear banner ────────────────────────────────────────── */}
      {atRisk.length > 0 ? (
        <div className="flex items-start gap-3 rounded-lg border border-red-200 dark:border-red-900/50 bg-red-100 dark:bg-red-900/20 px-4 py-3 text-sm text-red-700 dark:text-red-400">
          <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
            <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm-.75 3.75a.75.75 0 0 1 1.5 0v3.5a.75.75 0 0 1-1.5 0v-3.5zm.75 7a.875.875 0 1 1 0-1.75.875.875 0 0 1 0 1.75z" />
          </svg>
          <span>
            <strong>{atRisk.length}</strong> item{atRisk.length !== 1 ? 's' : ''} will stock out
            before their replenishment order arrives.
          </span>
        </div>
      ) : (
        <div className="flex items-center gap-3 rounded-lg border border-emerald-200 dark:border-emerald-900/50 bg-emerald-100 dark:bg-emerald-900/20 px-4 py-3 text-sm text-emerald-700 dark:text-emerald-400">
          <span className="h-2 w-2 shrink-0 rounded-full bg-emerald-500" />
          All products have sufficient supply to cover their lead times.
        </div>
      )}

      {/* ── Legend ─────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-4 text-[10px] text-slate-400 dark:text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-4 rounded-full bg-emerald-500 inline-block" />
          Supply
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-4 rounded-full bg-blue-400 inline-block" />
          Lead time
        </span>
        <span className="ml-auto text-[10px] text-slate-400 dark:text-slate-500">sorted by risk</span>
      </div>

      {/* ── Risk rows ──────────────────────────────────────────────────────── */}
      <div className="space-y-2">
        {displayed.map(p => (
          <RiskRow key={p.product_id} p={p} maxDays={maxDays} />
        ))}
      </div>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <p className="text-[11px] text-slate-400 dark:text-slate-500">
        {displayed.length} of {data.products.length} products · rows where supply bar is shorter
        than lead time bar are at risk
      </p>
    </div>
  );
}
