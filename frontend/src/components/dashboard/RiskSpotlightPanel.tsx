import type { LeadTimeRiskResponse } from '../../types/inventory';

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data: LeadTimeRiskResponse;
}

export default function RiskSpotlightPanel({ data }: Props) {
  // Products where supply runs out before restock arrives
  const atRisk = data.products
    .filter(p => p.days_of_supply <= p.lead_time_days)
    .sort((a, b) => (a.days_of_supply - a.lead_time_days) - (b.days_of_supply - b.lead_time_days));

  if (atRisk.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2.5 py-8 text-center h-92">
        <span className="h-2 w-2 rounded-full bg-emerald-500" />
        <p className="text-sm text-slate-400">
          No stockout risk. All products have enough supply to cover their lead times.
        </p>
      </div>
    );
  }

  const shown = atRisk.slice(0, 6);

  return (
    <div className="space-y-3 h-92">
      {/* Alert banner */}
      <div className="flex items-center gap-2.5 rounded-lg border border-red-100 bg-red-100 px-3 py-2.5 text-sm">
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-red-500" />
        <span className="text-red-700">
          <strong>{atRisk.length}</strong> item{atRisk.length !== 1 ? 's' : ''} will stock out
          before restocking
        </span>
      </div>

      {/* Risk item list */}
      <div className="space-y-1.5">
        {shown.map(p => {
          const gap = Math.round(p.days_of_supply - p.lead_time_days); // negative = shortage
          return (
            <div
              key={p.product_id}
              className="flex items-center justify-between rounded-lg border border-slate-100 bg-white px-3 py-2.5"
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-slate-800">{p.product_id}</p>
                <p className="text-[11px] text-slate-400">{p.category}</p>
              </div>
              <div className="ml-3 shrink-0 text-right">
                <p className="text-sm font-semibold tabular-nums text-red-600">
                  {Math.round(p.days_of_supply)}d supply
                </p>
                <p className="text-[11px] text-slate-400">
                  {p.lead_time_days}d lead ·{' '}
                  <span className="text-red-500 font-medium">{gap}d gap</span>
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {atRisk.length > 6 && (
        <p className="text-center text-[11px] text-slate-400">
          +{atRisk.length - 6} more in Lead-Time Risk tab
        </p>
      )}

      <p className="text-[11px] text-slate-400">
        Gap = Days of Supply − Lead Time. Negative gap means stockout before restock.
      </p>
    </div>
  );
}
