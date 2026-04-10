import type { SellThroughResponse } from '../../types/inventory';

// ── Helpers ──────────────────────────────────────────────────────────────────

function getHealthClass(rate: number): { bar: string; text: string; bg: string; border: string } {
  if (rate >= 70) return { bar: 'bg-emerald-500', text: 'text-emerald-600', bg: 'bg-emerald-100', border: 'border-emerald-100' };
  if (rate >= 40) return { bar: 'bg-amber-500',   text: 'text-amber-600',   bg: 'bg-amber-100',   border: 'border-amber-100'   };
  return              { bar: 'bg-red-500',     text: 'text-red-500',     bg: 'bg-red-100',     border: 'border-red-100'     };
}

// ── Component ─────────────────────────────────────────────────────────────────

const MAX_DISPLAY = 12;

interface Props {
  data: SellThroughResponse;
}

export default function SellThroughModule({ data }: Props) {
  if (data.products.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-slate-400">
        No sell-through data available for this store.
      </div>
    );
  }

  // Already sorted descending by sell-through rate from the API
  const products = data.products.slice(0, MAX_DISPLAY);
  const topPerformer    = products[0];
  const bottomPerformer = products[products.length - 1];

  return (
    <div className="space-y-4">

      {/* ── Callout cards ──────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        {topPerformer && (
          <div className="rounded-lg border border-emerald-100 bg-emerald-100 px-3.5 py-3">
            <p className="text-[9px] font-semibold uppercase tracking-widest text-emerald-600">
              Top Performer
            </p>
            <p className="mt-1 truncate text-[13px] font-semibold text-slate-800">
              {topPerformer.product_id}
            </p>
            <p className="text-[11px] font-medium text-emerald-600">
              {Math.round(topPerformer.sell_through_rate)}% sell-through
            </p>
          </div>
        )}
        {bottomPerformer && bottomPerformer.product_id !== topPerformer?.product_id && (
          <div className="rounded-lg border border-red-100 bg-red-100 px-3.5 py-3">
            <p className="text-[9px] font-semibold uppercase tracking-widest text-red-500">
              Needs Attention
            </p>
            <p className="mt-1 truncate text-[13px] font-semibold text-slate-800">
              {bottomPerformer.product_id}
            </p>
            <p className="text-[11px] font-medium text-red-500">
              {Math.round(bottomPerformer.sell_through_rate)}% sell-through
            </p>
          </div>
        )}
      </div>

      {/* ── Column headers ─────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 px-1">
        <span className="w-5 shrink-0" />
        <span className="w-28 shrink-0 text-[9px] font-semibold uppercase tracking-widest text-slate-400">
          Product
        </span>
        <span className="flex-1 text-[9px] font-semibold uppercase tracking-widest text-slate-400">
          Sell-through rate
        </span>
        <span className="w-9 shrink-0 text-right text-[9px] font-semibold uppercase tracking-widest text-slate-400">
          Rate
        </span>
      </div>

      {/* ── Ranked list ────────────────────────────────────────────────────── */}
      <div className="space-y-1.5">
        {products.map((p, i) => {
          const rate  = Math.round(p.sell_through_rate);
          const theme = getHealthClass(rate);

          return (
            <div key={p.product_id} className="group flex items-center gap-3 rounded-lg px-1 py-1 hover:bg-slate-50 transition-colors">
              {/* Rank */}
              <span className="w-5 shrink-0 text-right text-[10px] font-semibold tabular-nums text-slate-300">
                {i + 1}
              </span>

              {/* Product + category */}
              <div className="w-28 shrink-0">
                <p className="truncate text-[12px] font-medium text-slate-700 leading-tight">
                  {p.product_id}
                </p>
                <p className="truncate text-[10px] text-slate-400 leading-tight">{p.category}</p>
              </div>

              {/* Progress track */}
              <div className="flex flex-1 items-center gap-2">
                <div className="flex-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${theme.bar} transition-all duration-500`}
                    style={{ width: `${rate}%` }}
                  />
                </div>
              </div>

              {/* Rate label */}
              <span className={`w-9 shrink-0 text-right text-[11px] font-semibold tabular-nums ${theme.text}`}>
                {rate}%
              </span>
            </div>
          );
        })}
      </div>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <p className="text-[11px] text-slate-400">
        {products.length} of {data.products.length} products · sorted by sell-through rate (high → low)
      </p>
    </div>
  );
}
