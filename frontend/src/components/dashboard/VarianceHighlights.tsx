import type { ShrinkageResponse } from '../../types/inventory';

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatVariance(v: number): string {
  return v > 0 ? `+${v.toLocaleString()}` : v.toLocaleString();
}

function varianceColor(v: number): string {
  if (v > 0) return 'text-amber-600';
  if (v < 0) return 'text-slate-400';
  return 'text-emerald-600';
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data: ShrinkageResponse;
}

export default function VarianceHighlights({ data }: Props) {
  // Sort by absolute magnitude to show the most extreme values
  const sorted = [...data.products]
    .sort((a, b) => Math.abs(b.shrinkage) - Math.abs(a.shrinkage))
    .slice(0, 8);

  if (sorted.length === 0) {
    return (
      <div className="flex items-center justify-center py-8 text-sm text-slate-400">
        No variance data available.
      </div>
    );
  }

  const totalPos = data.products.filter(p => p.shrinkage > 0).length;
  const totalNeg = data.products.filter(p => p.shrinkage < 0).length;

  return (
    <div className="space-y-3 h-92 overflow-x-auto">
      {/* Summary strip */}
      <div className="flex items-center gap-4 text-[11px] text-slate-400">
        {totalPos > 0 && (
          <span>
            <span className="font-semibold text-amber-600">{totalPos}</span> with surplus
          </span>
        )}
        {totalNeg > 0 && (
          <span>
            <span className="font-semibold text-slate-500">{totalNeg}</span> anomaly
          </span>
        )}
        <span className="ml-auto">by magnitude</span>
      </div>

      {/* Item list */}
      <div className="space-y-1.5">
        {sorted.map(p => (
          <div
            key={p.product_id}
            className="flex items-center justify-between rounded-lg border border-slate-100 bg-white px-3 py-2.5"
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-slate-800">{p.product_id}</p>
              <p className="text-[11px] text-slate-400">{p.category}</p>
            </div>
            <span className={`ml-3 shrink-0 text-sm font-semibold tabular-nums ${varianceColor(p.shrinkage)}`}>
              {formatVariance(p.shrinkage)}
            </span>
          </div>
        ))}
      </div>

      <p className="text-[11px] text-slate-400">
        Variance = Received − Sold − On Hand. Positive = unaccounted units.
      </p>
    </div>
  );
}
