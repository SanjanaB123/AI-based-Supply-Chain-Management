import type { DaysOfSupplyResponse } from '../../types/inventory';

// ── Helpers ──────────────────────────────────────────────────────────────────

type Health = 'critical' | 'low' | 'healthy';

const HEALTH_STYLES: Record<Health, { rowBg: string; dayText: string; barColor: string }> = {
  critical: { rowBg: 'bg-red-100/60',   dayText: 'text-red-700',    barColor: '#ef4444' },
  low:      { rowBg: 'bg-amber-100/60', dayText: 'text-amber-700',  barColor: '#f59e0b' },
  healthy:  { rowBg: '',               dayText: 'text-emerald-700', barColor: '#10b981' },
};

function isHealth(s: string): s is Health {
  return s === 'critical' || s === 'low' || s === 'healthy';
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data: DaysOfSupplyResponse;
}

export default function DaysOfSupplyModule({ data }: Props) {
  if (data.products.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-slate-400">
        No days-of-supply data available for this store.
      </div>
    );
  }

  const { thresholds, products } = data;
  const criticalCount = products.filter(p => p.stock_health === 'critical').length;
  const lowCount = products.filter(p => p.stock_health === 'low').length;

  // For progress bars: scale against the max days value (capped at 90 for readability)
  const maxDays = Math.min(90, Math.max(...products.map(p => p.days_of_supply)));

  return (
    <div className="space-y-4">
      {/* Summary row */}
      <div className="flex items-center gap-5 text-sm">
        {criticalCount > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 shrink-0 rounded-full bg-red-500" />
            <span className="font-semibold text-red-700">{criticalCount}</span>
            <span className="text-slate-500">critical</span>
          </span>
        )}
        {lowCount > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 shrink-0 rounded-full bg-amber-500" />
            <span className="font-semibold text-amber-700">{lowCount}</span>
            <span className="text-slate-500">low</span>
          </span>
        )}
        {criticalCount === 0 && lowCount === 0 && (
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 shrink-0 rounded-full bg-emerald-500" />
            <span className="text-slate-500">All products adequately stocked</span>
          </span>
        )}
        <span className="ml-auto text-[11px] text-slate-400 shrink-0">
          Critical &lt;{thresholds.critical_below}d · Low &lt;{thresholds.low_below}d
        </span>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-lg border border-slate-100">
        <div className="max-h-72 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10 bg-white">
              <tr className="border-b border-slate-100">
                <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Product
                </th>
                <th className="hidden px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-widest text-slate-400 sm:table-cell">
                  Category
                </th>
                <th className="px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Days Left
                </th>
                <th className="px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  On Hand
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {products.map(p => {
                const health = isHealth(p.stock_health) ? p.stock_health : 'healthy';
                const styles = HEALTH_STYLES[health];
                const barPct = Math.min(100, (Math.min(p.days_of_supply, maxDays) / maxDays) * 100);
                return (
                  <tr key={p.product_id} className={styles.rowBg}>
                    <td className="px-3 py-2.5 font-medium text-slate-800">
                      {p.product_id}
                    </td>
                    <td className="hidden px-3 py-2.5 text-slate-500 sm:table-cell">
                      {p.category}
                    </td>
                    <td className="px-3 py-2.5">
                      <div className="flex items-center justify-end gap-2">
                        <div className="h-1.5 w-14 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{ width: `${barPct}%`, backgroundColor: styles.barColor }}
                          />
                        </div>
                        <span
                          className={`w-9 text-right text-sm font-semibold tabular-nums ${styles.dayText}`}
                        >
                          {Math.round(p.days_of_supply)}d
                        </span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-slate-600">
                      {p.current_stock.toLocaleString()}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-[11px] text-slate-400">
        Sorted by days remaining (ascending). Progress bar scaled to{' '}
        {maxDays >= 90 ? '90+' : maxDays} days.
      </p>
    </div>
  );
}
