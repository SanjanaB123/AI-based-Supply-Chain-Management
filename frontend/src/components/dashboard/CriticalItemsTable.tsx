import type { DaysOfSupplyResponse, LeadTimeRiskResponse } from '../../types/inventory';

// ── Helpers ───────────────────────────────────────────────────────────────────

const STATUS_CHIP: Record<'critical' | 'low', { bg: string; text: string }> = {
  critical: { bg: 'bg-red-200',   text: 'text-red-700'   },
  low:      { bg: 'bg-amber-200', text: 'text-amber-700' },
};

const DAYS_TEXT: Record<'critical' | 'low', string> = {
  critical: 'text-red-600',
  low:      'text-amber-600',
};

const ROW_BG: Record<'critical' | 'low', string> = {
  critical: 'bg-red-100',
  low:      'bg-amber-100',
};

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  dosData: DaysOfSupplyResponse;
  ltrData: LeadTimeRiskResponse;
}

export default function CriticalItemsTable({ dosData, ltrData }: Props) {
  // Build lead-time lookup by product_id
  const ltMap = new Map(ltrData.products.map(p => [p.product_id, p.lead_time_days]));

  const urgentProducts = dosData.products
    .filter(p => p.stock_health === 'critical' || p.stock_health === 'low')
    .sort((a, b) => a.days_of_supply - b.days_of_supply)
    .slice(0, 12);

  if (urgentProducts.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-10 text-center h-86">
        <span className="h-2 w-2 rounded-full bg-emerald-500" />
        <p className="text-sm text-slate-400">All products adequately stocked.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3 h-86">
      {/* Summary strip */}
      <div className="flex items-center gap-4 text-sm">
        {(() => {
          const crit = urgentProducts.filter(p => p.stock_health === 'critical').length;
          const low  = urgentProducts.filter(p => p.stock_health === 'low').length;
          return (
            <>
              {crit > 0 && (
                <span className="flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
                  <span className="font-semibold text-red-700">{crit}</span>
                  <span className="text-slate-500">critical</span>
                </span>
              )}
              {low > 0 && (
                <span className="flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
                  <span className="font-semibold text-amber-700">{low}</span>
                  <span className="text-slate-500">low stock</span>
                </span>
              )}
              <span className="ml-auto text-[11px] text-slate-400 shrink-0">
                Sorted by days remaining
              </span>
            </>
          );
        })()}
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
                <th className="px-3 py-2.5 text-center text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Status
                </th>
                <th className="px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Days Left
                </th>
                <th className="hidden px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-400 lg:table-cell">
                  Lead Time
                </th>
                <th className="hidden px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-400 lg:table-cell">
                  Exposure
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {urgentProducts.map(p => {
                const health = p.stock_health as 'critical' | 'low';
                const ltDays = ltMap.get(p.product_id);
                const atRisk = ltDays !== undefined && p.days_of_supply <= ltDays;
                return (
                  <tr key={p.product_id} className={ROW_BG[health]}>
                    <td className="px-3 py-2.5 font-medium text-slate-800">{p.product_id}</td>
                    <td className="hidden px-3 py-2.5 text-slate-500 sm:table-cell">{p.category}</td>
                    <td className="px-3 py-2.5 text-center">
                      <span
                        className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${STATUS_CHIP[health].bg} ${STATUS_CHIP[health].text}`}
                      >
                        {health}
                      </span>
                    </td>
                    <td className={`px-3 py-2.5 text-right tabular-nums font-semibold ${DAYS_TEXT[health]}`}>
                      {Math.round(p.days_of_supply)}d
                    </td>
                    <td className="hidden px-3 py-2.5 text-right tabular-nums text-slate-500 lg:table-cell">
                      {ltDays !== undefined ? `${ltDays}d` : '—'}
                    </td>
                    <td className="hidden px-3 py-2.5 text-right lg:table-cell">
                      {atRisk ? (
                        <span className="text-[10px] font-semibold text-red-500">Stockout risk</span>
                      ) : (
                        <span className="text-[10px] text-slate-300">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-[11px] text-slate-400">
        Showing {urgentProducts.length} urgent item{urgentProducts.length !== 1 ? 's' : ''} ·
        Exposure = stockout before replenishment arrives
      </p>
    </div>
  );
}
