import type { ShrinkageResponse } from '../../types/inventory';

// ── Helpers ──────────────────────────────────────────────────────────────────

function varianceTextClass(v: number): string {
  if (v > 0) return 'text-amber-700 font-semibold';
  if (v < 0) return 'text-slate-400';
  return 'text-emerald-700 font-semibold';
}

function varianceRowBg(v: number): string {
  if (v > 10) return 'bg-amber-100/50';
  if (v < 0) return 'bg-slate-50/40';
  return '';
}

function formatVariance(v: number): string {
  if (v > 0) return `+${v.toLocaleString()}`;
  return v.toLocaleString();
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data: ShrinkageResponse;
}

export default function ShrinkageModule({ data }: Props) {
  if (data.products.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-slate-400">
        No inventory variance data available for this store.
      </div>
    );
  }

  const { total_shrinkage, products } = data;
  const positiveCount = products.filter(p => p.shrinkage > 0).length;

  return (
    <div className="space-y-4">
      {/* Summary card */}
      <div className="flex items-start justify-between rounded-lg border border-slate-100 bg-slate-50 px-4 py-3.5">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
            Total Inventory Variance
          </p>
          <p
            className={`mt-1 text-2xl font-bold tracking-tight ${
              total_shrinkage > 0 ? 'text-amber-700' : 'text-slate-800'
            }`}
          >
            {formatVariance(total_shrinkage)}{' '}
            <span className="text-base font-medium text-slate-500">units</span>
          </p>
          {positiveCount > 0 && (
            <p className="mt-1 text-xs text-slate-500">
              {positiveCount} product{positiveCount !== 1 ? 's' : ''} with unaccounted units
            </p>
          )}
        </div>
        <div className="shrink-0 text-right text-[11px] leading-5 text-slate-400">
          <p>Received − Sold − On Hand</p>
          <p>Positive = unaccounted units</p>
          <p>Negative = data anomaly</p>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-lg border border-slate-100">
        <div className="max-h-64 overflow-y-auto">
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
                  Received
                </th>
                <th className="px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Sold
                </th>
                <th className="px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  On Hand
                </th>
                <th className="px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Variance
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {products.map(p => (
                <tr key={p.product_id} className={varianceRowBg(p.shrinkage)}>
                  <td className="px-3 py-2.5 font-medium text-slate-800">{p.product_id}</td>
                  <td className="hidden px-3 py-2.5 text-slate-500 sm:table-cell">{p.category}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums text-slate-600">
                    {p.total_received.toLocaleString()}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums text-slate-600">
                    {p.total_sold.toLocaleString()}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums text-slate-600">
                    {p.current_stock.toLocaleString()}
                  </td>
                  <td className={`px-3 py-2.5 text-right tabular-nums ${varianceTextClass(p.shrinkage)}`}>
                    {formatVariance(p.shrinkage)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-[11px] text-slate-400">
        Variance = Received − Sold − On Hand. Positive values indicate units not accounted for in
        current stock records.
      </p>
    </div>
  );
}
