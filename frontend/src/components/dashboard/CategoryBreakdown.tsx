import type { StockLevelsResponse } from '../../types/inventory';

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data: StockLevelsResponse;
}

interface CategoryRow {
  name: string;
  critical: number;
  low: number;
  healthy: number;
  total: number;
}

export default function CategoryBreakdown({ data }: Props) {
  // Aggregate products by category
  const catMap = new Map<string, CategoryRow>();

  for (const p of data.products) {
    const row = catMap.get(p.category) ?? {
      name: p.category,
      critical: 0,
      low: 0,
      healthy: 0,
      total: 0,
    };
    if (p.stock_health === 'critical') row.critical++;
    else if (p.stock_health === 'low') row.low++;
    else row.healthy++;
    row.total++;
    catMap.set(p.category, row);
  }

  const rows = [...catMap.values()].sort((a, b) => b.total - a.total);

  if (rows.length === 0) {
    return (
      <div className="flex items-center justify-center py-8 text-sm text-slate-400">
        No category data available.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Color legend */}
      <div className="flex items-center gap-4 text-[10px] text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-red-500" />
          Critical
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-amber-500" />
          Low
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
          Healthy
        </span>
      </div>

      {/* Category rows */}
      <div className="space-y-3.5">
        {rows.map(row => {
          const critPct    = (row.critical / row.total) * 100;
          const lowPct     = (row.low      / row.total) * 100;
          const healthyPct = (row.healthy  / row.total) * 100;
          return (
            <div key={row.name}>
              <div className="mb-1.5 flex items-center justify-between">
                <p className="text-sm font-medium text-slate-700 truncate">{row.name}</p>
                <span className="ml-2 shrink-0 text-xs text-slate-400">{row.total}</span>
              </div>
              {/* Stacked health bar */}
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                <div className="flex h-full">
                  <div
                    style={{ width: `${critPct}%` }}
                    className="bg-red-500 transition-all duration-500"
                  />
                  <div
                    style={{ width: `${lowPct}%` }}
                    className="bg-amber-500 transition-all duration-500"
                  />
                  <div
                    style={{ width: `${healthyPct}%` }}
                    className="bg-emerald-500 transition-all duration-500"
                  />
                </div>
              </div>
              <div className="mt-1 flex items-center gap-3 text-[10px]">
                {row.critical > 0 && (
                  <span className="font-medium text-red-600">{row.critical} critical</span>
                )}
                {row.low > 0 && (
                  <span className="font-medium text-amber-600">{row.low} low</span>
                )}
                <span className="font-medium text-emerald-600">{row.healthy} healthy</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
