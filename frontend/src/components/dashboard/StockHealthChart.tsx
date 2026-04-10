import { PieChart, Pie, Tooltip, ResponsiveContainer } from 'recharts';
import type { StockHealthResponse, StockStatus } from '../../types/inventory';

// ── Constants ────────────────────────────────────────────────────────────────

const STATUS_COLORS: Record<StockStatus, string> = {
  critical: '#ef4444',
  low: '#f59e0b',
  healthy: '#10b981',
};

const STATUS_LABELS: Record<StockStatus, string> = {
  critical: 'Critical',
  low: 'Low',
  healthy: 'Healthy',
};

const STATUS_SUBTEXTS: Record<StockStatus, string> = {
  critical: 'Need immediate reorder',
  low: 'Approaching reorder point',
  healthy: 'Adequate supply',
};

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data: StockHealthResponse;
}

export default function StockHealthChart({ data }: Props) {
  // Embed fill into data objects — recharts v3 reads `fill` per-entry without Cell
  const chartData = data.breakdown.map(item => ({
    name: item.status,
    value: item.count,
    fill: STATUS_COLORS[item.status],
  }));

  const healthyItem = data.breakdown.find(b => b.status === 'healthy');
  const healthPct = healthyItem?.percentage ?? 0;

  return (
    <div className="rounded-xl border border-slate-100 bg-white shadow-sm">
      <div className="flex flex-col gap-6 p-5">

        {/* Donut with DOM center label */}
        <div className="relative mx-auto h-44 w-44 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={58}
                outerRadius={84}
                paddingAngle={3}
                dataKey="value"
                startAngle={90}
                endAngle={-270}
                strokeWidth={0}
              />
              <Tooltip
                formatter={(value, name) => [
                  `${value ?? 0} products`,
                  STATUS_LABELS[name as StockStatus] ?? String(name),
                ]}
                contentStyle={{
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0',
                  fontSize: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                  padding: '6px 10px',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          {/* Center overlay */}
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-3xl font-bold tracking-tight text-slate-900">
              {data.total_products}
            </span>
            <span className="text-[11px] font-medium uppercase tracking-widest text-slate-400">
              products
            </span>
          </div>
        </div>

        {/* Legend */}
        <div className="min-w-0">
          <div className="divide-y divide-slate-50">
            {data.breakdown.map(item => (
              <div
                key={item.status}
                className="flex items-center justify-between py-3 first:pt-0 last:pb-0"
              >
                <div className="flex items-start gap-3">
                  <span
                    className="mt-1 h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: STATUS_COLORS[item.status] }}
                  />
                  <div>
                    <p className="text-sm font-medium text-slate-700">
                      {STATUS_LABELS[item.status]}
                    </p>
                    <p className="text-xs text-slate-400">{STATUS_SUBTEXTS[item.status]}</p>
                  </div>
                </div>
                <div className="flex items-baseline gap-3 shrink-0 pl-4">
                  <span className="text-sm font-semibold text-slate-900">{item.count}</span>
                  <span className="w-9 text-right text-xs text-slate-400">{item.percentage}%</span>
                </div>
              </div>
            ))}
          </div>

          {/* Total row */}
          <div className="mt-2.5 flex items-center justify-between border-t border-slate-100 pt-3">
            <span className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">
              Total
            </span>
            <div className="flex items-baseline gap-3">
              <span className="text-sm font-semibold text-slate-900">{data.total_products}</span>
              <span className="w-9 text-right text-xs text-slate-400">100%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Health score footer */}
      <div className="flex items-center justify-between border-t border-slate-50 px-5 py-3.5">
        <span className="text-xs text-slate-400">Store health score</span>
        <div className="flex items-center gap-2.5">
          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-emerald-500 transition-all duration-500"
              style={{ width: `${healthPct}%` }}
            />
          </div>
          <span className="text-xs font-semibold text-slate-700">{healthPct}%</span>
        </div>
      </div>
    </div>
  );
}
