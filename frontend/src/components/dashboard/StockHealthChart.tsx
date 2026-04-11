import { PieChart, Pie, Tooltip, ResponsiveContainer } from 'recharts';
import type { StockHealthResponse, StockStatus } from '../../types/inventory';
import { useTheme } from '../../app/theme/useTheme';

// ── Constants ────────────────────────────────────────────────────────────────

const STATUS_COLORS: Record<StockStatus, string> = {
  critical: '#ef4444',
  low:      '#f59e0b',
  healthy:  '#10b981',
};

const STATUS_LABELS: Record<StockStatus, string> = {
  critical: 'Critical',
  low:      'Low',
  healthy:  'Healthy',
};

const STATUS_SUBTEXTS: Record<StockStatus, string> = {
  critical: 'Need immediate reorder',
  low:      'Approaching reorder point',
  healthy:  'Adequate supply',
};

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data: StockHealthResponse;
}

export default function StockHealthChart({ data }: Props) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const chartData = data.breakdown.map(item => ({
    name:  item.status,
    value: item.count,
    fill:  STATUS_COLORS[item.status],
  }));

  const healthyItem = data.breakdown.find(b => b.status === 'healthy');
  const healthPct   = healthyItem?.percentage ?? 0;

  return (
    <div className="rounded-xl border border-slate-200/80 dark:border-slate-700/80 bg-white dark:bg-slate-800 shadow-sm max-h-130">
      <div className="flex flex-col gap-5 p-5">

        {/* Donut with center label */}
        <div className="relative mx-auto h-56 w-56 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={76}
                outerRadius={112}
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
                  border: isDark ? '1px solid #334155' : '1px solid #e2e8f0',
                  fontSize: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
                  padding: '6px 10px',
                  backgroundColor: isDark ? '#1e293b' : '#ffffff',
                  color: isDark ? '#e2e8f0' : '#1e293b',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          {/* Center overlay */}
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100 xl:text-4xl tabular-nums">
              {data.total_products}
            </span>
            <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400 dark:text-slate-500">
              products
            </span>
          </div>
        </div>

        {/* Legend */}
        <div className="min-w-0">
          <div className="divide-y divide-slate-100 dark:divide-slate-700">
            {data.breakdown.map(item => (
              <div
                key={item.status}
                className="flex items-center justify-between py-2.5 first:pt-0 last:pb-0"
              >
                <div className="flex items-start gap-3">
                  <span
                    className="mt-1 h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: STATUS_COLORS[item.status] }}
                  />
                  <div>
                    <p className="text-[13px] font-medium text-slate-700 dark:text-slate-200">
                      {STATUS_LABELS[item.status]}
                    </p>
                    <p className="text-[11px] text-slate-400 dark:text-slate-500">{STATUS_SUBTEXTS[item.status]}</p>
                  </div>
                </div>
                <div className="flex items-baseline gap-2.5 shrink-0 pl-4">
                  <span className="text-sm font-semibold text-slate-900 dark:text-slate-100 tabular-nums">{item.count}</span>
                  <span className="w-8 text-right text-[11px] text-slate-400 dark:text-slate-500 tabular-nums">{item.percentage}%</span>
                </div>
              </div>
            ))}
          </div>

          {/* Total row */}
          <div className="mt-2 flex items-center justify-between border-t border-slate-100 dark:border-slate-700 pt-2.5">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400 dark:text-slate-500">
              Total
            </span>
            <div className="flex items-baseline gap-2.5">
              <span className="text-sm font-semibold text-slate-900 dark:text-slate-100 tabular-nums">{data.total_products}</span>
              <span className="w-8 text-right text-[11px] text-slate-400 dark:text-slate-500">100%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Health score footer */}
      <div className="flex items-center justify-between border-t border-slate-100 dark:border-slate-700 px-5 py-3">
        <span className="text-[11px] text-slate-400 dark:text-slate-500">Store health score</span>
        <div className="flex items-center gap-2.5">
          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-700">
            <div
              className="h-full rounded-full bg-emerald-500 transition-all duration-500"
              style={{ width: `${healthPct}%` }}
            />
          </div>
          <span className="text-[12px] font-semibold text-slate-700 dark:text-slate-300 tabular-nums">{healthPct}%</span>
        </div>
      </div>
    </div>
  );
}
