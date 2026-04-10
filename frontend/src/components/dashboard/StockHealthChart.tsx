import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import type { StockHealthResponse, StockStatus } from '../../types/inventory';

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

interface Props {
  data: StockHealthResponse;
}

export default function StockHealthChart({ data }: Props) {
  const chartData = data.breakdown.map(item => ({
    name: item.status,
    value: item.count,
  }));

  return (
    <div className="rounded-xl bg-white shadow-sm p-6">
      <h3 className="mb-6 text-sm font-semibold text-gray-700">
        Stock Health Distribution
      </h3>
      <div className="flex flex-col gap-8 md:flex-row md:items-center md:gap-12">

        {/* Donut chart — DOM overlay for center label avoids SVG text complexity */}
        <div className="relative mx-auto h-52 w-52 shrink-0 md:mx-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={68}
                outerRadius={96}
                paddingAngle={3}
                dataKey="value"
                startAngle={90}
                endAngle={-270}
                strokeWidth={0}
              >
                {chartData.map((entry, i) => (
                  <Cell
                    key={`cell-${i}`}
                    fill={STATUS_COLORS[entry.name as StockStatus]}
                  />
                ))}
              </Pie>
              <Tooltip
                formatter={(value, name) => [
                  `${value ?? 0} products`,
                  STATUS_LABELS[name as StockStatus] ?? String(name),
                ]}
                contentStyle={{
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb',
                  fontSize: '13px',
                  boxShadow: '0 1px 6px rgba(0,0,0,0.07)',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          {/* Center label overlay */}
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-bold text-gray-900">{data.total_products}</span>
            <span className="mt-0.5 text-xs text-gray-400">products</span>
          </div>
        </div>

        {/* Breakdown legend */}
        <div className="flex-1">
          <div className="divide-y divide-gray-100">
            {data.breakdown.map(item => (
              <div
                key={item.status}
                className="flex items-center justify-between py-3.5 first:pt-0 last:pb-0"
              >
                <div className="flex items-center gap-3">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: STATUS_COLORS[item.status] }}
                  />
                  <span className="text-sm font-medium text-gray-700">
                    {STATUS_LABELS[item.status]}
                  </span>
                </div>
                <div className="flex items-baseline gap-3">
                  <span className="text-sm font-bold text-gray-900">{item.count}</span>
                  <span className="w-12 text-right text-xs text-gray-400">
                    {item.percentage}%
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-center justify-between border-t border-gray-100 pt-3.5">
            <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">
              Total
            </span>
            <span className="text-sm font-bold text-gray-900">
              {data.total_products} products
            </span>
          </div>
        </div>

      </div>
    </div>
  );
}
