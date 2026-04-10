import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  LabelList,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { SellThroughResponse } from '../../types/inventory';

// ── Helpers ──────────────────────────────────────────────────────────────────

function getRateColor(rate: number): string {
  if (rate >= 70) return '#10b981';
  if (rate >= 40) return '#f59e0b';
  return '#ef4444';
}

interface ChartItem {
  name: string;
  rate: number;
  category: string;
  sold: number;
  received: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function SellThroughTooltip({ active, payload }: { active?: boolean; payload?: any[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as ChartItem;
  return (
    <div
      style={{
        background: 'white',
        border: '1px solid #e2e8f0',
        borderRadius: 8,
        padding: '8px 12px',
        fontSize: 12,
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
      }}
    >
      <p style={{ fontWeight: 600, color: '#1e293b', marginBottom: 2 }}>{d.name}</p>
      <p style={{ color: '#64748b', marginBottom: 6 }}>{d.category}</p>
      <p style={{ color: '#475569' }}>
        Sell-through:{' '}
        <strong style={{ color: getRateColor(d.rate) }}>{d.rate}%</strong>
      </p>
      <p style={{ color: '#94a3b8', fontSize: 11, marginTop: 2 }}>
        {d.sold.toLocaleString()} sold / {d.received.toLocaleString()} received
      </p>
    </div>
  );
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

  const products = data.products.slice(0, MAX_DISPLAY);
  const topPerformer = products[0];
  const bottomPerformer = products[products.length - 1];

  const chartData: ChartItem[] = products.map(p => ({
    name: p.product_id,
    rate: Math.round(p.sell_through_rate),
    category: p.category,
    sold: p.total_sold,
    received: p.total_received,
  }));

  const chartHeight = Math.max(180, chartData.length * 30 + 24);

  return (
    <div className="space-y-4">
      {/* Top / bottom performer callouts */}
      <div className="grid grid-cols-2 gap-3">
        {topPerformer && (
          <div className="rounded-lg border border-emerald-100 bg-emerald-50 px-4 py-3">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-emerald-600">
              Top Performer
            </p>
            <p className="mt-1 truncate text-sm font-semibold text-slate-800">
              {topPerformer.product_id}
            </p>
            <p className="text-xs text-emerald-600">
              {Math.round(topPerformer.sell_through_rate)}% sell-through
            </p>
          </div>
        )}
        {bottomPerformer && bottomPerformer.product_id !== topPerformer?.product_id && (
          <div className="rounded-lg border border-red-100 bg-red-50 px-4 py-3">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-red-500">
              Needs Attention
            </p>
            <p className="mt-1 truncate text-sm font-semibold text-slate-800">
              {bottomPerformer.product_id}
            </p>
            <p className="text-xs text-red-500">
              {Math.round(bottomPerformer.sell_through_rate)}% sell-through
            </p>
          </div>
        )}
      </div>

      {/* Horizontal bar chart */}
      <div style={{ height: chartHeight }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 0, right: 44, left: 4, bottom: 0 }}
            barSize={13}
          >
            <XAxis
              type="number"
              domain={[0, 100]}
              tickFormatter={v => `${v}%`}
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={58}
              tick={{ fontSize: 10, fill: '#64748b' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<SellThroughTooltip />} cursor={{ fill: 'rgba(241,245,249,0.7)' }} />
            <Bar dataKey="rate" radius={[0, 3, 3, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={getRateColor(entry.rate)} fillOpacity={0.88} />
              ))}
              <LabelList
                dataKey="rate"
                position="right"
                style={{ fontSize: 10, fill: '#64748b' }}
                formatter={(v: number) => `${v}%`}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-[11px] text-slate-400">
        Showing {products.length} of {data.products.length} products · Sorted by sell-through rate
        (high → low)
      </p>
    </div>
  );
}
