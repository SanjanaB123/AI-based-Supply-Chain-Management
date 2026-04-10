import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { LeadTimeRiskResponse, LeadTimeRiskProduct } from '../../types/inventory';

// ── Thresholds (match backend constants) ─────────────────────────────────────

const CRITICAL_BELOW = 14;
const LOW_BELOW = 45;

// ── Tooltip ───────────────────────────────────────────────────────────────────

interface TooltipProps {
  active?: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  payload?: Array<{ payload: any }>;
}

function ScatterTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload as LeadTimeRiskProduct;
  const atRisk = p.days_of_supply <= p.lead_time_days;
  return (
    <div
      style={{
        background: 'white',
        border: '1px solid #e2e8f0',
        borderRadius: 8,
        padding: '8px 12px',
        fontSize: 12,
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        minWidth: 140,
      }}
    >
      <p style={{ fontWeight: 600, color: '#1e293b', marginBottom: 2 }}>{p.product_id}</p>
      <p style={{ color: '#64748b', marginBottom: 6 }}>{p.category}</p>
      <p style={{ color: '#475569' }}>
        Lead time: <strong>{p.lead_time_days}d</strong>
      </p>
      <p style={{ color: '#475569' }}>
        Days of supply: <strong>{Math.round(p.days_of_supply)}d</strong>
      </p>
      {atRisk && (
        <p style={{ color: '#ef4444', fontWeight: 600, marginTop: 6, fontSize: 11 }}>
          ⚠ Stockout before restock
        </p>
      )}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data: LeadTimeRiskResponse;
}

export default function LeadTimeRiskModule({ data }: Props) {
  if (data.products.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-slate-400">
        No lead-time data available for this store.
      </div>
    );
  }

  const critical = data.products.filter(p => p.stock_health === 'critical');
  const low = data.products.filter(p => p.stock_health === 'low');
  const healthy = data.products.filter(p => p.stock_health === 'healthy');

  // Items where supply will run out before restock arrives
  const atRisk = data.products.filter(p => p.days_of_supply <= p.lead_time_days);

  return (
    <div className="space-y-4">
      {/* At-risk callout */}
      {atRisk.length > 0 ? (
        <div className="flex items-start gap-3 rounded-lg border border-red-100 bg-red-50 px-4 py-3 text-sm text-red-700">
          <svg
            className="mt-0.5 h-4 w-4 shrink-0"
            viewBox="0 0 16 16"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm-.75 3.75a.75.75 0 0 1 1.5 0v3.5a.75.75 0 0 1-1.5 0v-3.5zm.75 7a.875.875 0 1 1 0-1.75.875.875 0 0 1 0 1.75z" />
          </svg>
          <span>
            <strong>{atRisk.length}</strong> item{atRisk.length !== 1 ? 's' : ''} will stock
            out before their replenishment order arrives.
          </span>
        </div>
      ) : (
        <div className="flex items-center gap-3 rounded-lg border border-emerald-100 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
          <span className="h-2 w-2 shrink-0 rounded-full bg-emerald-500" />
          All products have sufficient supply to cover their lead times.
        </div>
      )}

      {/* Scatter chart */}
      <div style={{ height: 268 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 24, bottom: 28, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              type="number"
              dataKey="lead_time_days"
              name="Lead Time"
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
              label={{
                value: 'Lead Time (days)',
                position: 'insideBottom',
                offset: -14,
                fontSize: 10,
                fill: '#94a3b8',
              }}
            />
            <YAxis
              type="number"
              dataKey="days_of_supply"
              name="Days of Supply"
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
              label={{
                value: 'Days of Supply',
                angle: -90,
                position: 'insideLeft',
                offset: 14,
                fontSize: 10,
                fill: '#94a3b8',
              }}
            />
            {/* Threshold reference lines */}
            <ReferenceLine
              y={CRITICAL_BELOW}
              stroke="#ef4444"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{ value: `Critical (${CRITICAL_BELOW}d)`, position: 'insideTopRight', fill: '#ef4444', fontSize: 10 }}
            />
            <ReferenceLine
              y={LOW_BELOW}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{ value: `Low (${LOW_BELOW}d)`, position: 'insideTopRight', fill: '#f59e0b', fontSize: 10 }}
            />
            <Tooltip content={<ScatterTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11, color: '#64748b', paddingTop: 4 }}
            />
            <Scatter name="Critical" data={critical} fill="#ef4444" fillOpacity={0.8} r={5} />
            <Scatter name="Low" data={low} fill="#f59e0b" fillOpacity={0.8} r={5} />
            <Scatter name="Healthy" data={healthy} fill="#10b981" fillOpacity={0.7} r={5} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <p className="text-[11px] text-slate-400">
        Each point is a product. Items where Days of Supply ≤ Lead Time are at risk of stockout.
      </p>
    </div>
  );
}
