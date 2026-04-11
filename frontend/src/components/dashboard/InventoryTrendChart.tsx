import { useState, useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { DaysOfSupplyResponse } from '../../types/inventory';
import { useTheme } from '../../app/theme/useTheme';

// ── Types ──────────────────────────────────────────────────────────────────────

type Timeframe = '7d' | '30d' | '90d' | '180d' | '365d';

interface TimeframeOption {
  id: Timeframe;
  label: string;
  days: number;
  points: number;
}

const TIMEFRAME_OPTIONS: TimeframeOption[] = [
  { id: '7d',   label: 'Last week',    days: 7,   points: 7  },
  { id: '30d',  label: 'Last 30 days', days: 30,  points: 10 },
  { id: '90d',  label: 'Last 3 mos',   days: 90,  points: 13 },
  { id: '180d', label: 'Last 6 mos',   days: 180, points: 12 },
  { id: '365d', label: 'Last 12 mos',  days: 365, points: 13 },
];

interface DataPoint {
  label: string;
  critical: number;
  low: number;
  healthy: number;
}

// ── Data derivation ────────────────────────────────────────────────────────────

function dateLabel(daysAgo: number, tf: Timeframe): string {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  if (tf === '7d') {
    return d.toLocaleDateString('en-US', { weekday: 'short' });
  }
  if (tf === '30d' || tf === '90d') {
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }
  return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
}

function buildTrendData(data: DaysOfSupplyResponse, tf: Timeframe): DataPoint[] {
  const opt = TIMEFRAME_OPTIONS.find(o => o.id === tf)!;
  const { days, points } = opt;
  const { critical_below, low_below } = data.thresholds;

  return Array.from({ length: points }, (_, i) => {
    const daysAgo = Math.round((days / (points - 1)) * (points - 1 - i));
    let critical = 0;
    let low      = 0;
    let healthy  = 0;

    for (const p of data.products) {
      const estimatedDos = p.days_of_supply + daysAgo;
      if      (estimatedDos < critical_below) critical++;
      else if (estimatedDos < low_below)       low++;
      else                                     healthy++;
    }

    return { label: dateLabel(daysAgo, tf), critical, low, healthy };
  });
}

// ── Custom tooltip ─────────────────────────────────────────────────────────────

interface TooltipEntry {
  name: string;
  value: number;
  color: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string;
  isDark: boolean;
}

function CustomTooltip({ active, payload, label, isDark }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const total = payload.reduce((s, p) => s + (p.value ?? 0), 0);

  return (
    <div className={`rounded-xl border px-3.5 py-3 shadow-md min-w-35 ${
      isDark
        ? 'border-slate-700 bg-slate-800'
        : 'border-slate-200 bg-white'
    }`}>
      <p className={`mb-2 text-[11px] font-semibold uppercase tracking-wide ${
        isDark ? 'text-slate-400' : 'text-slate-500'
      }`}>
        {label}
      </p>
      {[...payload].reverse().map(entry => (
        <div key={entry.name} className="flex items-center gap-2 py-0.5">
          <span
            className="h-2 w-2 shrink-0 rounded-full"
            style={{ background: entry.color }}
          />
          <span className={`text-[12px] capitalize w-14 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
            {entry.name}
          </span>
          <span className={`ml-auto text-[12px] font-semibold tabular-nums ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
            {entry.value}
          </span>
        </div>
      ))}
      <div className={`mt-2 flex items-center justify-between border-t pt-2 ${isDark ? 'border-slate-700' : 'border-slate-100'}`}>
        <span className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Total</span>
        <span className={`text-[12px] font-semibold tabular-nums ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{total}</span>
      </div>
    </div>
  );
}

// ── Component ──────────────────────────────────────────────────────────────────

interface Props {
  data: DaysOfSupplyResponse;
}

export default function InventoryTrendChart({ data }: Props) {
  const [timeframe, setTimeframe] = useState<Timeframe>('30d');
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const chartData = useMemo(
    () => buildTrendData(data, timeframe),
    [data, timeframe],
  );

  const yMax = data.products.length + Math.max(2, Math.ceil(data.products.length * 0.08));

  // Theme-aware chart colors
  const gridColor    = isDark ? '#1e293b' : '#f1f5f9';
  const axisColor    = isDark ? '#475569' : '#94a3b8';
  const cursorColor  = isDark ? '#475569' : '#cbd5e1';

  return (
    <div>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[13px] font-semibold text-slate-800 dark:text-slate-200 leading-tight">
            Stock Health Over Time
          </p>
          <p className="mt-0.5 text-[11px] text-slate-400 dark:text-slate-500 leading-snug">
            Projected from current daily sales velocity · products by health status
          </p>
        </div>

        {/* Timeframe pill selector */}
        <div className="flex items-center gap-0.5 rounded-lg bg-slate-100 dark:bg-slate-700 p-0.5">
          {TIMEFRAME_OPTIONS.map(opt => (
            <button
              key={opt.id}
              onClick={() => setTimeframe(opt.id)}
              className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                timeframe === opt.id
                  ? 'bg-white dark:bg-slate-600 text-slate-800 dark:text-slate-100 shadow-sm'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Chart ──────────────────────────────────────────────────────────── */}
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 4, right: 4, bottom: 0, left: -18 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={gridColor}
              vertical={false}
            />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10, fill: axisColor }}
              axisLine={false}
              tickLine={false}
              dy={5}
            />
            <YAxis
              domain={[0, yMax]}
              tick={{ fontSize: 10, fill: axisColor }}
              axisLine={false}
              tickLine={false}
              allowDecimals={false}
              width={36}
            />
            <Tooltip
              content={<CustomTooltip isDark={isDark} />}
              cursor={{ stroke: cursorColor, strokeWidth: 1 }}
            />

            <Area
              type="monotone"
              dataKey="critical"
              stackId="stack"
              stroke="#ef4444"
              strokeWidth={1.5}
              fill="#ef4444"
              fillOpacity={0.14}
            />
            <Area
              type="monotone"
              dataKey="low"
              stackId="stack"
              stroke="#f59e0b"
              strokeWidth={1.5}
              fill="#f59e0b"
              fillOpacity={0.16}
            />
            <Area
              type="monotone"
              dataKey="healthy"
              stackId="stack"
              stroke="#10b981"
              strokeWidth={1.5}
              fill="#10b981"
              fillOpacity={0.14}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* ── Legend + meta ──────────────────────────────────────────────────── */}
      <div className="mt-3.5 flex items-center gap-5 border-t border-slate-100 dark:border-slate-700 pt-3">
        {[
          { color: '#10b981', label: 'Healthy' },
          { color: '#f59e0b', label: 'Low stock' },
          { color: '#ef4444', label: 'Critical'  },
        ].map(({ color, label }) => (
          <span key={label} className="flex items-center gap-1.5 text-[11px] text-slate-500 dark:text-slate-400">
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
            {label}
          </span>
        ))}
        <span className="ml-auto text-[10px] text-slate-400 dark:text-slate-500 text-right">
          {data.products.length} products · estimated projection
        </span>
      </div>
    </div>
  );
}
