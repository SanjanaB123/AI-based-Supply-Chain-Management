import { useState } from 'react';
import type { DaysOfSupplyResponse } from '../../types/inventory';

// ── Props ──────────────────────────────────────────────────────────────────────

interface Props {
  data: DaysOfSupplyResponse;
}

// ── Cell color helpers ─────────────────────────────────────────────────────────

interface CellStyle {
  bg: string;
  label: string;
}

function getCellStyle(days: number): CellStyle {
  if (days <= 7)  return { bg: 'bg-red-600',     label: '≤7d'  };
  if (days <= 14) return { bg: 'bg-red-400',      label: '≤14d' };
  if (days <= 30) return { bg: 'bg-amber-400',    label: '≤30d' };
  if (days <= 60) return { bg: 'bg-emerald-300',  label: '≤60d' };
  return                 { bg: 'bg-emerald-500',  label: '60d+' };
}

// ── Legend items ───────────────────────────────────────────────────────────────

const LEGEND_ITEMS: Array<{ bg: string; label: string }> = [
  { bg: 'bg-red-600',     label: '≤7d'  },
  { bg: 'bg-red-400',     label: '≤14d' },
  { bg: 'bg-amber-400',   label: '≤30d' },
  { bg: 'bg-emerald-300', label: '≤60d' },
  { bg: 'bg-emerald-500', label: '60d+' },
];

// ── Component ──────────────────────────────────────────────────────────────────

export default function InventoryHeatmapModule({ data }: Props) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const byCategory = new Map<string, typeof data.products>();
  for (const p of data.products) {
    const arr = byCategory.get(p.category) ?? [];
    arr.push(p);
    byCategory.set(p.category, arr);
  }

  const rows = [...byCategory.entries()].sort(
    (a, b) =>
      b[1].filter(p => p.stock_health === 'critical').length -
      a[1].filter(p => p.stock_health === 'critical').length,
  );

  const hovered       = hoveredId ? data.products.find(p => p.product_id === hoveredId) : null;
  const criticalCount = data.products.filter(p => p.stock_health === 'critical').length;
  const lowCount      = data.products.filter(p => p.stock_health === 'low').length;
  const healthyCount  = data.products.filter(p => p.stock_health === 'healthy').length;

  return (
    <div className="flex flex-col h-86">

      {/* ── Header ───────────────────────────────────────────────────────────── */}
      <div className="mb-2.5 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-slate-700 dark:text-slate-200 leading-tight">
            Inventory Health Density
          </p>
          <p className="mt-0.5 text-[10px] text-slate-400 dark:text-slate-500 leading-snug">
            Each cell = one product · color = days of supply remaining
          </p>
        </div>
        <div className="shrink-0 flex flex-col items-end gap-0.5 text-[10px]">
          <span className="flex items-center gap-1 text-red-500 dark:text-red-400 font-semibold">
            <span className="h-1.5 w-1.5 rounded-full bg-red-500 inline-block" />
            {criticalCount} critical
          </span>
          <span className="flex items-center gap-1 text-amber-500 dark:text-amber-400 font-semibold">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-500 inline-block" />
            {lowCount} low
          </span>
          <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-semibold">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 inline-block" />
            {healthyCount} healthy
          </span>
        </div>
      </div>

      {/* ── Hover info bar ────────────────────────────────────────────────────── */}
      <div
        className={`mb-2 transition-opacity duration-100 ${hovered ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        aria-live="polite"
      >
        <div className="rounded-lg border border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/50 px-3 py-1.5 flex items-center gap-2.5">
          <span
            className={`h-3 w-3 rounded-[3px] shrink-0 ${hovered ? getCellStyle(hovered.days_of_supply).bg : 'bg-slate-200 dark:bg-slate-600'}`}
          />
          <span className="text-[11px] font-semibold text-slate-700 dark:text-slate-200 truncate">
            {hovered?.product_id ?? '–'}
          </span>
          <span className="text-[11px] text-slate-400 dark:text-slate-500">
            {hovered ? `${Math.round(hovered.days_of_supply)}d remaining` : ''}
          </span>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 ml-auto shrink-0">
            {hovered?.category ?? ''}
          </span>
        </div>
      </div>

      {/* ── Category grid ─────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto space-y-1.5 pr-0.5">
        {rows.map(([category, products]) => (
          <div key={category} className="flex items-start gap-2">
            <p
              className="w-19 shrink-0 pt-0.5 text-[10px] leading-tight text-slate-400 dark:text-slate-500 truncate text-right"
              title={category}
            >
              {category}
            </p>
            <div className="flex flex-wrap gap-0.75">
              {[...products]
                .sort((a, b) => a.days_of_supply - b.days_of_supply)
                .map(p => {
                  const { bg } = getCellStyle(p.days_of_supply);
                  const isHov  = hoveredId === p.product_id;
                  return (
                    <div
                      key={p.product_id}
                      className={`h-3.5 w-3.5 rounded-[3px] cursor-default transition-transform duration-100 ${bg} ${
                        isHov ? 'scale-125 ring-1 ring-slate-500 ring-offset-1' : 'hover:scale-110'
                      }`}
                      onMouseEnter={() => setHoveredId(p.product_id)}
                      onMouseLeave={() => setHoveredId(null)}
                      aria-label={`${p.product_id}: ${Math.round(p.days_of_supply)}d`}
                    />
                  );
                })}
            </div>
          </div>
        ))}
      </div>

      {/* ── Legend ────────────────────────────────────────────────────────────── */}
      <div className="mt-3 flex items-center gap-1 flex-wrap border-t border-slate-100 dark:border-slate-700 pt-2.5">
        <span className="mr-1.5 text-[9px] font-medium text-slate-400 dark:text-slate-500 uppercase tracking-wide">
          Supply
        </span>
        {LEGEND_ITEMS.map(({ bg, label }) => (
          <span key={label} className="flex items-center gap-0.5 mr-0.5">
            <span className={`h-2.5 w-2.5 rounded-xs ${bg}`} />
            <span className="text-[9px] text-slate-400 dark:text-slate-500">{label}</span>
          </span>
        ))}
        <span className="ml-auto text-[9px] text-slate-400 dark:text-slate-500">
          {data.products.length} products
        </span>
      </div>
    </div>
  );
}
