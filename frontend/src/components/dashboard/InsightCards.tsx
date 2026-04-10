import type {
  SellThroughResponse,
  DaysOfSupplyResponse,
  ShrinkageResponse,
} from '../../types/inventory';

// ── Types ─────────────────────────────────────────────────────────────────────

type AccentColor = 'emerald' | 'red' | 'amber' | 'blue';

interface Insight {
  title: string;
  value: string;
  sub: string;
  accent: AccentColor;
}

// ── Style map ─────────────────────────────────────────────────────────────────

const ACCENT: Record<AccentColor, { bg: string; border: string; label: string; dot: string }> = {
  emerald: { bg: 'bg-emerald-50', border: 'border-emerald-100', label: 'text-emerald-600', dot: 'bg-emerald-500' },
  red:     { bg: 'bg-red-50',     border: 'border-red-100',     label: 'text-red-600',     dot: 'bg-red-500'     },
  amber:   { bg: 'bg-amber-50',   border: 'border-amber-100',   label: 'text-amber-600',   dot: 'bg-amber-500'   },
  blue:    { bg: 'bg-blue-50',    border: 'border-blue-100',    label: 'text-blue-600',    dot: 'bg-blue-500'    },
};

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  sellThrough: SellThroughResponse;
  daysOfSupply: DaysOfSupplyResponse;
  shrinkage: ShrinkageResponse;
}

export default function InsightCards({ sellThrough, daysOfSupply, shrinkage }: Props) {
  const insights: Insight[] = [];

  // Top performer by sell-through rate
  const stSorted = [...sellThrough.products].sort(
    (a, b) => b.sell_through_rate - a.sell_through_rate,
  );
  if (stSorted.length > 0) {
    const top = stSorted[0];
    insights.push({
      title: 'Top Performer',
      value: top.product_id,
      sub: `${Math.round(top.sell_through_rate)}% sell-through · ${top.category}`,
      accent: 'emerald',
    });
  }

  // Most urgent stockout (lowest days_of_supply among critical)
  const criticals = [...daysOfSupply.products]
    .filter(p => p.stock_health === 'critical')
    .sort((a, b) => a.days_of_supply - b.days_of_supply);
  if (criticals.length > 0) {
    const urgent = criticals[0];
    insights.push({
      title: 'Urgent Stockout',
      value: urgent.product_id,
      sub: `${Math.round(urgent.days_of_supply)}d remaining · ${urgent.category}`,
      accent: 'red',
    });
  } else {
    // Fall back to lowest days_of_supply overall
    const lowest = [...daysOfSupply.products].sort((a, b) => a.days_of_supply - b.days_of_supply);
    if (lowest.length > 0) {
      const p = lowest[0];
      insights.push({
        title: 'Lowest Supply',
        value: p.product_id,
        sub: `${Math.round(p.days_of_supply)}d remaining · ${p.category}`,
        accent: 'amber',
      });
    }
  }

  // Largest positive variance anomaly
  const posVariance = [...shrinkage.products]
    .filter(p => p.shrinkage > 0)
    .sort((a, b) => b.shrinkage - a.shrinkage);
  if (posVariance.length > 0) {
    const top = posVariance[0];
    insights.push({
      title: 'Variance Anomaly',
      value: top.product_id,
      sub: `+${top.shrinkage.toLocaleString()} units unaccounted · ${top.category}`,
      accent: 'amber',
    });
  }

  // Best stock position (highest days_of_supply, healthy status)
  const healthiest = [...daysOfSupply.products]
    .filter(p => p.stock_health === 'healthy')
    .sort((a, b) => b.days_of_supply - a.days_of_supply);
  if (healthiest.length > 0) {
    const best = healthiest[0];
    insights.push({
      title: 'Best Stock Position',
      value: best.product_id,
      sub: `${Math.round(best.days_of_supply)}d of supply · ${best.category}`,
      accent: 'blue',
    });
  }

  if (insights.length === 0) {
    return (
      <div className="flex items-center justify-center py-8 text-sm text-slate-400">
        No insights available.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {insights.map(insight => {
        const s = ACCENT[insight.accent];
        return (
          <div
            key={insight.title}
            className={`rounded-xl border ${s.border} ${s.bg} px-4 py-3.5`}
          >
            <div className="mb-1 flex items-center gap-2">
              <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${s.dot}`} />
              <p className={`text-[10px] font-semibold uppercase tracking-widest ${s.label}`}>
                {insight.title}
              </p>
            </div>
            <p className="truncate text-sm font-semibold text-slate-800">{insight.value}</p>
            <p className="mt-0.5 truncate text-xs text-slate-500">{insight.sub}</p>
          </div>
        );
      })}
    </div>
  );
}
