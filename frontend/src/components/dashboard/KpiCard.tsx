type ColorVariant = 'red' | 'amber' | 'emerald' | 'indigo';

interface Props {
  label: string;
  value: number;
  subtext: string;
  color: ColorVariant;
}

const dotColor: Record<ColorVariant, string> = {
  red:     'bg-red-500',
  amber:   'bg-amber-500',
  emerald: 'bg-emerald-500',
  indigo:  'bg-indigo-500',
};

const accentBar: Record<ColorVariant, string> = {
  red:     'bg-red-500',
  amber:   'bg-amber-500',
  emerald: 'bg-emerald-500',
  indigo:  'bg-indigo-500',
};

const labelColor: Record<ColorVariant, string> = {
  red:     'text-red-600',
  amber:   'text-amber-600',
  emerald: 'text-emerald-600',
  indigo:  'text-indigo-600',
};

export default function KpiCard({ label, value, subtext, color }: Props) {
  return (
    <div className="flex flex-col rounded-xl border border-slate-200/80 bg-white shadow-sm overflow-hidden">
      {/* Colored accent bar across the top */}
      <div className={`h-0.5 w-full ${accentBar[color]}`} />

      <div className="flex flex-col gap-3 p-5">
        {/* Header row */}
        <div className="flex items-center justify-between">
          <p className={`text-[10px] font-semibold uppercase tracking-widest xl:text-[11px] ${labelColor[color]}`}>
            {label}
          </p>
          <span className={`h-2 w-2 shrink-0 rounded-full ${dotColor[color]}`} />
        </div>

        {/* Value + subtext */}
        <div>
          <p className="text-3xl font-bold tracking-tight text-slate-900 xl:text-4xl tabular-nums">
            {value.toLocaleString()}
          </p>
          <p className="mt-1 text-xs text-slate-400 xl:text-[13px]">{subtext}</p>
        </div>
      </div>
    </div>
  );
}
