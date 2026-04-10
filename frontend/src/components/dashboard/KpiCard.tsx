type ColorVariant = 'red' | 'amber' | 'emerald' | 'indigo';

interface Props {
  label: string;
  value: number;
  subtext: string;
  color: ColorVariant;
}

const dotColor: Record<ColorVariant, string> = {
  red: 'bg-red-500',
  amber: 'bg-amber-500',
  emerald: 'bg-emerald-500',
  indigo: 'bg-indigo-500',
};

export default function KpiCard({ label, value, subtext, color }: Props) {
  return (
    <div className="flex flex-col gap-4 rounded-xl border border-slate-100 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <p className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">
          {label}
        </p>
        <span className={`h-2 w-2 shrink-0 rounded-full ${dotColor[color]}`} />
      </div>
      <div>
        <p className="text-3xl font-bold tracking-tight text-slate-900">
          {value.toLocaleString()}
        </p>
        <p className="mt-1 text-xs text-slate-400">{subtext}</p>
      </div>
    </div>
  );
}
