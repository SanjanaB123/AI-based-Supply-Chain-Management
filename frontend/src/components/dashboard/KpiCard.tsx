type ColorVariant = 'red' | 'amber' | 'emerald' | 'indigo';

interface Props {
  label: string;
  value: number;
  subtext: string;
  color: ColorVariant;
}

const barColor: Record<ColorVariant, string> = {
  red: 'bg-red-500',
  amber: 'bg-amber-500',
  emerald: 'bg-emerald-500',
  indigo: 'bg-indigo-500',
};

const valueColor: Record<ColorVariant, string> = {
  red: 'text-red-600',
  amber: 'text-amber-600',
  emerald: 'text-emerald-600',
  indigo: 'text-indigo-600',
};

export default function KpiCard({ label, value, subtext, color }: Props) {
  return (
    <div className="overflow-hidden rounded-xl bg-white shadow-sm">
      <div className={`h-1.5 ${barColor[color]}`} />
      <div className="p-6">
        <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">{label}</p>
        <p className={`mt-2 text-4xl font-bold ${valueColor[color]}`}>
          {value.toLocaleString()}
        </p>
        <p className="mt-1.5 text-sm text-gray-500">{subtext}</p>
      </div>
    </div>
  );
}
