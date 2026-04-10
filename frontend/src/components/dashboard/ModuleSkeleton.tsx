// Generic animated skeleton for analytics module panels.
// Mimics a horizontal bar-chart layout (label | bar | value).

const ROWS = [
  { labelW: 'w-14', barW: '82%' },
  { labelW: 'w-16', barW: '65%' },
  { labelW: 'w-12', barW: '91%' },
  { labelW: 'w-14', barW: '73%' },
  { labelW: 'w-10', barW: '54%' },
  { labelW: 'w-16', barW: '88%' },
  { labelW: 'w-12', barW: '44%' },
  { labelW: 'w-14', barW: '78%' },
];

export default function ModuleSkeleton() {
  return (
    <div className="animate-pulse">
      {/* Callout placeholder row */}
      <div className="mb-4 grid grid-cols-2 gap-3">
        <div className="h-16 rounded-lg bg-slate-100" />
        <div className="h-16 rounded-lg bg-slate-100" />
      </div>

      {/* Bar rows */}
      <div className="space-y-2.5">
        {ROWS.map((row, i) => (
          <div key={i} className="flex items-center gap-2.5">
            <div className={`h-2.5 shrink-0 rounded-full bg-slate-100 ${row.labelW}`} />
            <div className="flex-1 h-4 overflow-hidden rounded bg-slate-100">
              <div className="h-full rounded bg-slate-100" style={{ width: row.barW }} />
            </div>
            <div className="h-2.5 w-8 shrink-0 rounded-full bg-slate-100" />
          </div>
        ))}
      </div>

      {/* Footer note placeholder */}
      <div className="mt-4 h-2.5 w-48 rounded-full bg-slate-100" />
    </div>
  );
}
