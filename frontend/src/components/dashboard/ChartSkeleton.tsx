export default function ChartSkeleton() {
  return (
    <div className="animate-pulse rounded-xl border border-slate-100 dark:border-slate-700/80 bg-white dark:bg-slate-800 shadow-sm">
      <div className="flex flex-col gap-6 p-5">

        {/* Donut placeholder */}
        <div className="mx-auto h-44 w-44 shrink-0 rounded-full bg-slate-100 dark:bg-slate-700" />

        {/* Legend placeholder */}
        <div className="min-w-0">
          {[0, 1, 2].map(i => (
            <div
              key={i}
              className="flex items-center justify-between py-3 border-b border-slate-50 dark:border-slate-800 last:border-0"
            >
              <div className="flex items-start gap-3">
                <div className="mt-1 h-2 w-2 rounded-full bg-slate-100 dark:bg-slate-700" />
                <div className="space-y-1.5">
                  <div className="h-3 w-16 rounded-full bg-slate-100 dark:bg-slate-700" />
                  <div className="h-2.5 w-32 rounded-full bg-slate-100 dark:bg-slate-700" />
                </div>
              </div>
              <div className="flex gap-3 pl-4">
                <div className="h-3 w-5 rounded-full bg-slate-100 dark:bg-slate-700" />
                <div className="h-3 w-8 rounded-full bg-slate-100 dark:bg-slate-700" />
              </div>
            </div>
          ))}
          <div className="mt-2.5 flex items-center justify-between border-t border-slate-100 dark:border-slate-700 pt-3">
            <div className="h-2.5 w-10 rounded-full bg-slate-100 dark:bg-slate-700" />
            <div className="h-3 w-14 rounded-full bg-slate-100 dark:bg-slate-700" />
          </div>
        </div>
      </div>

      {/* Footer placeholder */}
      <div className="flex items-center justify-between border-t border-slate-50 dark:border-slate-800 px-5 py-3.5">
        <div className="h-2.5 w-28 rounded-full bg-slate-100 dark:bg-slate-700" />
        <div className="h-1.5 w-24 rounded-full bg-slate-100 dark:bg-slate-700" />
      </div>
    </div>
  );
}
