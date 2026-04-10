export default function KpiSkeleton() {
  return (
    <div className="animate-pulse flex flex-col gap-4 rounded-xl border border-slate-100 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="h-2.5 w-20 rounded-full bg-slate-100" />
        <div className="h-2 w-2 rounded-full bg-slate-100" />
      </div>
      <div>
        <div className="h-8 w-14 rounded-lg bg-slate-100" />
        <div className="mt-2 h-2.5 w-32 rounded-full bg-slate-100" />
      </div>
    </div>
  );
}
