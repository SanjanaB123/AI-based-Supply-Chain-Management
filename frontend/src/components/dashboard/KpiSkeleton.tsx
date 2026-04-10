export default function KpiSkeleton() {
  return (
    <div className="animate-pulse overflow-hidden rounded-xl bg-white shadow-sm">
      <div className="h-1.5 bg-gray-200" />
      <div className="p-6">
        <div className="h-3 w-24 rounded bg-gray-200" />
        <div className="mt-3 h-10 w-16 rounded bg-gray-200" />
        <div className="mt-2.5 h-3 w-28 rounded bg-gray-200" />
      </div>
    </div>
  );
}
