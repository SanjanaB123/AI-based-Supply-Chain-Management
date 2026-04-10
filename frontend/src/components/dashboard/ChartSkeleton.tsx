export default function ChartSkeleton() {
  return (
    <div className="animate-pulse rounded-xl bg-white shadow-sm p-6">
      <div className="mb-6 h-4 w-52 rounded bg-gray-200" />
      <div className="flex flex-col gap-8 md:flex-row md:items-center md:gap-12">
        <div className="mx-auto h-52 w-52 flex-shrink-0 rounded-full bg-gray-200 md:mx-0" />
        <div className="flex-1 space-y-4">
          <div className="h-4 w-full rounded bg-gray-200" />
          <div className="h-4 w-3/4 rounded bg-gray-200" />
          <div className="h-4 w-5/6 rounded bg-gray-200" />
          <div className="h-px w-full bg-gray-100" />
          <div className="h-4 w-2/5 rounded bg-gray-200" />
        </div>
      </div>
    </div>
  );
}
