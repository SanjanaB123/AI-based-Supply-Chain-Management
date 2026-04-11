import { useInventoryData } from '../hooks/useInventoryData';

import CriticalItemsTable from '../components/dashboard/CriticalItemsTable';
import InventoryHeatmapModule from '../components/dashboard/InventoryHeatmapModule';
import CategoryBreakdown from '../components/dashboard/CategoryBreakdown';
import ModuleSkeleton from '../components/dashboard/ModuleSkeleton';

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-400 dark:text-slate-500 xl:text-[11px]">
      {children}
    </h2>
  );
}

function ModuleCard({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-slate-200/80 dark:border-slate-700/80 bg-white dark:bg-slate-800 p-5 shadow-sm ${className}`}>
      {children}
    </div>
  );
}

function EmptyModule() {
  return (
    <div className="flex items-center justify-center py-10 text-sm text-slate-400 dark:text-slate-500">
      Select a store to view data.
    </div>
  );
}

function InsightSkeleton() {
  return (
    <div className="animate-pulse space-y-3">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="rounded-xl border border-slate-100 dark:border-slate-800 px-4 py-3.5 space-y-1.5">
          <div className="h-2 w-20 rounded-full bg-slate-100 dark:bg-slate-700" />
          <div className="h-3.5 w-32 rounded-full bg-slate-100 dark:bg-slate-700" />
        </div>
      ))}
    </div>
  );
}

function CategorySkeleton() {
  return (
    <div className="animate-pulse space-y-4">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="space-y-1.5">
          <div className="flex justify-between">
            <div className="h-3 w-20 rounded-full bg-slate-100 dark:bg-slate-700" />
            <div className="h-2.5 w-6 rounded-full bg-slate-100 dark:bg-slate-700" />
          </div>
          <div className="h-2 w-full rounded-full bg-slate-100 dark:bg-slate-700" />
        </div>
      ))}
    </div>
  );
}

export default function InventoryPage() {
  const { stockLevels, daysOfSupply, leadTimeRisk, isLoading } = useInventoryData(
    'Inventory',
    'Urgent items, density mapping, and category breakdown',
  );

  return (
    <div className="space-y-6">
      {/* Urgent Items Table */}
      <section>
        <SectionLabel>Urgent Items</SectionLabel>
        <ModuleCard>
          {isLoading ? (
            <ModuleSkeleton />
          ) : daysOfSupply && leadTimeRisk ? (
            <CriticalItemsTable dosData={daysOfSupply} ltrData={leadTimeRisk} />
          ) : (
            <EmptyModule />
          )}
        </ModuleCard>
      </section>

      {/* Density + Category */}
      <section>
        <div className="grid gap-5 xl:grid-cols-2 xl:items-start">
          <div>
            <SectionLabel>Inventory Density</SectionLabel>
            <ModuleCard>
              {isLoading ? <InsightSkeleton /> : daysOfSupply ? <InventoryHeatmapModule data={daysOfSupply} /> : <EmptyModule />}
            </ModuleCard>
          </div>
          <div>
            <SectionLabel>Category Breakdown</SectionLabel>
            <ModuleCard>
              {isLoading ? <CategorySkeleton /> : stockLevels ? <CategoryBreakdown data={stockLevels} /> : <EmptyModule />}
            </ModuleCard>
          </div>
        </div>
      </section>
    </div>
  );
}
