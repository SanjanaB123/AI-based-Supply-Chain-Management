import { useInventoryData } from '../hooks/useInventoryData';

import VarianceHighlights from '../components/dashboard/VarianceHighlights';
import ShrinkageModule from '../components/dashboard/ShrinkageModule';
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

function SmallPanelSkeleton() {
  return (
    <div className="animate-pulse space-y-3">
      {[0, 1, 2, 3, 4].map((i) => (
        <div key={i} className="flex items-center justify-between rounded-lg border border-slate-50 dark:border-slate-800 px-3 py-2.5">
          <div className="space-y-1.5">
            <div className="h-3 w-28 rounded-full bg-slate-100 dark:bg-slate-700" />
            <div className="h-2 w-16 rounded-full bg-slate-100 dark:bg-slate-700" />
          </div>
          <div className="h-3 w-10 rounded-full bg-slate-100 dark:bg-slate-700" />
        </div>
      ))}
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

export default function VariancePage() {
  const { shrinkage, isLoading } = useInventoryData(
    'Variance',
    'Shrinkage tracking and variance analysis',
  );

  return (
    <div className="space-y-6">
      {/* Variance Highlights */}
      <section>
        <SectionLabel>Variance Highlights</SectionLabel>
        <ModuleCard>
          {isLoading ? <SmallPanelSkeleton /> : shrinkage ? <VarianceHighlights data={shrinkage} /> : <EmptyModule />}
        </ModuleCard>
      </section>

      {/* Shrinkage Details */}
      <section>
        <SectionLabel>Shrinkage Analysis</SectionLabel>
        <ModuleCard>
          {isLoading ? <ModuleSkeleton /> : shrinkage ? <ShrinkageModule data={shrinkage} /> : <EmptyModule />}
        </ModuleCard>
      </section>
    </div>
  );
}
