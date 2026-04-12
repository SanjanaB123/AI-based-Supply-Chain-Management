import { useInventoryData } from '../hooks/useInventoryData';

import RiskSpotlightPanel from '../components/dashboard/RiskSpotlightPanel';
import LeadTimeRiskModule from '../components/dashboard/LeadTimeRiskModule';
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

export default function RiskPage() {
  const { leadTimeRisk, isLoading } = useInventoryData(
    'Risk',
    'Lead-time risk analysis and stockout warnings',
  );

  return (
    <div className="space-y-6">
      {/* Risk Spotlight */}
      <section>
        <SectionLabel>Risk Spotlight</SectionLabel>
        <ModuleCard>
          {isLoading ? <SmallPanelSkeleton /> : leadTimeRisk ? <RiskSpotlightPanel data={leadTimeRisk} /> : <EmptyModule />}
        </ModuleCard>
      </section>

      {/* Lead-Time Risk */}
      <section>
        <SectionLabel>Lead-Time Risk Analysis</SectionLabel>
        <ModuleCard>
          {isLoading ? <ModuleSkeleton /> : leadTimeRisk ? <LeadTimeRiskModule data={leadTimeRisk} /> : <EmptyModule />}
        </ModuleCard>
      </section>
    </div>
  );
}
