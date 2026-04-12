import { useState } from 'react';
import { useInventoryData } from '../hooks/useInventoryData';

import StockHealthChart from '../components/dashboard/StockHealthChart';
import ChartSkeleton from '../components/dashboard/ChartSkeleton';
import ModuleSkeleton from '../components/dashboard/ModuleSkeleton';
import SellThroughModule from '../components/dashboard/SellThroughModule';
import DaysOfSupplyModule from '../components/dashboard/DaysOfSupplyModule';

type AnalyticsTab = 'sell-through' | 'days-of-supply';

const TABS: Array<{ id: AnalyticsTab; label: string }> = [
  { id: 'sell-through', label: 'Sell-Through' },
  { id: 'days-of-supply', label: 'Days of Supply' },
];

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-400 dark:text-slate-500 xl:text-[11px]">
      {children}
    </h2>
  );
}

function EmptyModule() {
  return (
    <div className="flex items-center justify-center py-10 text-sm text-slate-400 dark:text-slate-500">
      Select a store to view data.
    </div>
  );
}

export default function AnalyticsPage() {
  const { stockHealth, sellThrough, daysOfSupply, isLoading } = useInventoryData(
    'Analytics',
    'Health distribution and sales performance metrics',
  );

  const [activeTab, setActiveTab] = useState<AnalyticsTab>('sell-through');

  function renderActiveModule() {
    if (isLoading) return <ModuleSkeleton />;
    if (activeTab === 'sell-through') return sellThrough ? <SellThroughModule data={sellThrough} /> : <EmptyModule />;
    if (activeTab === 'days-of-supply') return daysOfSupply ? <DaysOfSupplyModule data={daysOfSupply} /> : <EmptyModule />;
    return null;
  }

  return (
    <div className="space-y-6">
      {/* Health Distribution */}
      <section>
        <SectionLabel>Health Distribution</SectionLabel>
        {isLoading ? <ChartSkeleton /> : stockHealth ? <StockHealthChart data={stockHealth} /> : null}
      </section>

      {/* Analytics Tabs */}
      <section>
        <SectionLabel>Performance Analytics</SectionLabel>
        <div className="rounded-xl border border-slate-200/80 dark:border-slate-700/80 bg-white dark:bg-slate-800 shadow-sm">
          <div className="flex items-center gap-1 overflow-x-auto border-b border-slate-100 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-900/60 px-3 py-2 rounded-t-xl">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`shrink-0 rounded-lg px-3.5 py-1.5 text-[12px] font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-500 dark:text-slate-400 hover:bg-slate-200/70 dark:hover:bg-slate-700 hover:text-slate-800 dark:hover:text-slate-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="p-5">{renderActiveModule()}</div>
        </div>
      </section>
    </div>
  );
}
