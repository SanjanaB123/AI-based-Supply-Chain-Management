import { useCallback, useEffect, useState } from 'react';
import { useCurrentUser } from '../hooks/useCurrentUser';
import { useTopBar } from '../app/TopBarContext';
import {
  fetchStores,
  fetchStockLevels,
  fetchStockHealth,
  fetchSellThrough,
  fetchDaysOfSupply,
  fetchLeadTimeRisk,
  fetchShrinkage,
} from '../lib/inventory';
import type {
  StockLevelsResponse,
  StockHealthResponse,
  SellThroughResponse,
  DaysOfSupplyResponse,
  LeadTimeRiskResponse,
  ShrinkageResponse,
} from '../types/inventory';

import StoreSelector from '../components/dashboard/StoreSelector';
import KpiCard from '../components/dashboard/KpiCard';
import KpiSkeleton from '../components/dashboard/KpiSkeleton';
import StockHealthChart from '../components/dashboard/StockHealthChart';
import ChartSkeleton from '../components/dashboard/ChartSkeleton';
import ModuleSkeleton from '../components/dashboard/ModuleSkeleton';
import SellThroughModule from '../components/dashboard/SellThroughModule';
import DaysOfSupplyModule from '../components/dashboard/DaysOfSupplyModule';
import LeadTimeRiskModule from '../components/dashboard/LeadTimeRiskModule';
import ShrinkageModule from '../components/dashboard/ShrinkageModule';
import CriticalItemsTable from '../components/dashboard/CriticalItemsTable';
import InventoryHeatmapModule from '../components/dashboard/InventoryHeatmapModule';
import InventoryTrendChart from '../components/dashboard/InventoryTrendChart';
import CategoryBreakdown from '../components/dashboard/CategoryBreakdown';
import RiskSpotlightPanel from '../components/dashboard/RiskSpotlightPanel';
import VarianceHighlights from '../components/dashboard/VarianceHighlights';

// ── Types ─────────────────────────────────────────────────────────────────────

type AnalyticsTab = 'sell-through' | 'days-of-supply' | 'lead-time-risk' | 'shrinkage';

const ANALYTICS_TABS: Array<{ id: AnalyticsTab; label: string }> = [
  { id: 'sell-through',   label: 'Sell-Through'   },
  { id: 'days-of-supply', label: 'Days of Supply' },
  { id: 'lead-time-risk', label: 'Lead-Time Risk' },
  { id: 'shrinkage',      label: 'Variance'       },
];

// ── Skeletons for lower panels ────────────────────────────────────────────────

function SmallPanelSkeleton() {
  return (
    <div className="animate-pulse space-y-3">
      <div className="flex items-center gap-2">
        <div className="h-1.5 w-1.5 rounded-full bg-slate-100 dark:bg-slate-700" />
        <div className="h-2.5 w-24 rounded-full bg-slate-100 dark:bg-slate-700" />
      </div>
      {[0, 1, 2, 3, 4].map(i => (
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

function CategorySkeleton() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="flex gap-4">
        {[0, 1, 2].map(i => (
          <div key={i} className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-slate-100 dark:bg-slate-700" />
            <div className="h-2 w-10 rounded-full bg-slate-100 dark:bg-slate-700" />
          </div>
        ))}
      </div>
      {[0, 1, 2, 3].map(i => (
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

function InsightSkeleton() {
  return (
    <div className="animate-pulse space-y-3">
      {[0, 1, 2, 3].map(i => (
        <div key={i} className="rounded-xl border border-slate-100 dark:border-slate-800 px-4 py-3.5 space-y-1.5">
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-1.5 rounded-full bg-slate-100 dark:bg-slate-700" />
            <div className="h-2 w-20 rounded-full bg-slate-100 dark:bg-slate-700" />
          </div>
          <div className="h-3.5 w-32 rounded-full bg-slate-100 dark:bg-slate-700" />
          <div className="h-2.5 w-44 rounded-full bg-slate-100 dark:bg-slate-700" />
        </div>
      ))}
    </div>
  );
}

// ── Alert icon ────────────────────────────────────────────────────────────────

function AlertIcon() {
  return (
    <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm-.75 3.75a.75.75 0 0 1 1.5 0v3.5a.75.75 0 0 1-1.5 0v-3.5zm.75 7a.875.875 0 1 1 0-1.75.875.875 0 0 1 0 1.75z" />
    </svg>
  );
}

// ── Section header + card helpers ─────────────────────────────────────────────

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

// ── Dashboard ─────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { getToken } = useCurrentUser();
  const { setTopBarSlot, setPageMeta } = useTopBar();

  // Store list
  const [stores, setStores]               = useState<string[]>([]);
  const [selectedStore, setSelectedStore] = useState('');
  const [storesLoading, setStoresLoading] = useState(true);
  const [storesError, setStoresError]     = useState<string | null>(null);

  // Store data — fetched in parallel
  const [stockLevels, setStockLevels]   = useState<StockLevelsResponse | null>(null);
  const [stockHealth, setStockHealth]   = useState<StockHealthResponse | null>(null);
  const [sellThrough, setSellThrough]   = useState<SellThroughResponse | null>(null);
  const [daysOfSupply, setDaysOfSupply] = useState<DaysOfSupplyResponse | null>(null);
  const [leadTimeRisk, setLeadTimeRisk] = useState<LeadTimeRiskResponse | null>(null);
  const [shrinkage, setShrinkage]       = useState<ShrinkageResponse | null>(null);

  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError]     = useState<string | null>(null);

  // Active analytics tab
  const [activeTab, setActiveTab] = useState<AnalyticsTab>('sell-through');

  // ── Set page title in top bar ─────────────────────────────────────────────

  useEffect(() => {
    setPageMeta(
      'Inventory Overview',
      'Real-time stock intelligence across your store network',
    );
    return () => setPageMeta('', '');
  }, [setPageMeta]);

  // ── Store change handler (stable ref) ────────────────────────────────────

  const handleStoreChange = useCallback((store: string) => {
    setSelectedStore(store);
    setStockLevels(null);
    setStockHealth(null);
    setSellThrough(null);
    setDaysOfSupply(null);
    setLeadTimeRisk(null);
    setShrinkage(null);
  }, []);

  // ── Inject store selector into desktop top bar ────────────────────────────

  useEffect(() => {
    if (storesLoading) {
      setTopBarSlot(
        <div className="h-9 w-40 animate-pulse rounded-lg bg-slate-200 dark:bg-slate-700" />,
      );
    } else if (stores.length > 0) {
      setTopBarSlot(
        <StoreSelector
          stores={stores}
          selected={selectedStore}
          onChange={handleStoreChange}
          variant="light"
        />,
      );
    } else {
      setTopBarSlot(null);
    }
    return () => setTopBarSlot(null);
  }, [stores, selectedStore, storesLoading, setTopBarSlot, handleStoreChange]);

  // ── Load stores on mount ──────────────────────────────────────────────────

  useEffect(() => {
    (async () => {
      try {
        const token = await getToken();
        const result = await fetchStores(token);
        setStores(result.stores);
        if (result.stores.length > 0) setSelectedStore(result.stores[0]);
      } catch {
        setStoresError('Unable to load stores. Check your connection and try again.');
      } finally {
        setStoresLoading(false);
      }
    })();
  }, [getToken]);

  // ── Load all store data in parallel when store changes ───────────────────

  useEffect(() => {
    if (!selectedStore) return;
    let cancelled = false;

    (async () => {
      setDataLoading(true);
      setDataError(null);
      try {
        const token = await getToken();
        const [levels, health, st, dos, ltr, shr] = await Promise.all([
          fetchStockLevels(selectedStore, token),
          fetchStockHealth(selectedStore, token),
          fetchSellThrough(selectedStore, token),
          fetchDaysOfSupply(selectedStore, token),
          fetchLeadTimeRisk(selectedStore, token),
          fetchShrinkage(selectedStore, token),
        ]);
        if (!cancelled) {
          setStockLevels(levels);
          setStockHealth(health);
          setSellThrough(st);
          setDaysOfSupply(dos);
          setLeadTimeRisk(ltr);
          setShrinkage(shr);
        }
      } catch {
        if (!cancelled) setDataError('Unable to load store data. Please try again.');
      } finally {
        if (!cancelled) setDataLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedStore, getToken]);

  const isLoading = storesLoading || dataLoading;

  // ── Render active analytics module ────────────────────────────────────────

  function renderActiveModule() {
    if (isLoading) return <ModuleSkeleton />;

    if (activeTab === 'sell-through') {
      if (!sellThrough) return <EmptyModule />;
      return <SellThroughModule data={sellThrough} />;
    }
    if (activeTab === 'days-of-supply') {
      if (!daysOfSupply) return <EmptyModule />;
      return <DaysOfSupplyModule data={daysOfSupply} />;
    }
    if (activeTab === 'lead-time-risk') {
      if (!leadTimeRisk) return <EmptyModule />;
      return <LeadTimeRiskModule data={leadTimeRisk} />;
    }
    if (activeTab === 'shrinkage') {
      if (!shrinkage) return <EmptyModule />;
      return <ShrinkageModule data={shrinkage} />;
    }
    return null;
  }

  // ── Layout ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">

      {/* ── Mobile-only store selector ──────────────────────────────────────── */}
      <div className="md:hidden">
        {storesLoading ? (
          <div className="h-9 w-40 animate-pulse rounded-lg bg-slate-200 dark:bg-slate-700" />
        ) : !storesError && stores.length > 0 ? (
          <StoreSelector stores={stores} selected={selectedStore} onChange={handleStoreChange} />
        ) : null}
      </div>

      {/* ── Error banners ───────────────────────────────────────────────────── */}
      {storesError && (
        <div className="flex items-start gap-3 rounded-xl border border-red-200 dark:border-red-900/50 bg-red-100 dark:bg-red-900/20 px-4 py-3.5 text-sm text-red-600 dark:text-red-400">
          <AlertIcon />
          {storesError}
        </div>
      )}
      {dataError && !dataLoading && (
        <div className="flex items-start gap-3 rounded-xl border border-red-200 dark:border-red-900/50 bg-red-100 dark:bg-red-900/20 px-4 py-3.5 text-sm text-red-600 dark:text-red-400">
          <AlertIcon />
          {dataError}
        </div>
      )}

      {/* ── KPI row ─────────────────────────────────────────────────────────── */}
      <section>
        <SectionLabel>Stock Status</SectionLabel>
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {isLoading ? (
            <>
              <KpiSkeleton />
              <KpiSkeleton />
              <KpiSkeleton />
              <KpiSkeleton />
            </>
          ) : stockLevels ? (
            <>
              <KpiCard
                label="Critical"
                value={stockLevels.summary.critical}
                subtext="need immediate reorder"
                color="red"
              />
              <KpiCard
                label="Low Stock"
                value={stockLevels.summary.low}
                subtext="approaching reorder point"
                color="amber"
              />
              <KpiCard
                label="Healthy"
                value={stockLevels.summary.healthy}
                subtext="adequate supply"
                color="emerald"
              />
              <KpiCard
                label="Total Units"
                value={stockLevels.total_stock}
                subtext="across all products"
                color="indigo"
              />
            </>
          ) : null}
        </div>
      </section>

      {/* ── Full-width: Inventory Health Trend ──────────────────────────────── */}
      <section>
        <SectionLabel>Inventory Trend</SectionLabel>
        <ModuleCard>
          {isLoading ? (
            <div className="animate-pulse space-y-4">
              <div className="flex items-start justify-between">
                <div className="space-y-1.5">
                  <div className="h-3.5 w-44 rounded-full bg-slate-100 dark:bg-slate-700" />
                  <div className="h-2.5 w-64 rounded-full bg-slate-100 dark:bg-slate-700" />
                </div>
                <div className="flex gap-1">
                  {[0,1,2,3,4].map(i => (
                    <div key={i} className="h-7 w-20 rounded-md bg-slate-100 dark:bg-slate-700" />
                  ))}
                </div>
              </div>
              <div className="h-56 rounded-lg bg-slate-100 dark:bg-slate-700" />
              <div className="flex gap-4 border-t border-slate-100 dark:border-slate-700 pt-3">
                {[0,1,2].map(i => (
                  <div key={i} className="flex items-center gap-1.5">
                    <div className="h-2 w-2 rounded-full bg-slate-100 dark:bg-slate-700" />
                    <div className="h-2.5 w-14 rounded-full bg-slate-100 dark:bg-slate-700" />
                  </div>
                ))}
              </div>
            </div>
          ) : daysOfSupply ? (
            <InventoryTrendChart data={daysOfSupply} />
          ) : (
            <EmptyModule />
          )}
        </ModuleCard>
      </section>

      {/* ── Health chart + Analytics tabs ───────────────────────────────────── */}
      <section>
        <div className="grid gap-5 xl:grid-cols-3 xl:items-start">

          {/* Health distribution chart */}
          <div className="xl:col-span-1">
            <SectionLabel>Health Distribution</SectionLabel>
            {isLoading ? (
              <ChartSkeleton />
            ) : stockHealth ? (
              <StockHealthChart data={stockHealth} />
            ) : null}
          </div>

          {/* Analytics panel */}
          <div className="xl:col-span-2">
            <SectionLabel>Analytics</SectionLabel>
            <div className="rounded-xl border border-slate-200/80 dark:border-slate-700/80 bg-white dark:bg-slate-800 shadow-sm">
              {/* Tab bar */}
              <div className="flex items-center gap-1 overflow-x-auto border-b border-slate-100 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-900/60 px-3 py-2 rounded-t-xl">
                {ANALYTICS_TABS.map(tab => (
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
              {/* Module content */}
              <div className="p-5 h-117 overflow-x-auto">
                {renderActiveModule()}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Lower grid: Urgent Items + Inventory Density ────────────────────── */}
      <section>
        <div className="grid gap-5 xl:grid-cols-3 xl:items-start">

          {/* Critical items table */}
          <div className="xl:col-span-2">
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
          </div>

          {/* Inventory density heatmap */}
          <div className="xl:col-span-1">
            <SectionLabel>Inventory Density</SectionLabel>
            <ModuleCard>
              {isLoading ? (
                <InsightSkeleton />
              ) : daysOfSupply ? (
                <InventoryHeatmapModule data={daysOfSupply} />
              ) : (
                <EmptyModule />
              )}
            </ModuleCard>
          </div>
        </div>
      </section>

      {/* ── Lower grid: Risk + Category + Variance ───────────────────────────── */}
      <section>
        <div className="grid gap-5 xl:grid-cols-3 xl:items-start">

          {/* Risk spotlight */}
          <div>
            <SectionLabel>Risk Spotlight</SectionLabel>
            <ModuleCard>
              {isLoading ? (
                <SmallPanelSkeleton />
              ) : leadTimeRisk ? (
                <RiskSpotlightPanel data={leadTimeRisk} />
              ) : (
                <EmptyModule />
              )}
            </ModuleCard>
          </div>

          {/* Category breakdown */}
          <div>
            <SectionLabel>Category Breakdown</SectionLabel>
            <ModuleCard>
              {isLoading ? (
                <CategorySkeleton />
              ) : stockLevels ? (
                <CategoryBreakdown data={stockLevels} />
              ) : (
                <EmptyModule />
              )}
            </ModuleCard>
          </div>

          {/* Variance highlights */}
          <div>
            <SectionLabel>Variance Highlights</SectionLabel>
            <ModuleCard>
              {isLoading ? (
                <SmallPanelSkeleton />
              ) : shrinkage ? (
                <VarianceHighlights data={shrinkage} />
              ) : (
                <EmptyModule />
              )}
            </ModuleCard>
          </div>
        </div>
      </section>

    </div>
  );
}

// ── Empty module fallback ─────────────────────────────────────────────────────

function EmptyModule() {
  return (
    <div className="flex items-center justify-center py-10 text-sm text-slate-400 dark:text-slate-500">
      Select a store to view data.
    </div>
  );
}
