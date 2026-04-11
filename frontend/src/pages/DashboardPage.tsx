import { useInventoryData } from '../hooks/useInventoryData';

import StoreSelector from '../components/dashboard/StoreSelector';
import KpiCard from '../components/dashboard/KpiCard';
import KpiSkeleton from '../components/dashboard/KpiSkeleton';
import InventoryTrendChart from '../components/dashboard/InventoryTrendChart';

// ── Helpers ──────────────────────────────────────────────────────────────────

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

function AlertIcon() {
  return (
    <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm-.75 3.75a.75.75 0 0 1 1.5 0v3.5a.75.75 0 0 1-1.5 0v-3.5zm.75 7a.875.875 0 1 1 0-1.75.875.875 0 0 1 0 1.75z" />
    </svg>
  );
}

function EmptyModule() {
  return (
    <div className="flex items-center justify-center py-10 text-sm text-slate-400 dark:text-slate-500">
      Select a store to view data.
    </div>
  );
}

// ── Dashboard ────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const {
    stores, selectedStore, storesLoading, storesError,
    stockLevels, daysOfSupply,
    dataError, dataLoading, isLoading, handleStoreChange,
  } = useInventoryData('Inventory Overview', 'Real-time stock intelligence across your store network');

  return (
    <div className="space-y-6">
      {/* Mobile store selector */}
      <div className="md:hidden">
        {storesLoading ? (
          <div className="h-9 w-40 animate-pulse rounded-lg bg-slate-200 dark:bg-slate-700" />
        ) : !storesError && stores.length > 0 ? (
          <StoreSelector stores={stores} selected={selectedStore} onChange={handleStoreChange} />
        ) : null}
      </div>

      {/* Error banners */}
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

      {/* KPI row */}
      <section>
        <SectionLabel>Stock Status</SectionLabel>
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {isLoading ? (
            <><KpiSkeleton /><KpiSkeleton /><KpiSkeleton /><KpiSkeleton /></>
          ) : stockLevels ? (
            <>
              <KpiCard label="Critical" value={stockLevels.summary.critical} subtext="need immediate reorder" color="red" />
              <KpiCard label="Low Stock" value={stockLevels.summary.low} subtext="approaching reorder point" color="amber" />
              <KpiCard label="Healthy" value={stockLevels.summary.healthy} subtext="adequate supply" color="emerald" />
              <KpiCard label="Total Units" value={stockLevels.total_stock} subtext="across all products" color="indigo" />
            </>
          ) : null}
        </div>
      </section>

      {/* Inventory Trend */}
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
              </div>
              <div className="h-56 rounded-lg bg-slate-100 dark:bg-slate-700" />
            </div>
          ) : daysOfSupply ? (
            <InventoryTrendChart data={daysOfSupply} />
          ) : (
            <EmptyModule />
          )}
        </ModuleCard>
      </section>
    </div>
  );
}
