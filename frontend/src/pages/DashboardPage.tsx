import { useEffect, useState } from 'react';
import { useCurrentUser } from '../hooks/useCurrentUser';
import { fetchStores, fetchStockLevels, fetchStockHealth } from '../lib/inventory';
import type { StockLevelsResponse, StockHealthResponse } from '../types/inventory';
import StoreSelector from '../components/dashboard/StoreSelector';
import KpiCard from '../components/dashboard/KpiCard';
import KpiSkeleton from '../components/dashboard/KpiSkeleton';
import SectionContainer from '../components/dashboard/SectionContainer';
import StockHealthChart from '../components/dashboard/StockHealthChart';
import ChartSkeleton from '../components/dashboard/ChartSkeleton';

export default function DashboardPage() {
  const { getToken } = useCurrentUser();

  // Store list
  const [stores, setStores] = useState<string[]>([]);
  const [selectedStore, setSelectedStore] = useState('');
  const [storesLoading, setStoresLoading] = useState(true);
  const [storesError, setStoresError] = useState<string | null>(null);

  // Store data
  const [stockLevels, setStockLevels] = useState<StockLevelsResponse | null>(null);
  const [stockHealth, setStockHealth] = useState<StockHealthResponse | null>(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState<string | null>(null);

  // Load store list on mount
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

  // Load stock data whenever selected store changes
  useEffect(() => {
    if (!selectedStore) return;
    let cancelled = false;

    (async () => {
      setDataLoading(true);
      setDataError(null);
      try {
        const token = await getToken();
        const [levels, health] = await Promise.all([
          fetchStockLevels(selectedStore, token),
          fetchStockHealth(selectedStore, token),
        ]);
        if (!cancelled) {
          setStockLevels(levels);
          setStockHealth(health);
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

  return (
    <div className="mx-auto max-w-7xl space-y-8">

      {/* Page header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Inventory Overview</h1>
          <p className="mt-1 text-sm text-gray-500">
            Real-time stock intelligence across your store network
          </p>
        </div>
        {storesLoading ? (
          <div className="h-9 w-36 animate-pulse rounded-lg bg-gray-200" />
        ) : !storesError && stores.length > 0 ? (
          <StoreSelector
            stores={stores}
            selected={selectedStore}
            onChange={store => {
              setSelectedStore(store);
              setStockLevels(null);
              setStockHealth(null);
            }}
          />
        ) : null}
      </div>

      {/* Error banners */}
      {storesError && (
        <div className="rounded-lg border border-red-100 bg-red-50 px-5 py-4 text-sm text-red-700">
          {storesError}
        </div>
      )}
      {dataError && !dataLoading && (
        <div className="rounded-lg border border-red-100 bg-red-50 px-5 py-4 text-sm text-red-700">
          {dataError}
        </div>
      )}

      {/* KPI cards */}
      <SectionContainer title="Stock Status">
        <div className="grid grid-cols-2 gap-5 md:grid-cols-4">
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
                label="Critical Stock"
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
                label="Healthy Stock"
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
      </SectionContainer>

      {/* Stock health chart */}
      <SectionContainer title="Health Distribution">
        {isLoading ? (
          <ChartSkeleton />
        ) : stockHealth ? (
          <StockHealthChart data={stockHealth} />
        ) : null}
      </SectionContainer>

    </div>
  );
}
