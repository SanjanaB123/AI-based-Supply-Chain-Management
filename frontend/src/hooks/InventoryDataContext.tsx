import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { useCurrentUser } from './useCurrentUser';
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

export interface InventoryData {
  stores: string[];
  selectedStore: string;
  storesLoading: boolean;
  storesError: string | null;
  stockLevels: StockLevelsResponse | null;
  stockHealth: StockHealthResponse | null;
  sellThrough: SellThroughResponse | null;
  daysOfSupply: DaysOfSupplyResponse | null;
  leadTimeRisk: LeadTimeRiskResponse | null;
  shrinkage: ShrinkageResponse | null;
  dataLoading: boolean;
  dataError: string | null;
  isLoading: boolean;
  handleStoreChange: (store: string) => void;
}

const InventoryDataContext = createContext<InventoryData | null>(null);

export function InventoryDataProvider({ children }: { children: ReactNode }) {
  const { getToken } = useCurrentUser();

  const [stores, setStores] = useState<string[]>([]);
  const [selectedStore, setSelectedStore] = useState('');
  const [storesLoading, setStoresLoading] = useState(true);
  const [storesError, setStoresError] = useState<string | null>(null);

  const [stockLevels, setStockLevels] = useState<StockLevelsResponse | null>(null);
  const [stockHealth, setStockHealth] = useState<StockHealthResponse | null>(null);
  const [sellThrough, setSellThrough] = useState<SellThroughResponse | null>(null);
  const [daysOfSupply, setDaysOfSupply] = useState<DaysOfSupplyResponse | null>(null);
  const [leadTimeRisk, setLeadTimeRisk] = useState<LeadTimeRiskResponse | null>(null);
  const [shrinkage, setShrinkage] = useState<ShrinkageResponse | null>(null);

  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState<string | null>(null);

  const handleStoreChange = useCallback((store: string) => {
    setSelectedStore(store);
    setStockLevels(null);
    setStockHealth(null);
    setSellThrough(null);
    setDaysOfSupply(null);
    setLeadTimeRisk(null);
    setShrinkage(null);
  }, []);

  // Load stores once
  useEffect(() => {
    (async () => {
      try {
        const token = await getToken();
        const result = await fetchStores(token);
        setStores(result.stores);
        if (result.stores.length > 0) setSelectedStore(result.stores[0]);
      } catch {
        setStoresError('Unable to load stores.');
      } finally {
        setStoresLoading(false);
      }
    })();
  }, [getToken]);

  // Load data when store changes
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
        if (!cancelled) setDataError('Unable to load store data.');
      } finally {
        if (!cancelled) setDataLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [selectedStore, getToken]);

  return (
    <InventoryDataContext.Provider
      value={{
        stores, selectedStore, storesLoading, storesError,
        stockLevels, stockHealth, sellThrough, daysOfSupply, leadTimeRisk, shrinkage,
        dataLoading, dataError, isLoading: storesLoading || dataLoading,
        handleStoreChange,
      }}
    >
      {children}
    </InventoryDataContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useSharedInventoryData(): InventoryData {
  const ctx = useContext(InventoryDataContext);
  if (!ctx) throw new Error('useSharedInventoryData must be used within InventoryDataProvider');
  return ctx;
}
