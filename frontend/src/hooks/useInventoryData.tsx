/**
 * Page-level hook that connects to the shared InventoryDataContext
 * and sets the page title + store selector in the top bar.
 */
import { useEffect } from 'react';
import { useTopBar } from '../app/TopBarContext';
import { useSharedInventoryData } from './InventoryDataContext';
import type { InventoryData } from './InventoryDataContext';
import StoreSelector from '../components/dashboard/StoreSelector';

export type { InventoryData };

export function useInventoryData(pageTitle: string, pageSubtitle: string): InventoryData {
  const { setTopBarSlot, setPageMeta } = useTopBar();
  const data = useSharedInventoryData();

  // Set page title
  useEffect(() => {
    setPageMeta(pageTitle, pageSubtitle);
    return () => setPageMeta('', '');
  }, [setPageMeta, pageTitle, pageSubtitle]);

  // Inject store selector into top bar
  useEffect(() => {
    if (data.storesLoading) {
      setTopBarSlot(
        <div className="h-9 w-40 animate-pulse rounded-lg bg-slate-200 dark:bg-slate-700" />,
      );
    } else if (data.stores.length > 0) {
      setTopBarSlot(
        <StoreSelector
          stores={data.stores}
          selected={data.selectedStore}
          onChange={data.handleStoreChange}
          variant="light"
        />,
      );
    } else {
      setTopBarSlot(null);
    }
    return () => setTopBarSlot(null);
  }, [data.stores, data.selectedStore, data.storesLoading, setTopBarSlot, data.handleStoreChange]);

  return data;
}
