import { apiFetch } from './api';
import type { StoresResponse, StockLevelsResponse, StockHealthResponse } from '../types/inventory';

export function fetchStores(token: string | null) {
  return apiFetch<StoresResponse>('/api/stores', {}, token);
}

export function fetchStockLevels(storeId: string, token: string | null) {
  return apiFetch<StockLevelsResponse>(
    `/api/stock-levels?store=${encodeURIComponent(storeId)}`,
    {},
    token,
  );
}

export function fetchStockHealth(storeId: string, token: string | null) {
  return apiFetch<StockHealthResponse>(
    `/api/stock-health?store=${encodeURIComponent(storeId)}`,
    {},
    token,
  );
}
