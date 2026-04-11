import { apiFetch } from './api';
import type {
  StoresResponse,
  StockLevelsResponse,
  StockHealthResponse,
  SellThroughResponse,
  DaysOfSupplyResponse,
  LeadTimeRiskResponse,
  ShrinkageResponse,
} from '../types/inventory';

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

export function fetchSellThrough(storeId: string, token: string | null) {
  return apiFetch<SellThroughResponse>(
    `/api/sell-through?store=${encodeURIComponent(storeId)}`,
    {},
    token,
  );
}

export function fetchDaysOfSupply(storeId: string, token: string | null) {
  return apiFetch<DaysOfSupplyResponse>(
    `/api/days-of-supply?store=${encodeURIComponent(storeId)}`,
    {},
    token,
  );
}

export function fetchLeadTimeRisk(storeId: string, token: string | null) {
  return apiFetch<LeadTimeRiskResponse>(
    `/api/lead-time-risk?store=${encodeURIComponent(storeId)}`,
    {},
    token,
  );
}

export function fetchShrinkage(storeId: string, token: string | null) {
  return apiFetch<ShrinkageResponse>(
    `/api/shrinkage?store=${encodeURIComponent(storeId)}`,
    {},
    token,
  );
}
