// ── Shared ─────────────────────────────────────────────────────────────────

export type StockStatus = 'critical' | 'low' | 'healthy';

// ── /api/stores ────────────────────────────────────────────────────────────

export interface StoresResponse {
  stores: string[];
}

// ── /api/stock-levels ──────────────────────────────────────────────────────

export interface StockLevelProduct {
  product_id: string;
  category: string;
  current_stock: number;
  stock_health: StockStatus;
}

export interface StockLevelSummary {
  critical: number;
  low: number;
  healthy: number;
}

export interface StockLevelsResponse {
  store: string;
  total_stock: number;
  summary: StockLevelSummary;
  products: StockLevelProduct[];
}

// ── /api/stock-health ──────────────────────────────────────────────────────

export interface StockHealthBreakdownItem {
  status: StockStatus;
  count: number;
  percentage: number;
}

export interface StockHealthResponse {
  store: string;
  total_products: number;
  breakdown: StockHealthBreakdownItem[];
}

// ── /api/sell-through ──────────────────────────────────────────────────────

export interface SellThroughProduct {
  product_id: string;
  category: string;
  sell_through_rate: number;
  total_sold: number;
  total_received: number;
}

export interface SellThroughResponse {
  store: string;
  products: SellThroughProduct[];
}

// ── /api/days-of-supply ────────────────────────────────────────────────────

export interface DaysOfSupplyProduct {
  product_id: string;
  category: string;
  days_of_supply: number;
  stock_health: StockStatus;
  current_stock: number;
  daily_sales: number;
}

export interface DaysOfSupplyThresholds {
  critical_below: number;
  low_below: number;
}

export interface DaysOfSupplyResponse {
  store: string;
  thresholds: DaysOfSupplyThresholds;
  products: DaysOfSupplyProduct[];
}

// ── /api/lead-time-risk ────────────────────────────────────────────────────

export interface LeadTimeRiskProduct {
  product_id: string;
  category: string;
  lead_time_days: number;
  days_of_supply: number;
  stock_health: StockStatus;
}

export interface LeadTimeRiskResponse {
  store: string;
  products: LeadTimeRiskProduct[];
}

// ── /api/shrinkage ─────────────────────────────────────────────────────────

export interface ShrinkageProduct {
  product_id: string;
  category: string;
  total_received: number;
  total_sold: number;
  current_stock: number;
  shrinkage: number;
}

export interface ShrinkageResponse {
  store: string;
  total_shrinkage: number;
  products: ShrinkageProduct[];
}
