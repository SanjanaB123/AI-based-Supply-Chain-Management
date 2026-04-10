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
