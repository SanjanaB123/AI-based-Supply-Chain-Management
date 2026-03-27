export type ChatRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  payload?: unknown;
}

export type StockHealthStatus = "critical" | "low" | "healthy";

export interface StoresResponse {
  stores: string[];
}

export interface StockLevelsResponse {
  store: string;
  total_stock: number;
  summary: {
    critical: number;
    low: number;
    healthy: number;
  };
  products: {
    product_id: string;
    category: string;
    current_stock: number;
    stock_health: StockHealthStatus;
  }[];
}

export interface SellThroughResponse {
  store: string;
  products: {
    product_id: string;
    category: string;
    sell_through_rate: number;
    total_sold: number;
    total_received: number;
  }[];
}

export interface DaysOfSupplyResponse {
  store: string;
  thresholds: {
    critical_below: number;
    low_below: number;
  };
  products: {
    product_id: string;
    category: string;
    days_of_supply: number;
    stock_health: StockHealthStatus;
    current_stock: number;
    daily_sales: number;
  }[];
}

export interface StockHealthResponse {
  store: string;
  total_products: number;
  breakdown: {
    status: StockHealthStatus;
    count: number;
    percentage: number;
  }[];
}

export interface LeadTimeRiskResponse {
  store: string;
  products: {
    product_id: string;
    category: string;
    lead_time_days: number;
    days_of_supply: number;
    stock_health: StockHealthStatus;
  }[];
}

export interface ShrinkageResponse {
  store: string;
  total_shrinkage: number;
  products: {
    product_id: string;
    category: string;
    total_received: number;
    total_sold: number;
    current_stock: number;
    shrinkage: number;
  }[];
}
