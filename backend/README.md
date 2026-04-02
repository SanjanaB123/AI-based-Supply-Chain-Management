# Inventory Dashboard Backend

FastAPI backend serving 6 endpoints for the inventory dashboard **with integrated AI chat capabilities**. Data source: **MongoDB Atlas** `inventory_forecasting.inventory_snapshot` collection (5 stores, 20 products each, 100 rows total).

## Quick Start

```bash
cd backend
pip install -r requirements.txt

# Set environment variables
export MONGO_URI="mongodb+srv://your-connection-string"
export MONGO_DB="inventory_forecasting"  # optional, defaults to this
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # for AI chat
export MCP_SERVER_URL="https://your-mcp-server-url/mcp"  # optional

uvicorn main:app --reload --port 8000
```

Docs at: `http://localhost:8000/docs`

---

## Features

### 📊 Inventory Analytics (6 Endpoints)
- Stock levels, sell-through rates, days of supply
- Stock health breakdown, lead time risk analysis
- Shrinkage tracking and store summaries

### 🤖 AI Chat Assistant (2 Endpoints)
- Natural language queries about inventory
- Powered by Claude Haiku + SetFit classification
- Connects to MCP server for real-time data access

---

## Computed Fields

| Field | Formula |
|---|---|
| `daily_sales` | `Total Units Sold / 365` |
| `days_of_supply` | `Current Stock / daily_sales` |
| `sell_through_rate` | `(Total Units Sold / Total Units Received) * 100` |
| `shrinkage` | `Total Units Received - Total Units Sold - Current Stock` |
| `stock_health` | `critical` if dos < 14, `low` if dos < 45, else `healthy` |

---

## Endpoints

### `GET /api/stores`

Returns list of all store IDs.

```json
{
  "stores": ["S001", "S002", "S003", "S004", "S005"]
}
```

---

### 1. `GET /api/stock-levels?store=S001`

**Panel:** Current stock by product (horizontal bar chart, sorted low to high)

```json
{
  "store": "S001",
  "total_stock": 60703,
  "summary": {
    "critical": 10,
    "low": 7,
    "healthy": 3
  },
  "products": [
    {
      "product_id": "P0008",
      "category": "Household",
      "current_stock": 222,
      "stock_health": "critical"
    },
    {
      "product_id": "P0010",
      "category": "Snacks",
      "current_stock": 335,
      "stock_health": "critical"
    }
  ]
}
```

**Frontend objects:**

```ts
interface StockLevelsResponse {
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
    stock_health: "critical" | "low" | "healthy";
  }[];
}
```

---

### 2. `GET /api/sell-through?store=S001`

**Panel:** Sell-through rate by product (horizontal bar chart, sorted high to low)

```json
{
  "store": "S001",
  "products": [
    {
      "product_id": "P0010",
      "category": "Snacks",
      "sell_through_rate": 99.91,
      "total_sold": 35818,
      "total_received": 35852
    },
    {
      "product_id": "P0007",
      "category": "Beverages",
      "sell_through_rate": 99.89,
      "total_sold": 71664,
      "total_received": 71745
    }
  ]
}
```

**Frontend objects:**

```ts
interface SellThroughResponse {
  store: string;
  products: {
    product_id: string;
    category: string;
    sell_through_rate: number;  // percentage 0-100
    total_sold: number;
    total_received: number;
  }[];
}
```

---

### 3. `GET /api/days-of-supply?store=S001`

**Panel:** Days of supply per product (horizontal bar chart + color thresholds)

```json
{
  "store": "S001",
  "thresholds": {
    "critical_below": 14,
    "low_below": 45
  },
  "products": [
    {
      "product_id": "P0010",
      "category": "Snacks",
      "days_of_supply": 3.41,
      "stock_health": "critical",
      "current_stock": 335,
      "daily_sales": 98.13
    },
    {
      "product_id": "P0007",
      "category": "Beverages",
      "days_of_supply": 3.48,
      "stock_health": "critical",
      "current_stock": 684,
      "daily_sales": 196.34
    }
  ]
}
```

**Frontend objects:**

```ts
interface DaysOfSupplyResponse {
  store: string;
  thresholds: {
    critical_below: number;
    low_below: number;
  };
  products: {
    product_id: string;
    category: string;
    days_of_supply: number;
    stock_health: "critical" | "low" | "healthy";
    current_stock: number;
    daily_sales: number;
  }[];
}
```

---

### 4. `GET /api/stock-health?store=S001`

**Panel:** Stock health breakdown (donut chart)

```json
{
  "store": "S001",
  "total_products": 20,
  "breakdown": [
    {
      "status": "critical",
      "count": 10,
      "percentage": 50.0
    },
    {
      "status": "low",
      "count": 7,
      "percentage": 35.0
    },
    {
      "status": "healthy",
      "count": 3,
      "percentage": 15.0
    }
  ]
}
```

**Frontend objects:**

```ts
interface StockHealthResponse {
  store: string;
  total_products: number;
  breakdown: {
    status: "critical" | "low" | "healthy";
    count: number;
    percentage: number;
  }[];
}
```

---

### 5. `GET /api/lead-time-risk?store=S001`

**Panel:** Lead time vs days of supply (scatter plot)

```json
{
  "store": "S001",
  "products": [
    {
      "product_id": "P0001",
      "category": "Groceries",
      "lead_time_days": 4,
      "days_of_supply": 26.7,
      "stock_health": "low"
    },
    {
      "product_id": "P0002",
      "category": "Beverages",
      "lead_time_days": 7,
      "days_of_supply": 62.04,
      "stock_health": "healthy"
    }
  ]
}
```

**Frontend objects:**

```ts
interface LeadTimeRiskResponse {
  store: string;
  products: {
    product_id: string;
    category: string;
    lead_time_days: number;   // x-axis
    days_of_supply: number;   // y-axis
    stock_health: "critical" | "low" | "healthy";
  }[];
}
```

---

### 6. `GET /api/shrinkage?store=S001`

**Panel:** Shrinkage / loss (table). Negative values mean stock gain (received > sold + on-hand), likely timing differences.

```json
{
  "store": "S001",
  "total_shrinkage": -9519,
  "products": [
    {
      "product_id": "P0019",
      "category": "Personal Care",
      "total_received": 20957,
      "total_sold": 20556,
      "current_stock": 496,
      "shrinkage": -95
    },
    {
      "product_id": "P0008",
      "category": "Household",
      "total_received": 22809,
      "total_sold": 22717,
      "current_stock": 222,
      "shrinkage": -130
    }
  ]
}
```

**Frontend objects:**

```ts
interface ShrinkageResponse {
  store: string;
  total_shrinkage: number;
  products: {
    product_id: string;
    category: string;
    total_received: number;
    total_sold: number;
    current_stock: number;
    shrinkage: number;  // positive = unaccounted loss, negative = gain
  }[];
}
```

---

## Color Mapping (for frontend reference)

| Status | Use for |
|---|---|
| `critical` | Red badge / red bar |
| `low` | Yellow/amber badge / amber bar |
| `healthy` | Green badge / green bar |

## Category Colors (from mock)

| Category | Color suggestion |
|---|---|
| Groceries | Blue |
| Beverages | Teal/Green |
| Household | Orange/Amber |
| Personal Care | Purple/Lavender |
| Snacks | Red/Coral |

---

## 🤖 AI Chat Endpoints

### `POST /api/chat`

**AI-powered chat interface for natural language inventory queries.**

```json
// Request
{
  "message": "What products need restocking at store S001?",
  "username": "user123"
}

// Response
{
  "response": "Based on current inventory levels, I found 3 products at store S001 that need restocking: P0008 (Household) with 222 units, P0010 (Snacks) with 335 units, and P0019 (Personal Care) with 496 units.",
  "agent": "ai-assistant"
}
```

### `GET /api/ai-status`

**Check if AI chat service is available.**

```json
{
  "status": "online",
  "ai_enabled": true,
  "model_loaded": true,
  "mcp_server": "https://your-mcp-server-url/mcp"
}
```

---

## Deployment Architecture

This backend is designed for **separate deployment** of components:

```
backend/
├── main.py              # Integrated FastAPI app (Inventory + AI Chat)
├── api/                 # AI Chat service (separate deployment)
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── mcp/                 # MCP Server (separate deployment)
│   ├── server.py
│   ├── Dockerfile
│   └── requirements.txt
└── training/            # ML Model training pipeline
```

**Development**: Run integrated `main.py` for full functionality  
**Production**: Deploy `api/` and `mcp/` separately for scalability
