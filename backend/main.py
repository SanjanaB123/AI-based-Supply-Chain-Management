from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

app = FastAPI(title="Inventory Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "better_inventory_snapshot.csv")
N_DAYS = 365  # period over which Total Units Sold is measured

CRITICAL_THRESHOLD = 14   # days of supply < 14 → critical
LOW_THRESHOLD = 45         # days of supply < 45 → low, else healthy

df = pd.read_csv(DATA_PATH)
# strip any whitespace from column names
df.columns = df.columns.str.strip()

VALID_STORES = sorted(df["Store ID"].unique().tolist())


def _filter_store(store_id: str) -> pd.DataFrame:
    if store_id not in VALID_STORES:
        raise HTTPException(status_code=404, detail=f"Store '{store_id}' not found. Valid stores: {VALID_STORES}")
    return df[df["Store ID"] == store_id].copy()


def _compute_derived(frame: pd.DataFrame) -> pd.DataFrame:
    frame["daily_sales"] = frame["Total Units Sold"] / N_DAYS
    frame["days_of_supply"] = (frame["Current Stock"] / frame["daily_sales"]).round(2)
    frame["days_of_supply"] = frame["days_of_supply"].fillna(0)
    frame["sell_through_rate"] = ((frame["Total Units Sold"] / frame["Total Units Received"]) * 100).round(2)
    frame["shrinkage"] = frame["Total Units Received"] - frame["Total Units Sold"] - frame["Current Stock"]

    def health(dos):
        if dos < CRITICAL_THRESHOLD:
            return "critical"
        elif dos < LOW_THRESHOLD:
            return "low"
        return "healthy"

    frame["stock_health"] = frame["days_of_supply"].apply(health)
    return frame


# ──────────────────────────────────────────────
# GET /api/stores  — list all store IDs
# ──────────────────────────────────────────────
@app.get("/api/stores")
def get_stores():
    return {"stores": VALID_STORES}


# ──────────────────────────────────────────────
# 1. Stock Levels  — horizontal bar chart data
# ──────────────────────────────────────────────
@app.get("/api/stock-levels")
def stock_levels(store: str = Query(..., description="Store ID, e.g. S001")):
    frame = _compute_derived(_filter_store(store))
    frame = frame.sort_values("Current Stock")

    total_stock = int(frame["Current Stock"].sum())
    counts = frame["stock_health"].value_counts().to_dict()

    products = []
    for _, r in frame.iterrows():
        products.append({
            "product_id": r["Product ID"],
            "category": r["Category"],
            "current_stock": int(r["Current Stock"]),
            "stock_health": r["stock_health"],
        })

    return {
        "store": store,
        "total_stock": total_stock,
        "summary": {
            "critical": counts.get("critical", 0),
            "low": counts.get("low", 0),
            "healthy": counts.get("healthy", 0),
        },
        "products": products,
    }


# ──────────────────────────────────────────────
# 2. Sell-Through Rate  — horizontal bar chart
# ──────────────────────────────────────────────
@app.get("/api/sell-through")
def sell_through(store: str = Query(..., description="Store ID, e.g. S001")):
    frame = _compute_derived(_filter_store(store))
    frame = frame.sort_values("sell_through_rate", ascending=False)

    products = []
    for _, r in frame.iterrows():
        products.append({
            "product_id": r["Product ID"],
            "category": r["Category"],
            "sell_through_rate": float(r["sell_through_rate"]),
            "total_sold": int(r["Total Units Sold"]),
            "total_received": int(r["Total Units Received"]),
        })

    return {
        "store": store,
        "products": products,
    }


# ──────────────────────────────────────────────
# 3. Days of Supply  — bar chart + color threshold
# ──────────────────────────────────────────────
@app.get("/api/days-of-supply")
def days_of_supply(store: str = Query(..., description="Store ID, e.g. S001")):
    frame = _compute_derived(_filter_store(store))
    frame = frame.sort_values("days_of_supply")

    products = []
    for _, r in frame.iterrows():
        products.append({
            "product_id": r["Product ID"],
            "category": r["Category"],
            "days_of_supply": float(r["days_of_supply"]),
            "stock_health": r["stock_health"],
            "current_stock": int(r["Current Stock"]),
            "daily_sales": round(float(r["daily_sales"]), 2),
        })

    return {
        "store": store,
        "thresholds": {
            "critical_below": CRITICAL_THRESHOLD,
            "low_below": LOW_THRESHOLD,
        },
        "products": products,
    }


# ──────────────────────────────────────────────
# 4. Stock Health Breakdown  — donut chart
# ──────────────────────────────────────────────
@app.get("/api/stock-health")
def stock_health(store: str = Query(..., description="Store ID, e.g. S001")):
    frame = _compute_derived(_filter_store(store))
    counts = frame["stock_health"].value_counts().to_dict()
    total = len(frame)

    breakdown = []
    for status in ["critical", "low", "healthy"]:
        count = counts.get(status, 0)
        breakdown.append({
            "status": status,
            "count": count,
            "percentage": round((count / total) * 100, 1) if total else 0,
        })

    return {
        "store": store,
        "total_products": total,
        "breakdown": breakdown,
    }


# ──────────────────────────────────────────────
# 5. Lead Time vs Days of Supply  — scatter plot
# ──────────────────────────────────────────────
@app.get("/api/lead-time-risk")
def lead_time_risk(store: str = Query(..., description="Store ID, e.g. S001")):
    frame = _compute_derived(_filter_store(store))

    products = []
    for _, r in frame.iterrows():
        products.append({
            "product_id": r["Product ID"],
            "category": r["Category"],
            "lead_time_days": int(r["Lead Time Days"]),
            "days_of_supply": float(r["days_of_supply"]),
            "stock_health": r["stock_health"],
        })

    return {
        "store": store,
        "products": products,
    }


# ──────────────────────────────────────────────
# 6. Shrinkage / Loss  — table
# ──────────────────────────────────────────────
@app.get("/api/shrinkage")
def shrinkage(store: str = Query(..., description="Store ID, e.g. S001")):
    frame = _compute_derived(_filter_store(store))
    frame = frame.sort_values("shrinkage", ascending=False)

    products = []
    for _, r in frame.iterrows():
        products.append({
            "product_id": r["Product ID"],
            "category": r["Category"],
            "total_received": int(r["Total Units Received"]),
            "total_sold": int(r["Total Units Sold"]),
            "current_stock": int(r["Current Stock"]),
            "shrinkage": int(r["shrinkage"]),
        })

    total_shrinkage = int(frame["shrinkage"].sum())

    return {
        "store": store,
        "total_shrinkage": total_shrinkage,
        "products": products,
    }
