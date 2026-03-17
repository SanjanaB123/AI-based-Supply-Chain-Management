"""
Backend API for AI-based Supply Chain Management Dashboard.

Provides 6 endpoints that serve inventory analytics data from
better_inventory_snapshot.csv for the frontend dashboard charts.
"""

import os
import math
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Supply Chain Dashboard API",
    description="Six analytics endpoints powering the inventory dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CSV_PATH = os.getenv(
    "INVENTORY_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "..", "better_inventory_snapshot.csv"),
)

PERIOD_DAYS_DEFAULT = 365  # assumed sales observation window


def _load_data() -> pd.DataFrame:
    """Read the snapshot CSV and normalise column names."""
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df.rename(
        columns={
            "Store ID": "store_id",
            "Product ID": "product_id",
            "Category": "category",
            "Region": "region",
            "Current Stock": "current_stock",
            "Total Units Sold": "total_sold",
            "Total Units Received": "total_received",
            "Lead Time Days": "lead_time_days",
        },
        inplace=True,
    )
    return df


DF = _load_data()

STORE_IDS = sorted(DF["store_id"].unique().tolist())


def _filter_store(store_id: Optional[str]) -> pd.DataFrame:
    if store_id is None:
        return DF.copy()
    sid = store_id.upper()
    if sid not in STORE_IDS:
        raise HTTPException(
            status_code=404,
            detail=f"Store '{store_id}' not found. Available: {STORE_IDS}",
        )
    return DF[DF["store_id"] == sid].copy()


def _health_label(days: float, critical: float, low: float) -> str:
    if math.isnan(days) or math.isinf(days):
        return "critical"
    if days < critical:
        return "critical"
    if days < low:
        return "low"
    return "healthy"


# ---------------------------------------------------------------------------
# Root / meta
# ---------------------------------------------------------------------------


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Supply Chain Dashboard API",
        "version": "1.0.0",
        "available_stores": STORE_IDS,
        "endpoints": [
            "/api/v1/stock-levels",
            "/api/v1/sell-through",
            "/api/v1/days-of-supply",
            "/api/v1/stock-health",
            "/api/v1/lead-time-risk",
            "/api/v1/shrinkage",
        ],
    }


# ---------------------------------------------------------------------------
# 1. Current stock by product  (horizontal bar chart)
# ---------------------------------------------------------------------------


@app.get("/api/v1/stock-levels", tags=["charts"])
def stock_levels(
    store_id: Optional[str] = Query(None, description="Filter by store, e.g. S001"),
):
    """
    Current units on hand per product, sorted low → high.

    Frontend usage: horizontal bar chart per store. Immediately shows which
    products are running low.
    """
    df = _filter_store(store_id)

    daily_rate = df["total_sold"] / PERIOD_DAYS_DEFAULT
    dos = df["current_stock"] / daily_rate.replace(0, float("nan"))

    records = (
        df.assign(
            daily_sales_rate=daily_rate.round(2),
            days_of_supply=dos.round(1),
            health=dos.apply(lambda d: _health_label(d, 15, 45)),
        )
        .sort_values("current_stock")
        .loc[
            :,
            [
                "store_id",
                "product_id",
                "category",
                "current_stock",
                "daily_sales_rate",
                "days_of_supply",
                "health",
            ],
        ]
    )
    return {
        "chart": "Current stock by product",
        "store_filter": store_id,
        "total_stock": int(df["current_stock"].sum()),
        "product_count": len(records),
        "data": records.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 2. Sell-through rate by product  (horizontal bar chart)
# ---------------------------------------------------------------------------


@app.get("/api/v1/sell-through", tags=["charts"])
def sell_through(
    store_id: Optional[str] = Query(None, description="Filter by store, e.g. S001"),
):
    """
    Sell-through rate = total_sold / total_received × 100.

    Shows which products move fast vs sit. Anything above 95 % is a fast
    mover that needs close watching.
    """
    df = _filter_store(store_id)

    rate = (df["total_sold"] / df["total_received"].replace(0, float("nan"))) * 100

    records = (
        df.assign(
            sell_through_pct=rate.round(2),
            is_fast_mover=rate >= 95,
        )
        .sort_values("sell_through_pct", ascending=False)
        .loc[
            :,
            [
                "store_id",
                "product_id",
                "category",
                "total_sold",
                "total_received",
                "sell_through_pct",
                "is_fast_mover",
            ],
        ]
    )

    # convert numpy bools to Python bools for JSON serialisation
    data = records.to_dict(orient="records")
    for row in data:
        row["is_fast_mover"] = bool(row["is_fast_mover"])

    return {
        "chart": "Sell-through rate by product",
        "store_filter": store_id,
        "avg_sell_through_pct": round(float(rate.mean()), 2),
        "fast_mover_count": int((rate >= 95).sum()),
        "data": data,
    }


# ---------------------------------------------------------------------------
# 3. Days of supply  (horizontal bar chart + colour threshold)
# ---------------------------------------------------------------------------


@app.get("/api/v1/days-of-supply", tags=["charts"])
def days_of_supply(
    store_id: Optional[str] = Query(None, description="Filter by store, e.g. S001"),
    period_days: int = Query(PERIOD_DAYS_DEFAULT, description="Sales observation window in days"),
    critical_threshold: float = Query(15, description="Days below which stock is critical"),
    low_threshold: float = Query(45, description="Days below which stock is low"),
):
    """
    Days of supply = current_stock / (total_sold / period_days).

    Tells a manager *how many days* of stock remain at the current sell rate.
    """
    df = _filter_store(store_id)

    daily_rate = df["total_sold"] / period_days
    dos = df["current_stock"] / daily_rate.replace(0, float("nan"))

    records = (
        df.assign(
            daily_sales_rate=daily_rate.round(2),
            days_of_supply=dos.round(1),
            health=dos.apply(lambda d: _health_label(d, critical_threshold, low_threshold)),
        )
        .sort_values("days_of_supply")
        .loc[
            :,
            [
                "store_id",
                "product_id",
                "category",
                "current_stock",
                "daily_sales_rate",
                "days_of_supply",
                "health",
            ],
        ]
    )
    return {
        "chart": "Days of supply",
        "store_filter": store_id,
        "period_days": period_days,
        "thresholds": {"critical_below": critical_threshold, "low_below": low_threshold},
        "data": records.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 4. Stock health breakdown  (donut chart)
# ---------------------------------------------------------------------------


@app.get("/api/v1/stock-health", tags=["charts"])
def stock_health(
    store_id: Optional[str] = Query(None, description="Filter by store, e.g. S001"),
    period_days: int = Query(PERIOD_DAYS_DEFAULT, description="Sales observation window in days"),
    critical_threshold: float = Query(15, description="Days below which stock is critical"),
    low_threshold: float = Query(45, description="Days below which stock is low"),
):
    """
    Bucket every product into Critical / Low / Healthy based on days of supply.

    Frontend usage: donut or stacked bar chart showing fleet-level health.
    """
    df = _filter_store(store_id)

    daily_rate = df["total_sold"] / period_days
    dos = df["current_stock"] / daily_rate.replace(0, float("nan"))
    labels = dos.apply(lambda d: _health_label(d, critical_threshold, low_threshold))

    counts = labels.value_counts().to_dict()
    breakdown = {
        "critical": counts.get("critical", 0),
        "low": counts.get("low", 0),
        "healthy": counts.get("healthy", 0),
    }

    return {
        "chart": "Stock health breakdown",
        "store_filter": store_id,
        "total_products": len(df),
        "total_stock": int(df["current_stock"].sum()),
        "thresholds": {"critical_below": critical_threshold, "low_below": low_threshold},
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# 5. Lead time vs days of supply  (scatter plot)
# ---------------------------------------------------------------------------


@app.get("/api/v1/lead-time-risk", tags=["charts"])
def lead_time_risk(
    store_id: Optional[str] = Query(None, description="Filter by store, e.g. S001"),
    period_days: int = Query(PERIOD_DAYS_DEFAULT, description="Sales observation window in days"),
):
    """
    Scatter data: x = lead_time_days, y = days_of_supply.

    Products in the bottom-right quadrant (long lead time, low stock) are
    the highest-risk items.
    """
    df = _filter_store(store_id)

    daily_rate = df["total_sold"] / period_days
    dos = df["current_stock"] / daily_rate.replace(0, float("nan"))

    records = (
        df.assign(
            days_of_supply=dos.round(1),
            risk_zone=(dos < 15) & (df["lead_time_days"] >= 5),
        )
        .loc[
            :,
            [
                "store_id",
                "product_id",
                "category",
                "lead_time_days",
                "days_of_supply",
                "current_stock",
                "risk_zone",
            ],
        ]
    )

    data = records.to_dict(orient="records")
    for row in data:
        row["risk_zone"] = bool(row["risk_zone"])

    return {
        "chart": "Lead time vs days of supply",
        "store_filter": store_id,
        "axis": {"x": "lead_time_days", "y": "days_of_supply"},
        "high_risk_count": sum(1 for r in data if r["risk_zone"]),
        "data": data,
    }


# ---------------------------------------------------------------------------
# 6. Shrinkage / loss indicator  (table)
# ---------------------------------------------------------------------------


@app.get("/api/v1/shrinkage", tags=["charts"])
def shrinkage(
    store_id: Optional[str] = Query(None, description="Filter by store, e.g. S001"),
):
    """
    Shrinkage = total_received − total_sold − current_stock.

    Positive values indicate unaccounted stock (damaged, stolen, expired).
    Returned sorted by shrinkage descending so the worst offenders appear first.
    """
    df = _filter_store(store_id)

    loss = df["total_received"] - df["total_sold"] - df["current_stock"]
    loss_pct = (loss / df["total_received"].replace(0, float("nan"))) * 100

    records = (
        df.assign(
            shrinkage_units=loss,
            shrinkage_pct=loss_pct.round(2),
            has_unaccounted_loss=loss > 0,
        )
        .sort_values("shrinkage_units", ascending=False)
        .loc[
            :,
            [
                "store_id",
                "product_id",
                "category",
                "total_received",
                "total_sold",
                "current_stock",
                "shrinkage_units",
                "shrinkage_pct",
                "has_unaccounted_loss",
            ],
        ]
    )

    data = records.to_dict(orient="records")
    for row in data:
        row["has_unaccounted_loss"] = bool(row["has_unaccounted_loss"])

    return {
        "chart": "Shrinkage / loss indicator",
        "store_filter": store_id,
        "total_shrinkage_units": int(loss.sum()),
        "products_with_loss": int((loss > 0).sum()),
        "data": data,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
