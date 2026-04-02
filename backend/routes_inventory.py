from fastapi import APIRouter, Query, HTTPException
import pandas as pd
import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

router = APIRouter(prefix="/api", tags=["Inventory"])

log = logging.getLogger(__name__)

N_DAYS = 365  # period over which Total Units Sold is measured
CRITICAL_THRESHOLD = 14   # days of supply < 14 → critical
LOW_THRESHOLD = 45         # days of supply < 45 → low, else healthy

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "inventory_forecasting")
MONGO_COLLECTION = "inventory_snapshot"


def load_data_from_mongo():
    """Load inventory data from MongoDB."""
    if not MONGO_URI:
        raise ValueError("MONGO_URI environment variable not set")

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        log.info("Successfully connected to MongoDB")

        collection = client[MONGO_DB][MONGO_COLLECTION]
        docs = list(collection.find({}, {"_id": 0}))

        if not docs:
            raise ValueError(f"Collection '{MONGO_DB}.{MONGO_COLLECTION}' is empty or does not exist")

        df = pd.DataFrame(docs)
        log.info(f"Loaded {len(df)} rows from MongoDB")
        client.close()
        return df

    except ConnectionFailure as exc:
        raise RuntimeError(f"Cannot connect to MongoDB: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Error loading data from MongoDB: {exc}") from exc


df = load_data_from_mongo()
df.columns = df.columns.str.strip()

VALID_STORES = sorted(df["Store ID"].unique().tolist())


def refresh_data():
    """Refresh data from MongoDB."""
    global df, VALID_STORES
    df = load_data_from_mongo()
    df.columns = df.columns.str.strip()
    VALID_STORES = sorted(df["Store ID"].unique().tolist())
    return df


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


@router.get("/stores")
def get_stores():
    return {"stores": VALID_STORES}


@router.get("/stock-levels")
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


@router.get("/sell-through")
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

    return {"store": store, "products": products}


@router.get("/days-of-supply")
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
        "thresholds": {"critical_below": CRITICAL_THRESHOLD, "low_below": LOW_THRESHOLD},
        "products": products,
    }


@router.get("/stock-health")
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

    return {"store": store, "total_products": total, "breakdown": breakdown}


@router.get("/lead-time-risk")
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

    return {"store": store, "products": products}


@router.get("/shrinkage")
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
    return {"store": store, "total_shrinkage": total_shrinkage, "products": products}
