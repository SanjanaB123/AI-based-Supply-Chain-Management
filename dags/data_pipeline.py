"""
supply_chain_pipeline DAG
─────────────────────────
Extracts raw inventory data from MongoDB, engineers features,
splits into train/val/test parquets, and writes metadata.json.

XCom carries only file paths (strings), not DataFrames.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from airflow.sdk import dag
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("/opt/airflow/data")
RAW_DIR    = DATA_ROOT / "raw"
FEAT_DIR   = DATA_ROOT / "features"
SPLIT_DIR  = DATA_ROOT / "processed"

# ── MongoDB defaults ──────────────────────────────────────────────────────────
MONGO_URI        = "mongodb://host.docker.internal:27017"
MONGO_DB         = "supply_chain"
MONGO_COLLECTION = "retail_store_inventory"

# ── Required source columns ───────────────────────────────────────────────────
REQUIRED_COLS = [
    "Date", "Store ID", "Product ID", "Units Sold",
    "Inventory Level", "Price", "Discount",
    "Holiday/Promotion", "Competitor Pricing",
    "Weather Condition", "Seasonality", "Units Ordered",
]

# ── Feature / metadata constants ──────────────────────────────────────────────
FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_ewm_28",
    "dow", "month",
    "Price", "Discount", "Holiday/Promotion",
    "Competitor Pricing", "Weather Condition", "Seasonality",
    "Inventory Level", "Units Ordered",
]
META_COLS       = ["as_of_date", "series_id", "horizon", "pipeline_version", "created_at"]
LABEL_COLS      = ["y"]
IDENTIFIER_COLS = ["Store ID", "Product ID"]
FINAL_COLS      = META_COLS + IDENTIFIER_COLS + FEATURE_COLS + LABEL_COLS

PIPELINE_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT
# ─────────────────────────────────────────────────────────────────────────────
def extract(
    uri: str = MONGO_URI,
    db_name: str = MONGO_DB,
    collection_name: str = MONGO_COLLECTION,
    **context,
) -> str:
    """Pull raw documents from MongoDB and write to Parquet. Returns file path."""
    log.info("Connecting to MongoDB at %s", uri)
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except ConnectionFailure as exc:
        raise RuntimeError(f"Cannot reach MongoDB: {exc}") from exc

    docs = list(client[db_name][collection_name].find({}, {"_id": 0}))
    client.close()

    if not docs:
        raise ValueError(f"Collection '{collection_name}' is empty or does not exist.")

    df = pd.DataFrame(docs)
    log.info("Extracted %d rows from '%s.%s'", len(df), db_name, collection_name)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    run_id  = context.get("run_id", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    out_path = RAW_DIR / f"raw_{run_id}.parquet"
    df.to_parquet(out_path, index=False)

    log.info("Raw data written to %s", out_path)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────
def transform(raw_path: str, horizon: int = 1, **context) -> str:
    """
    Feature engineering pipeline.
    Reads raw Parquet, returns path to feature Parquet.
    """
    df = pd.read_parquet(raw_path)
    log.info("Loaded %d rows from %s", len(df), raw_path)

    # ── 1. Data-quality checks ────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.drop_duplicates(subset=["Date", "Store ID", "Product ID"])
    dupes_removed = before - len(df)
    if dupes_removed:
        log.warning("Removed %d duplicate rows (Date, Store ID, Product ID)", dupes_removed)

    for col in ["Units Sold", "Inventory Level"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg:
                log.warning("Column '%s' has %d negative values — setting to 0", col, neg)
                df[col] = df[col].clip(lower=0)

    # ── 2. Sort ───────────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    # ── 3. Lag & rolling features (group = same product×store) ────────────────
    gs = df.groupby(["Store ID", "Product ID"])["Units Sold"]

    # Lag features (no leakage: shift > 0)
    df["sales_lag_1"]  = gs.shift(1)
    df["sales_lag_7"]  = gs.shift(7)
    df["sales_lag_14"] = gs.shift(14)

    # Rolling means — shift(1) prevents seeing today's sales;
    # min_periods chosen to avoid noisy early estimates without losing too many rows.
    df["sales_roll_mean_7"]  = gs.transform(
        lambda x: x.rolling(7,  min_periods=3).mean().shift(1)
    )
    df["sales_roll_mean_14"] = gs.transform(
        lambda x: x.rolling(14, min_periods=7).mean().shift(1)
    )
    df["sales_roll_mean_28"] = gs.transform(
        lambda x: x.rolling(28, min_periods=7).mean().shift(1)
    )

    # EWM monthly trend (no hard look-back loss, smoothed)
    df["sales_ewm_28"] = gs.transform(
        lambda x: x.shift(1).ewm(span=28, adjust=False).mean()
    )

    # ── 4. Calendar features ──────────────────────────────────────────────────
    df["dow"]   = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # ── 5. Label: next-step sales ─────────────────────────────────────────────
    df["y"] = gs.shift(-horizon)

    # ── 6. Drop rows with missing lag_14 or missing label ────────────────────
    df = df.dropna(subset=["sales_lag_14", "y"])

    # ── 7. MLOps metadata ────────────────────────────────────────────────────
    df["as_of_date"]        = df["Date"]
    df["series_id"]         = df["Store ID"] + "_" + df["Product ID"]
    df["horizon"]           = horizon
    df["pipeline_version"]  = PIPELINE_VERSION
    df["created_at"]        = datetime.utcnow().isoformat()

    df = df[FINAL_COLS].reset_index(drop=True)
    log.info("Feature engineering complete — %d rows, %d columns", len(df), len(df.columns))

    # ── 8. Write features ─────────────────────────────────────────────────────
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    run_id   = context.get("run_id", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    out_path = FEAT_DIR / f"features_{run_id}.parquet"
    df.to_parquet(out_path, index=False)

    log.info("Features written to %s", out_path)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load(features_path: str, **context) -> None:
    """
    Time-based train / val / test split and artifact export.

    Split boundaries (by as_of_date):
        train : oldest 70 %
        val   : next  15 %
        test  : newest 15 %

    Outputs (under data/processed/<run_id>/):
        train.parquet
        val.parquet
        test.parquet
        metadata.json
    """
    df = pd.read_parquet(features_path)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    dates = df["as_of_date"].sort_values().unique()
    n     = len(dates)

    train_end = dates[int(n * 0.70) - 1]
    val_end   = dates[int(n * 0.85) - 1]

    train_df = df[df["as_of_date"] <= train_end]
    val_df   = df[(df["as_of_date"] > train_end) & (df["as_of_date"] <= val_end)]
    test_df  = df[df["as_of_date"] > val_end]

    log.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    run_id   = context.get("run_id", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    out_dir  = SPLIT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir   / "val.parquet",   index=False)
    test_df.to_parquet(out_dir  / "test.parquet",  index=False)

    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "horizon":          int(df["horizon"].iloc[0]),
        "features":         FEATURE_COLS,
        "labels":           LABEL_COLS,
        "split_boundaries": {
            "train_end": str(train_end.date()),
            "val_end":   str(val_end.date()),
            "test_start": str((val_end + pd.Timedelta(days=1)).date()),
        },
        "row_counts": {
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
            "total": len(df),
        },
        "created_at": datetime.utcnow().isoformat(),
        "run_id":     run_id,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Artifacts written to %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    "owner":            "airflow",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="supply_chain_pipeline",
    description="Extract → feature-engineer → split inventory data for ML training",
    start_date=datetime(2026, 2, 21),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["supply-chain", "etl", "ml"],
)
def supply_chain_pipeline():
    extract_task = PythonOperator(
        task_id="extract",
        python_callable=extract,
        op_kwargs={
            "uri":             MONGO_URI,
            "db_name":         MONGO_DB,
            "collection_name": MONGO_COLLECTION,
        },
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=transform,
        op_kwargs={"raw_path": "{{ ti.xcom_pull(task_ids='extract') }}"},
    )

    load_task = PythonOperator(
        task_id="load",
        python_callable=load,
        op_kwargs={"features_path": "{{ ti.xcom_pull(task_ids='transform') }}"},
    )

    extract_task >> transform_task >> load_task


dag = supply_chain_pipeline()