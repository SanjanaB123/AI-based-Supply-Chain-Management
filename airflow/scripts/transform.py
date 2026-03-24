"""
transform.py  —  pipeline version 2.0
Feature engineering pipeline — standalone module.

This file contains ONLY the preprocessing and feature engineering logic,
extracted from the notebook and data_pipeline.py so it can be:
  - imported by data_pipeline.py (Airflow DAG)
  - imported by prepare_local_data.py (local / CI testing)
  - unit tested independently

Entry point
-----------
    result = transform(df, snap, horizon=1)

    df   : raw retail DataFrame (from MongoDB or CSV)
    snap : inventory snapshot DataFrame (from MongoDB or CSV)
Returns a fully engineered DataFrame ready for splitting and training.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Schema constants ──────────────────────────────────────────────────────────
REQUIRED_COLS = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Units Sold", "Inventory Level", "Units Ordered",
    "Demand Forecast", "Price", "Discount",
    "Holiday/Promotion", "Competitor Pricing", "Seasonality",
]

SNAPSHOT_REQUIRED_COLS = ["Store ID", "Product ID", "Lead Time Days"]

FEATURE_COLS = [
    # Lag features
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    # Rolling statistics
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_roll_std_7", "sales_ewm_28",
    # External demand signal (lagged — no leakage)
    "demand_forecast_lag1",
    # Pricing
    "price_vs_competitor", "effective_price",
    # Promotional & calendar
    "Holiday/Promotion", "Discount", "discount_x_holiday",
    "dow", "month", "is_weekend",
    # Inventory position
    "Inventory Level", "stockout_flag", "lead_time_demand",
    # Supply
    "Lead Time Days", "reorder_event",
    # Encoded categoricals
    "Category_enc", "Region_enc", "Seasonality_enc",
    # Baseline & weight
    "y_pred_baseline", "sample_weight",
]

META_COLS       = ["as_of_date", "series_id", "horizon", "pipeline_version", "created_at"]
LABEL_COLS      = ["y"]
IDENTIFIER_COLS = ["Store ID", "Product ID"]
FINAL_COLS      = META_COLS + IDENTIFIER_COLS + FEATURE_COLS + LABEL_COLS

PIPELINE_VERSION = "2.0"


# ── Main transform function ───────────────────────────────────────────────────
def transform(
    df:      pd.DataFrame,
    snap:    pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Full preprocessing and feature engineering pipeline.

    Parameters
    ----------
    df      : raw retail time-series DataFrame
    snap    : inventory snapshot DataFrame (provides Lead Time Days)
    horizon : forecast horizon in days (default 1)

    Returns
    -------
    pd.DataFrame : feature-engineered DataFrame with FINAL_COLS schema
    """
    df   = df.copy()
    snap = snap.copy()

    log.info("Starting transform — input shape: %s", df.shape)

    # ── 1. Schema validation ──────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in retail data: {missing}")

    missing_snap = [c for c in SNAPSHOT_REQUIRED_COLS if c not in snap.columns]
    if missing_snap:
        raise ValueError(f"Missing required columns in snapshot: {missing_snap}")

    # ── 2. Deduplication & negative clipping ──────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["Date", "Store ID", "Product ID"])
    dupes = before - len(df)
    if dupes:
        log.warning("Removed %d duplicate rows", dupes)

    for col in ["Units Sold", "Inventory Level"]:
        neg = (df[col] < 0).sum()
        if neg:
            log.warning("Column '%s': %d negative values — clipped to 0", col, neg)
            df[col] = df[col].clip(lower=0)

    # ── 3. Parse dates & sort ─────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    # ── 4. Snapshot join (Lead Time Days) ─────────────────────────────────────
    df = df.merge(
        snap[["Store ID", "Product ID", "Lead Time Days"]],
        on=["Store ID", "Product ID"],
        how="left",
    )
    missing_lead = df["Lead Time Days"].isnull().sum()
    if missing_lead:
        log.warning(
            "%d rows missing Lead Time Days — filling with median", missing_lead
        )
        df["Lead Time Days"] = df["Lead Time Days"].fillna(
            df["Lead Time Days"].median()
        )

    # Add series_id early so it's available throughout
    df["series_id"] = (
        df["Store ID"].astype(str) + "_" + df["Product ID"].astype(str)
    )

    # ── 5. Lag features ───────────────────────────────────────────────────────
    gs = df.groupby(["Store ID", "Product ID"])["Units Sold"]

    df["sales_lag_1"]  = gs.shift(1)
    df["sales_lag_7"]  = gs.shift(7)
    df["sales_lag_14"] = gs.shift(14)
    df["sales_lag_28"] = gs.shift(28)

    # ── 6. Rolling statistics ─────────────────────────────────────────────────
    df["sales_roll_mean_7"]  = gs.transform(
        lambda x: x.rolling(7,  min_periods=3).mean().shift(1)
    )
    df["sales_roll_mean_14"] = gs.transform(
        lambda x: x.rolling(14, min_periods=7).mean().shift(1)
    )
    df["sales_roll_mean_28"] = gs.transform(
        lambda x: x.rolling(28, min_periods=7).mean().shift(1)
    )
    df["sales_roll_std_7"] = gs.transform(
        lambda x: x.rolling(7,  min_periods=3).std().shift(1)
    )
    df["sales_ewm_28"] = gs.transform(
        lambda x: x.shift(1).ewm(span=28, adjust=False).mean()
    )

    # ── 7. Demand forecast lag (prevent same-day leakage) ─────────────────────
    df["demand_forecast_lag1"] = (
        df.groupby(["Store ID", "Product ID"])["Demand Forecast"].shift(1)
    )

    # ── 8. Pricing features ───────────────────────────────────────────────────
    # Price & Competitor Pricing are 0.97 correlated — use ratio instead of both
    df["price_vs_competitor"] = (
        df["Price"] / df["Competitor Pricing"].clip(lower=0.01)
    )
    df["effective_price"] = df["Price"] * (1 - df["Discount"] / 100)

    # ── 9. Inventory position ─────────────────────────────────────────────────
    df["stockout_flag"]    = (df["Inventory Level"] == 0).astype(int)
    df["reorder_event"]    = (df["Units Ordered"] > 0).astype(int)
    df["lead_time_demand"] = (
        df["sales_roll_mean_7"].clip(lower=0) * df["Lead Time Days"]
    )

    # ── 10. Promotional & calendar features ───────────────────────────────────
    df["discount_x_holiday"] = df["Discount"] * df["Holiday/Promotion"]
    df["dow"]        = df["Date"].dt.dayofweek
    df["month"]      = df["Date"].dt.month
    df["is_weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)

    # ── 11. Categorical encoding ──────────────────────────────────────────────
    category_order = ["Groceries", "Snacks", "Beverages", "Household", "Personal Care"]
    df["Category_enc"] = pd.Categorical(
        df["Category"], categories=category_order
    ).codes

    region_map = {"North": 0, "South": 1, "East": 2, "West": 3, "Central": 4}
    df["Region_enc"] = (
        df["Region"].map(region_map).fillna(-1).astype(int)
    )

    # Ordinal: meaningful order for tree models & LSTM
    season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    df["Seasonality_enc"] = (
        df["Seasonality"].map(season_map).fillna(-1).astype(int)
    )

    # ── 12. Baseline prediction & sample weights ──────────────────────────────
    df["y_pred_baseline"] = (
        df["sales_lag_1"].fillna(df["sales_lag_7"])
        if horizon == 1
        else df["sales_lag_7"]
    )

    store_freq = df["Store ID"].value_counts(normalize=True)
    df["sample_weight"] = (
        df["Store ID"]
        .map(lambda x: 1.0 / max(store_freq.get(x, 0), 1e-6))
        .clip(0.1, 10.0)
    )

    # ── 13. Label ─────────────────────────────────────────────────────────────
    df["y"] = (
        df.groupby(["Store ID", "Product ID"])["Units Sold"].shift(-horizon)
    )

    # ── 14. Drop warm-up rows (missing lag_28) and tail (missing label) ───────
    rows_before = len(df)
    df = df.dropna(subset=["sales_lag_28", "y"]).reset_index(drop=True)
    rows_dropped = rows_before - len(df)
    log.info(
        "Dropped %d warm-up / label rows (%.1f%% of total)",
        rows_dropped,
        rows_dropped / rows_before * 100,
    )

    # ── 15. MLOps metadata ────────────────────────────────────────────────────
    df["as_of_date"]       = df["Date"]
    df["horizon"]          = horizon
    df["pipeline_version"] = PIPELINE_VERSION
    df["created_at"]       = datetime.utcnow().isoformat()

    # ── 16. Logging ───────────────────────────────────────────────────────────
    log.info("Unique series  : %d", df["series_id"].nunique())
    log.info(
        "Date range     : %s → %s",
        df["Date"].min().date(), df["Date"].max().date(),
    )
    log.info("Final shape    : %d rows × %d cols", len(df), len(df.columns))

    null_check = df[FEATURE_COLS].isnull().sum()
    bad_cols   = null_check[null_check > 0]
    if len(bad_cols):
        log.warning("Features with nulls after transform:\n%s", bad_cols)
    else:
        log.info("Feature null check: zero nulls across all %d features", len(FEATURE_COLS))

    return df


def select_final_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict DataFrame to FINAL_COLS schema.
    Call this after transform() when writing to Parquet.
    """
    available = [c for c in FINAL_COLS if c in df.columns]
    missing   = [c for c in FINAL_COLS if c not in df.columns]
    if missing:
        log.warning("FINAL_COLS missing from DataFrame (skipped): %s", missing)
    return df[available].reset_index(drop=True)