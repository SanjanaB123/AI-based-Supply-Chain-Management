from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_ewm_28",
    "dow", "month", "is_weekend",
    "Price", "Discount", "Holiday/Promotion",
    "Competitor Pricing", "competitor_price_ratio",
    "Weather Condition", "Seasonality",
    "Category", "Region",
    "Inventory Level", "Units Ordered",
    "days_of_supply", "price_change",
    "sample_weight",
]

LABEL_COL       = "y"
DATE_COL        = "as_of_date"
IDENTIFIER_COLS = ["Store ID", "Product ID"]
EVAL_COLS       = ["y_pred_baseline"]


# ── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class SplitResult:
    """Holds train / validation / test DataFrames and metadata."""
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame
    train_end:   str = ""
    val_end:     str = ""
    test_end:    str = ""
    n_train:     int = 0
    n_val:       int = 0
    n_test:      int = 0

    def summary(self) -> dict:
        return {
            "train_rows":  self.n_train,
            "val_rows":    self.n_val,
            "test_rows":   self.n_test,
            "train_end":   self.train_end,
            "val_end":     self.val_end,
            "test_end":    self.test_end,
            "train_pct":   round(self.n_train / (self.n_train + self.n_val + self.n_test) * 100, 1),
            "val_pct":     round(self.n_val   / (self.n_train + self.n_val + self.n_test) * 100, 1),
            "test_pct":    round(self.n_test  / (self.n_train + self.n_val + self.n_test) * 100, 1),
        }


@dataclass
class WalkForwardFold:
    """Single fold from walk-forward validation."""
    fold_number:  int
    train:        pd.DataFrame
    val:          pd.DataFrame
    train_start:  str = ""
    train_end:    str = ""
    val_start:    str = ""
    val_end:      str = ""
    n_train:      int = 0
    n_val:        int = 0


# ── Core Splitting Functions ──────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    date_col:   str   = DATE_COL,
) -> SplitResult:
    # Splits a time-series DataFrame chronologically into train / val / test.

    assert train_frac + val_frac < 1.0, "train_frac + val_frac must be < 1.0"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Get unique sorted dates
    dates      = df[date_col].sort_values().unique()
    n_dates    = len(dates)

    train_end_idx = int(n_dates * train_frac)
    val_end_idx   = int(n_dates * (train_frac + val_frac))

    train_end_date = dates[train_end_idx - 1]
    val_end_date   = dates[val_end_idx - 1]

    train = df[df[date_col] <= train_end_date].copy()
    val   = df[(df[date_col] > train_end_date) & (df[date_col] <= val_end_date)].copy()
    test  = df[df[date_col] > val_end_date].copy()

    result = SplitResult(
        train      = train,
        val        = val,
        test       = test,
        train_end  = str(train_end_date.date()),
        val_end    = str(val_end_date.date()),
        test_end   = str(df[date_col].max().date()),
        n_train    = len(train),
        n_val      = len(val),
        n_test     = len(test),
    )

    log.info("=== Chronological Split ===")
    log.info("Train : %d rows  (ends %s)", result.n_train, result.train_end)
    log.info("Val   : %d rows  (ends %s)", result.n_val,   result.val_end)
    log.info("Test  : %d rows  (ends %s)", result.n_test,  result.test_end)

    return result


def walk_forward_validation(
    df:           pd.DataFrame,
    n_splits:     int = 5,
    val_months:   int = 2,
    gap_days:     int = 7,
    date_col:     str = DATE_COL,
) -> list[WalkForwardFold]:
    # Walk-Forward (expanding window) cross-validation for time series.

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    total_days = (max_date - min_date).days

    # Space fold cutoffs evenly across the training period
    # Reserve the last val_months*n_splits days for validation windows
    val_window_days = val_months * 30
    usable_days     = total_days - (val_window_days + gap_days)
    step_days       = usable_days // n_splits

    folds = []
    for fold_idx in range(n_splits):
        # Training window end
        train_end_date = min_date + pd.Timedelta(
            days=step_days * (fold_idx + 1)
        )
        # Validation window start (after gap)
        val_start_date = train_end_date + pd.Timedelta(days=gap_days)
        # Validation window end
        val_end_date   = val_start_date + pd.Timedelta(days=val_window_days)

        # Ensure we don't exceed data bounds
        if val_end_date > max_date:
            val_end_date = max_date
        if val_start_date >= max_date:
            break

        train_fold = df[df[date_col] <= train_end_date].copy()
        val_fold   = df[
            (df[date_col] > val_start_date) &
            (df[date_col] <= val_end_date)
        ].copy()

        if len(train_fold) == 0 or len(val_fold) == 0:
            continue

        fold = WalkForwardFold(
            fold_number = fold_idx + 1,
            train       = train_fold,
            val         = val_fold,
            train_start = str(min_date.date()),
            train_end   = str(train_end_date.date()),
            val_start   = str(val_start_date.date()),
            val_end     = str(val_end_date.date()),
            n_train     = len(train_fold),
            n_val       = len(val_fold),
        )
        folds.append(fold)

        log.info(
            "Fold %d | Train: %s → %s (%d rows) | Val: %s → %s (%d rows)",
            fold.fold_number,
            fold.train_start, fold.train_end, fold.n_train,
            fold.val_start,   fold.val_end,   fold.n_val,
        )

    return folds


def get_X_y(
    df:          pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    label_col:   str        = LABEL_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y from a split DataFrame.
    Encodes categorical columns automatically.

    Parameters
    ----------
    df           : train / val / test DataFrame
    feature_cols : list of feature column names
    label_col    : target column name

    Returns
    -------
    X : pd.DataFrame of features
    y : pd.Series of labels
    """
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("Missing feature columns (will be skipped): %s", missing)

    X = df[available].copy()
    y = df[label_col].copy()

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes

    return X, y


# ── Save Splits ───────────────────────────────────────────────────────────────

def save_splits(
    split:      SplitResult,
    output_dir: str,
    run_id:     str = "",
) -> dict[str, str]:
    """
    Saves train / val / test splits as Parquet files.

    Parameters
    ----------
    split      : SplitResult from chronological_split()
    output_dir : Directory to save files
    run_id     : Optional run identifier for file naming

    Returns
    -------
    Dict with paths to saved files
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    suffix = f"_{run_id}" if run_id else ""

    paths = {
        "train": str(out / f"train{suffix}.parquet"),
        "val":   str(out / f"val{suffix}.parquet"),
        "test":  str(out / f"test{suffix}.parquet"),
    }

    split.train.to_parquet(paths["train"], index=False)
    split.val.to_parquet(paths["val"],     index=False)
    split.test.to_parquet(paths["test"],   index=False)

    log.info("Splits saved to %s", output_dir)
    for name, path in paths.items():
        log.info("  %s → %s", name, path)

    return paths


# ── Airflow Task Callable ─────────────────────────────────────────────────────

def split_data(
    features_path: str,
    output_dir:    str  = "/opt/airflow/data/processed",
    train_frac:    float = 0.80,
    val_frac:      float = 0.10,
    n_splits:      int   = 5,
    **context,
) -> dict:
    """
    Airflow-callable function.
    Loads features parquet, runs chronological split + walk-forward validation,
    saves splits and returns paths via XCom.

    Parameters
    ----------
    features_path : Path to features parquet (from transform task via XCom)
    output_dir    : Where to save split files
    train_frac    : Training fraction (default 0.80)
    val_frac      : Validation fraction (default 0.10)
    n_splits      : Walk-forward folds (default 5)

    Returns
    -------
    Dict with split file paths and summary stats
    """
    log.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # ── Step 1: Chronological split ───────────────────────────────────────────
    log.info("Running chronological split (%.0f/%.0f/%.0f)...",
             train_frac * 100, val_frac * 100, (1 - train_frac - val_frac) * 100)
    split = chronological_split(df, train_frac=train_frac, val_frac=val_frac)

    # ── Step 2: Walk-forward validation on train set ──────────────────────────
    log.info("Running walk-forward validation with %d folds...", n_splits)
    folds = walk_forward_validation(split.train, n_splits=n_splits)
    log.info("Generated %d walk-forward folds", len(folds))

    # ── Step 3: Save splits ───────────────────────────────────────────────────
    run_id = context.get("run_id", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    paths  = save_splits(split, output_dir, run_id=run_id)

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    summary = split.summary()
    log.info("=== Split Summary ===")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)

    log.info("=== Walk-Forward Fold Summary ===")
    for fold in folds:
        log.info(
            "  Fold %d | Train: %s→%s (%d rows) | Val: %s→%s (%d rows)",
            fold.fold_number,
            fold.train_start, fold.train_end, fold.n_train,
            fold.val_start,   fold.val_end,   fold.n_val,
        )

    return {
        "train_path":   paths["train"],
        "val_path":     paths["val"],
        "test_path":    paths["test"],
        "split_summary": summary,
        "n_folds":      len(folds),
    }


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import glob
    files = glob.glob("/opt/airflow/data/features/features_*.parquet")
    if not files:
        print("No feature files found!")
    else:
        latest = sorted(files, key=os.path.getmtime)[-1]
        print(f"Testing with: {latest}")

        result = split_data(
            features_path=latest,
            output_dir="/opt/airflow/data/processed",
        )

        print("\n=== Results ===")
        for k, v in result.items():
            print(f"  {k}: {v}")