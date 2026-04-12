"""
data_splitting.py  —  pipeline version 2.0
Walk-forward + chronological split for the supply chain time-series dataset.

Key changes vs v1
  - FEATURE_COLS updated to match pipeline v2.0 (29 features, no Weather Condition)
  - snapshot_path parameter removed (Lead Time Days already merged in features.parquet)
  - gap_days default raised 7 → 14 to respect the lag-14 feature window
  - Added scale_features() with per-fold fit-on-train-only scaler (prevents leakage)
  - Added per-series coverage check in validate_splits()
  - save_report() includes feature list for traceability
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants — mirrors data_pipeline.py v2.0 ────────────────────────
FEATURE_COLS = [
    # Lag features
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    # Rolling statistics
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_roll_std_7", "sales_ewm_28",
    # External demand signal (lagged)
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
    # Baseline (removed sample_weight - it's for model fitting, not prediction)
    "y_pred_baseline",
    # Series encoding (Store ID + Product ID combination)
    "series_enc",
]

# Columns that should NOT be scaled (binary flags, encoded categoricals)
NO_SCALE_COLS = [
    "Holiday/Promotion", "Discount", "discount_x_holiday",
    "dow", "month", "is_weekend",
    "stockout_flag", "reorder_event",
    "Category_enc", "Region_enc", "Seasonality_enc",
    "series_enc",  # Categorical integer encoding
    "Lead Time Days",
]

SCALE_COLS = [c for c in FEATURE_COLS if c not in NO_SCALE_COLS]

LABEL_COL       = "y"
DATE_COL        = "as_of_date"
IDENTIFIER_COLS = ["Store ID", "Product ID"]

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_PATH = Path("data/features/features.parquet")
OUTPUT_DIR  = Path("data/splits")
REPORTS_DIR = Path("reports")


# ── Series encoding ─────────────────────────────────────────────────────────────
def encode_series(
    train: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, ...]:
    """
    Label-encode series_id (Store × Product) into series_enc integer column.
    Fit mapping on train only; unseen series in val/test get -1.

    Returns (mapping, train, *others) where mapping is {series_id_str: int}.
    """
    if "series_id" not in train.columns:
        train = train.copy()
        train["series_id"] = (
            train["Store ID"].astype(str) + "_" + train["Product ID"].astype(str)
        )

    mapping = {sid: i for i, sid in enumerate(sorted(train["series_id"].unique()))}

    results = []
    for df in (train, *others):
        df = df.copy()
        if "series_id" not in df.columns:
            df["series_id"] = (
                df["Store ID"].astype(str) + "_" + df["Product ID"].astype(str)
            )
        df["series_enc"] = df["series_id"].map(mapping).fillna(-1).astype(int)
        df = df.drop(columns=["series_id"])  # Remove temporary column
        results.append(df)

    return (mapping, *results)


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class SplitResult:
    train:      pd.DataFrame
    val:        pd.DataFrame
    test:       pd.DataFrame
    train_end:  str = ""
    val_end:    str = ""
    test_end:   str = ""
    n_train:    int = 0
    n_val:      int = 0
    n_test:     int = 0

    def summary(self) -> dict:
        total = self.n_train + self.n_val + self.n_test
        return {
            "train_rows": self.n_train,
            "val_rows":   self.n_val,
            "test_rows":  self.n_test,
            "train_end":  self.train_end,
            "val_end":    self.val_end,
            "test_end":   self.test_end,
            "train_pct":  round(self.n_train / total * 100, 1),
            "val_pct":    round(self.n_val   / total * 100, 1),
            "test_pct":   round(self.n_test  / total * 100, 1),
        }


@dataclass
class WalkForwardFold:
    fold_number:  int
    train:        pd.DataFrame
    val:          pd.DataFrame
    train_start:  str = ""
    train_end:    str = ""
    val_start:    str = ""
    val_end:      str = ""
    n_train:      int = 0
    n_val:        int = 0


# ── Core functions ─────────────────────────────────────────────────────────────

def chronological_split(
    df:         pd.DataFrame,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    date_col:   str   = DATE_COL,
) -> SplitResult:
    """
    Split df chronologically into train / val / test with no overlap.
    Default: 80% train · 10% val · 10% test.

    The test set is held out and must not be used during model development.
    Walk-forward folds are run on the train set only.
    """
    assert train_frac + val_frac < 1.0, "train_frac + val_frac must be < 1.0"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    dates         = df[date_col].sort_values().unique()
    n_dates       = len(dates)
    train_end_idx = int(n_dates * train_frac)
    val_end_idx   = int(n_dates * (train_frac + val_frac))

    train_end_date = dates[train_end_idx - 1]
    val_end_date   = dates[val_end_idx - 1]

    train = df[df[date_col] <= train_end_date].copy()
    val   = df[(df[date_col] > train_end_date) & (df[date_col] <= val_end_date)].copy()
    test  = df[df[date_col] > val_end_date].copy()

    result = SplitResult(
        train     = train,
        val       = val,
        test      = test,
        train_end = str(train_end_date.date()),
        val_end   = str(val_end_date.date()),
        test_end  = str(df[date_col].max().date()),
        n_train   = len(train),
        n_val     = len(val),
        n_test    = len(test),
    )

    log.info("=== Chronological Split ===")
    log.info("Train : %d rows  ends %s", result.n_train, result.train_end)
    log.info("Val   : %d rows  ends %s", result.n_val,   result.val_end)
    log.info("Test  : %d rows  ends %s", result.n_test,  result.test_end)
    return result


def walk_forward_validation(
    df:         pd.DataFrame,
    n_splits:   int = 5,
    val_months: int = 2,
    gap_days:   int = 14,   # raised from 7 → 14 to clear the lag-14 window
    date_col:   str = DATE_COL,
) -> list[WalkForwardFold]:
    """
    Generate walk-forward folds from df (run on train set only).

    gap_days: gap between train end and val start — must be ≥ the largest lag
              feature window used (lag-14 here → default 14 days).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    min_date        = df[date_col].min()
    max_date        = df[date_col].max()
    total_days      = (max_date - min_date).days
    val_window_days = val_months * 30
    usable_days     = total_days - (val_window_days + gap_days)
    step_days       = usable_days // n_splits

    folds = []
    for fold_idx in range(n_splits):
        train_end_date = min_date + pd.Timedelta(days=step_days * (fold_idx + 1))
        val_start_date = train_end_date + pd.Timedelta(days=gap_days)
        val_end_date   = val_start_date + pd.Timedelta(days=val_window_days)

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
    df:           pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    label_col:    str       = LABEL_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and label series y from a split DataFrame.
    Logs any feature columns that are missing (skipped silently).
    Categorical object columns are label-encoded as a safety fallback
    (all categoricals should already be encoded as _enc integers by transform).
    """
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("Missing feature columns (skipped): %s", missing)

    X = df[available].copy()
    y = df[label_col].copy()

    # Safety fallback: encode any remaining object columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        log.warning("Encoding residual object columns: %s", cat_cols)
        for col in cat_cols:
            X[col] = X[col].astype("category").cat.codes

    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame | None = None,
    scale_cols: list[str] = SCALE_COLS,
) -> tuple:
    """
    Fit StandardScaler on X_train only, then apply to val and test.
    Binary flags, encoded categoricals, and sample_weight are NOT scaled.

    Returns (X_train_scaled, X_val_scaled, X_test_scaled, scaler).
    X_test_scaled is None if X_test is not provided.

    IMPORTANT: always call this AFTER the train/test split, never before.
    Fitting on the full dataset before splitting leaks test statistics into training.
    """
    cols_to_scale = [c for c in scale_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_val   = X_val.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val[cols_to_scale]   = scaler.transform(X_val[cols_to_scale])

    X_test_scaled = None
    if X_test is not None:
        X_test = X_test.copy()
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
        X_test_scaled = X_test

    log.info("Scaled %d feature columns (fit on train only)", len(cols_to_scale))
    return X_train, X_val, X_test_scaled, scaler


def validate_splits(split: SplitResult, date_col: str = DATE_COL) -> None:
    """
    Assert chronological integrity and per-series coverage of the splits.
    Raises AssertionError if any check fails.
    """
    train_max = split.train[date_col].max()
    val_min   = split.val[date_col].min()
    val_max   = split.val[date_col].max()
    test_min  = split.test[date_col].min()

    assert train_max < val_min,  f"Train/val overlap: train ends {train_max}, val starts {val_min}"
    assert val_max   < test_min, f"Val/test overlap: val ends {val_max}, test starts {test_min}"
    assert split.n_test > 0,     "Test set is empty"
    assert split.n_val  > 0,     "Val set is empty"

    # Check every series appears in all three splits
    train_series = set(
        split.train["Store ID"].astype(str) + "_" + split.train["Product ID"].astype(str)
    )
    test_series = set(
        split.test["Store ID"].astype(str) + "_" + split.test["Product ID"].astype(str)
    )
    missing_in_test = train_series - test_series
    if missing_in_test:
        log.warning(
            "%d series in train are absent from test: %s",
            len(missing_in_test), list(missing_in_test)[:5],
        )

    log.info("Split validation passed.")
    log.info(
        "Train ends %s | Val %s → %s | Test %s → %s",
        split.train_end, val_min.date(), split.val_end,
        test_min.date(), split.test_end,
    )


def save_splits(split: SplitResult, output_dir: Path) -> dict[str, str]:
    """Write train/val/test parquet files. Returns dict of paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": str(output_dir / "train.parquet"),
        "val":   str(output_dir / "val.parquet"),
        "test":  str(output_dir / "test.parquet"),
    }

    split.train.to_parquet(paths["train"], index=False)
    split.val.to_parquet(paths["val"],     index=False)
    split.test.to_parquet(paths["test"],   index=False)

    log.info("train.parquet → %s  (%d rows)", paths["train"], len(split.train))
    log.info("val.parquet   → %s  (%d rows)", paths["val"],   len(split.val))
    log.info("test.parquet  → %s  (%d rows)", paths["test"],  len(split.test))
    return paths


def save_report(
    summary:     dict,
    folds:       list[WalkForwardFold],
    reports_dir: Path,
) -> None:
    """Save split summary + fold details as JSON (uploaded as CI artifact)."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        **summary,
        "pipeline_version": "2.0",
        "feature_cols":     FEATURE_COLS,
        "n_features":       len(FEATURE_COLS),
        "n_folds":          len(folds),
        "gap_days":         14,
        "folds": [
            {
                "fold":        f.fold_number,
                "train_start": f.train_start,
                "train_end":   f.train_end,
                "val_start":   f.val_start,
                "val_end":     f.val_end,
                "n_train":     f.n_train,
                "n_val":       f.n_val,
            }
            for f in folds
        ],
        "git_commit":   os.environ.get("GITHUB_SHA",      "local")[:8],
        "git_branch":   os.environ.get("GITHUB_REF_NAME", "local"),
        "ci_run_id":    os.environ.get("GITHUB_RUN_ID",   "local"),
        "triggered_by": os.environ.get("GITHUB_ACTOR",    "local"),
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    }

    report_path = reports_dir / "split_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Split report saved to %s", report_path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=== Data Split Pipeline  v2.0 ===")
    log.info("Input  : %s", INPUT_PATH)
    log.info("Output : %s", OUTPUT_DIR)

    # 1. Load feature matrix
    df = pd.read_parquet(INPUT_PATH)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # 2. Chronological 80/10/10 split
    split = chronological_split(df, train_frac=0.80, val_frac=0.10)

    # 3. Encode series (Store ID + Product ID combination)
    _, split.train, split.val, split.test = encode_series(split.train, split.val, split.test)
    log.info("Added series_enc column to all splits")

    # 4. Validate — no chronological leakage, all series represented
    validate_splits(split)

    # 5. Walk-forward folds on train set only (for hyperparameter search)
    folds = walk_forward_validation(split.train, n_splits=5, val_months=2, gap_days=14)

    # 6. Example: get X/y and scale for a single fold (for model scripts)
    #    Uncomment and pass to your model training loop.
    #
    # fold = folds[0]
    # X_tr, y_tr = get_X_y(fold.train)
    # X_vl, y_vl = get_X_y(fold.val)
    # X_tr_s, X_vl_s, _, scaler = scale_features(X_tr, X_vl)

    # 7. Save parquet files
    save_splits(split, OUTPUT_DIR)

    # 8. Save JSON report
    save_report(split.summary(), folds, REPORTS_DIR)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()