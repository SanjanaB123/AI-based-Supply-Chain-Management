"""
xgboost_model.py  —  pipeline version 2.0
Demand forecasting using XGBoost across all Store × Product series.

Unlike Prophet, XGBoost is trained as a single global model across all series.
series_id is encoded so the model can learn per-series patterns.
Walk-forward folds are used for robust evaluation before final test scoring.

Usage
-----
    python xgboost_model.py                        # uses data/splits/
    python xgboost_model.py --data-dir /my/splits  # custom path
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from datetime import datetime
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("Install xgboost: pip install xgboost") from e

try:
    import mlflow
    import mlflow.xgboost
    import mlflow.sklearn
except ImportError as e:
    raise ImportError("Install mlflow: pip install mlflow") from e

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Imports from scripts ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from data_splitting import encode_series

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_roll_std_7", "sales_ewm_28",
    "demand_forecast_lag1",
    "price_vs_competitor", "effective_price",
    "Holiday/Promotion", "Discount", "discount_x_holiday",
    "dow", "month", "is_weekend",
    "Inventory Level", "stockout_flag", "lead_time_demand",
    "Lead Time Days", "reorder_event",
    "Category_enc", "Region_enc", "Seasonality_enc",
    "y_pred_baseline",
    "series_enc",           # encoded series_id — added at runtime
]

# Columns NOT scaled (binary, categorical, ordinal)
NO_SCALE_COLS = [
    "Holiday/Promotion", "Discount", "discount_x_holiday",
    "dow", "month", "is_weekend", "stockout_flag", "reorder_event",
    "Category_enc", "Region_enc", "Seasonality_enc",
    "Lead Time Days", "series_enc",
]

TARGET_COL  = "y"
DATE_COL    = "as_of_date"
WEIGHT_COL  = "sample_weight"

# Default XGBoost hyperparameters (good starting point before Optuna tuning)
DEFAULT_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "mae",
    "max_depth":        6,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,       # L1 regularisation
    "reg_lambda":       1.0,       # L2 regularisation
    "random_state":     42,
    "n_jobs":           -1,
    "early_stopping_rounds": 30,
}

# ── MLflow Configuration ───────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", 
    "https://mlflow-833456981899.us-central1.run.app/"
)
MLFLOW_EXPERIMENT_NAME = "supply-chain-forecasting"

DEFAULT_DATA_DIR    = Path("data/splits")
DEFAULT_OUTPUT_DIR  = Path("data/models/xgboost")
DEFAULT_REPORTS_DIR = Path("reports")


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true > 0
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2   = r2_score(y_true, y_pred)

    metrics = {
        "mae":  round(float(mae),  4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 4),
        "r2":   round(float(r2),   4),
    }
    if label:
        log.info(
            "%s → MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
            label, mae, rmse, mape, r2,
        )
    return metrics

# ── Load best params ─────────────────────────────────────────────────────────────
def load_best_params(params_path: Path) -> dict:
    """Load best hyperparameters from Optuna tuning output."""
    with open(params_path) as f:
        params = json.load(f)
    log.info("Loaded best params from %s", params_path)
    for k, v in params.items():
        log.info("  %-25s = %s", k, v)
    return params

def get_X_y_w(
    df:           pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, sample_weights) for a split DataFrame."""
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("Missing feature columns (skipped): %s", missing)

    X = df[available].copy()
    y = df[TARGET_COL].copy()
    w = df[WEIGHT_COL].copy() if WEIGHT_COL in df.columns else pd.Series(
        np.ones(len(df)), index=df.index
    )
    return X, y, w


def scale(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame | None = None,
) -> tuple:
    """Fit StandardScaler on train only; transform val and test."""
    scale_cols = [c for c in X_train.columns if c not in NO_SCALE_COLS]
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val[scale_cols]   = scaler.transform(X_val[scale_cols])

    X_test_out = None
    if X_test is not None:
        X_test = X_test.copy()
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])
        X_test_out = X_test

    log.info("Scaled %d feature columns (fit on train only)", len(scale_cols))
    return X_train, X_val, X_test_out, scaler


# ── Walk-forward evaluation ───────────────────────────────────────────────────
def walk_forward_eval(
    train_df:   pd.DataFrame,
    n_splits:   int = 5,
    val_months: int = 2,
    gap_days:   int = 14,
    params:     dict = DEFAULT_PARAMS,
) -> list[dict]:
    """
    Run walk-forward evaluation on the train set.
    Returns a list of per-fold metric dicts.
    """
    train_df = train_df.copy()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    train_df = train_df.sort_values(DATE_COL)

    min_date        = train_df[DATE_COL].min()
    max_date        = train_df[DATE_COL].max()
    total_days      = (max_date - min_date).days
    val_window_days = val_months * 30
    step_days       = (total_days - val_window_days - gap_days) // n_splits

    fold_results = []

    for fold_idx in range(n_splits):
        train_end  = min_date + pd.Timedelta(days=step_days * (fold_idx + 1))
        val_start  = train_end + pd.Timedelta(days=gap_days)
        val_end    = val_start + pd.Timedelta(days=val_window_days)

        if val_end > max_date:
            val_end = max_date
        if val_start >= max_date:
            break

        fold_train = train_df[train_df[DATE_COL] <= train_end]
        fold_val   = train_df[
            (train_df[DATE_COL] > val_start) &
            (train_df[DATE_COL] <= val_end)
        ]

        if len(fold_train) == 0 or len(fold_val) == 0:
            continue

        # Encode series within the fold
        _, fold_train, fold_val = encode_series(fold_train, fold_val)

        X_tr, y_tr, w_tr = get_X_y_w(fold_train)
        X_vl, y_vl, _    = get_X_y_w(fold_val)

        X_tr_s, X_vl_s, _, _ = scale(X_tr, X_vl)

        # Fit — use val as eval set for early stopping
        fit_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        model = xgb.XGBRegressor(**fit_params, early_stopping_rounds=params.get("early_stopping_rounds", 30))
        model.fit(
            X_tr_s, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_vl_s, y_vl)],
            verbose=False,
        )

        y_pred = model.predict(X_vl_s).clip(min=0)
        metrics = compute_metrics(y_vl.values, y_pred,
                                  label=f"  Fold {fold_idx+1}")
        metrics["fold"] = fold_idx + 1
        metrics["n_train"] = len(fold_train)
        metrics["n_val"]   = len(fold_val)
        fold_results.append(metrics)

    # Summary across folds
    avg_mae  = np.mean([f["mae"]  for f in fold_results])
    avg_rmse = np.mean([f["rmse"] for f in fold_results])
    avg_mape = np.mean([f["mape"] for f in fold_results])
    log.info(
        "Walk-forward avg → MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%",
        avg_mae, avg_rmse, avg_mape,
    )
    return fold_results


# ── Final model training ──────────────────────────────────────────────────────
def train_final(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    params:   dict = DEFAULT_PARAMS,
) -> tuple[xgb.XGBRegressor, StandardScaler, dict, dict, dict]:
    """
    Train final model on full train set, evaluate on val and test.
    Returns (model, scaler, series_mapping, val_metrics, test_metrics).
    """
    series_mapping, train_df, val_df, test_df = encode_series(train_df, val_df, test_df)

    X_tr, y_tr, w_tr = get_X_y_w(train_df)
    X_vl, y_vl, _    = get_X_y_w(val_df)
    X_te, y_te, _    = get_X_y_w(test_df)

    X_tr_s, X_vl_s, X_te_s, scaler = scale(X_tr, X_vl, X_te)

    fit_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    model = xgb.XGBRegressor(**fit_params, early_stopping_rounds=params.get("early_stopping_rounds", 30))
    model.fit(
        X_tr_s, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_vl_s, y_vl)],
        verbose=100,
    )
    log.info("Best iteration: %d", model.best_iteration)

    val_pred  = model.predict(X_vl_s).clip(min=0)
    test_pred = model.predict(X_te_s).clip(min=0)

    val_metrics  = compute_metrics(y_vl.values,  val_pred,  label="XGBoost val")
    test_metrics = compute_metrics(y_te.values, test_pred, label="XGBoost test")

    return model, scaler, series_mapping, val_metrics, test_metrics


# ── Feature importance ────────────────────────────────────────────────────────
def get_feature_importance(
    model:        xgb.XGBRegressor,
    feature_cols: list[str],
) -> pd.DataFrame:
    scores = model.feature_importances_
    cols   = [c for c in feature_cols if c in model.feature_names_in_]
    fi = pd.DataFrame({"feature": cols, "importance": scores})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)

def load_best_params(params_path: Path) -> dict:
    """Load best hyperparameters from Optuna tuning output."""
    with open(params_path) as f:
        params = json.load(f)
    log.info("Loaded best params from %s", params_path)
    for k, v in params.items():
        log.info("  %-25s = %s", k, v)
    return params

# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_dir: Path, output_dir: Path, reports_dir: Path, params_path: Path | None = None) -> None:
    log.info("=== XGBoost Model Training  v2.0 ===")

    # Load hyperparameters
    if params_path and params_path.exists():
        params = load_best_params(params_path)
        log.info("Using Optuna best params")
    else:
        params = DEFAULT_PARAMS
        log.info("best_params.json not found — using DEFAULT_PARAMS")

    # Load splits
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df   = pd.read_parquet(data_dir / "val.parquet")
    test_df  = pd.read_parquet(data_dir / "test.parquet")
    log.info("Loaded  train=%d  val=%d  test=%d rows", len(train_df), len(val_df), len(test_df))

    # Training (no MLflow dependency)
    fold_results = walk_forward_eval(train_df, n_splits=5, params=params)
    model, scaler, series_mapping, val_metrics, test_metrics = train_final(train_df, val_df, test_df, params)
    fi_df = get_feature_importance(model, FEATURE_COLS)
    log.info("Top 10 features:\n%s", fi_df.head(10).to_string(index=False))

    # Save outputs locally (never depends on MLflow)
    output_dir.mkdir(parents=True, exist_ok=True)
    fi_path    = output_dir / "xgboost_feature_importance.csv"
    model_path = output_dir / "xgboost_model.json"
    scaler_path = output_dir / "scaler.pkl"
    
    fi_df.to_csv(fi_path, index=False)
    model.save_model(str(model_path))
    joblib.dump(scaler, scaler_path)
    mapping_path = output_dir / "series_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(series_mapping, f, indent=2)
    log.info("Model saved to %s", model_path)
    log.info("Scaler saved to %s", scaler_path)
    log.info("Series mapping saved to %s", mapping_path)

    # Upload preprocessing artifacts to GCS
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from artifact_io import save_preprocessing_artifacts
        gcs_uris = save_preprocessing_artifacts(scaler, scaler_path, series_mapping, output_dir)
        log.info("Preprocessing artifacts uploaded to GCS: %s", gcs_uris)
    except Exception as e:
        log.warning("GCS artifact upload failed — artifacts saved locally only. Error: %s", e)

    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model":            "xgboost",
        "pipeline_version": "2.0",
        "hyperparameters":  {k: v for k, v in params.items()},
        "best_iteration":   int(model.best_iteration),
        "n_features":       len(model.feature_names_in_),
        "walk_forward_folds":    fold_results,
        "walk_forward_avg_mae":  round(float(np.mean([f["mae"]  for f in fold_results])), 4),
        "walk_forward_avg_rmse": round(float(np.mean([f["rmse"] for f in fold_results])), 4),
        "walk_forward_avg_mape": round(float(np.mean([f["mape"] for f in fold_results])), 4),
        "val_metrics":      val_metrics,
        "test_metrics":     test_metrics,
        "top_features":     fi_df.head(10).to_dict("records"),
        "timestamp":        datetime.utcnow().isoformat() + "Z",
    }
    report_path = reports_dir / "xgboost_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved to %s", report_path)

    # MLflow logging (non-fatal — server may be unavailable)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="xgboost_supply_chain"):
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name="xgboost-supply-chain",
            )
            mlflow.log_params(params)
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("pipeline_version", "2.0")
            mlflow.log_param("n_features", len(model.feature_names_in_))
            mlflow.log_metrics({
                "val_mae":  val_metrics["mae"], "val_rmse":  val_metrics["rmse"],
                "val_mape": val_metrics["mape"], "val_r2":    val_metrics["r2"],
                "test_mae": test_metrics["mae"], "test_rmse": test_metrics["rmse"],
                "test_mape":test_metrics["mape"],"test_r2":   test_metrics["r2"],
                "walk_forward_avg_mae":  float(np.mean([f["mae"]  for f in fold_results])),
                "walk_forward_avg_rmse": float(np.mean([f["rmse"] for f in fold_results])),
                "walk_forward_avg_mape": float(np.mean([f["mape"] for f in fold_results])),
            })
            mlflow.log_artifact(str(model_path), "model_local")
            mlflow.log_artifact(str(fi_path),     "feature_importance")
            mlflow.log_artifact(str(report_path), "reports")
            mlflow.log_artifact(str(scaler_path), "preprocessing")
            mlflow.log_artifact(str(mapping_path), "preprocessing")
            mlflow.log_dict(fi_df.head(10).to_dict(), "top_features.json")
        log.info("MLflow logging complete.")
    except Exception as e:
        log.warning("MLflow logging failed — results saved locally. Error: %s", e)

    # Final summary
    log.info("=== XGBoost Results ===")
    log.info("Val  — MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
             val_metrics["mae"], val_metrics["rmse"], val_metrics["mape"], val_metrics["r2"])
    log.info("Test — MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
             test_metrics["mae"], test_metrics["rmse"], test_metrics["mape"], test_metrics["r2"])
    log.info("=== Done ===")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost demand forecasting model")
    parser.add_argument("--data-dir",    type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--params-path", type=Path, default=None)
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.reports_dir, args.params_path)