"""
prophet_model.py  —  pipeline version 2.0
Per-series demand forecasting using Facebook Prophet.

Prophet is trained independently for each Store × Product series.
Regressors from the feature matrix are added as additional signals.
Results are aggregated across all 100 series for final evaluation.

Usage
-----
    python prophet_model.py                        # uses data/splits/
    python prophet_model.py --data-dir /my/splits  # custom path
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

warnings.filterwarnings("ignore")

# Prophet import (inside module to avoid Airflow DAG parse issues)
try:
    from prophet import Prophet
except ImportError as e:
    raise ImportError("Install prophet: pip install prophet") from e

try:
    import mlflow
    import mlflow.prophet
except ImportError as e:
    raise ImportError("Install mlflow: pip install mlflow") from e

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Extra regressors passed to Prophet (must be present in the feature parquet)
REGRESSORS = [
    "Holiday/Promotion",
    "Discount",
    "price_vs_competitor",
    "effective_price",
    "discount_x_holiday",
    "Inventory Level",
    "stockout_flag",
    "lead_time_demand",
    "demand_forecast_lag1",
]

DATE_COL    = "as_of_date"
TARGET_COL  = "y"
SERIES_COLS = ["Store ID", "Product ID"]

# Default Prophet hyperparameters (used when best_params.json is not found)
DEFAULT_PARAMS = {
    "seasonality_mode":        "multiplicative",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "changepoint_range":       0.8,
    "n_changepoints":          25,
}

# Default paths
DEFAULT_DATA_DIR    = Path("data/splits")
DEFAULT_OUTPUT_DIR  = Path("data/models/prophet")
DEFAULT_REPORTS_DIR = Path("reports")

# ── MLflow Configuration ───────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", 
    "https://mlflow-952666479463.us-central1.run.app/",
)
MLFLOW_EXPERIMENT_NAME = "supply-chain-forecasting"


# ── Params loader ─────────────────────────────────────────────────────────────
def load_best_params(params_path: Path) -> dict:
    """Load best hyperparameters from Optuna tuning output."""
    with open(params_path) as f:
        params = json.load(f)
    log.info("Loaded best params from %s", params_path)
    for k, v in params.items():
        log.info("  %-30s = %s", k, v)
    return params


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute MAE, RMSE, MAPE, R² and a naive-baseline comparison."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true > 0          # avoid division by zero in MAPE
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


# ── Per-series training ───────────────────────────────────────────────────────
def train_prophet_series(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    regressors: list[str],
    series_id:  str,
    params:     dict = DEFAULT_PARAMS,
) -> tuple[Prophet, dict]:
    """
    Train one Prophet model for a single Store × Product series.

    Returns the fitted model and val metrics dict.
    """
    # Prophet expects columns: ds (datetime), y (float)
    def prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df[[DATE_COL, TARGET_COL] + [r for r in regressors if r in df.columns]].copy()
        out = out.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
        out["ds"] = pd.to_datetime(out["ds"])
        out["y"]  = out["y"].clip(lower=0)
        # Fill any regressor NaNs with 0 (warm-up rows already dropped upstream)
        for r in regressors:
            if r in out.columns:
                out[r] = out[r].fillna(0)
        return out

    train_prophet = prep(train_df)
    val_prophet   = prep(val_df)

    # ── Build model — use Optuna params if provided, else defaults ─────────────
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        seasonality_mode=        params.get("seasonality_mode",        DEFAULT_PARAMS["seasonality_mode"]),
        changepoint_prior_scale= params.get("changepoint_prior_scale", DEFAULT_PARAMS["changepoint_prior_scale"]),
        seasonality_prior_scale= params.get("seasonality_prior_scale", DEFAULT_PARAMS["seasonality_prior_scale"]),
        changepoint_range=       params.get("changepoint_range",       DEFAULT_PARAMS["changepoint_range"]),
        n_changepoints=          params.get("n_changepoints",          DEFAULT_PARAMS["n_changepoints"]),
    )

    # Add extra regressors
    for r in regressors:
        if r in train_prophet.columns:
            model.add_regressor(r)

    model.fit(train_prophet)

    # ── Predict on val ────────────────────────────────────────────────────────
    forecast = model.predict(val_prophet.drop(columns=["y"]))
    y_pred   = forecast["yhat"].clip(lower=0).values
    y_true   = val_prophet["y"].values

    metrics = compute_metrics(y_true, y_pred, label=f"  {series_id} val")
    return model, metrics


# ── Aggregate evaluation ──────────────────────────────────────────────────────
def evaluate_aggregate(
    all_true: list[float],
    all_pred: list[float],
    split_label: str,
) -> dict:
    """Compute metrics across all series combined."""
    metrics = compute_metrics(
        np.array(all_true),
        np.array(all_pred),
        label=f"Prophet {split_label} (all series)",
    )
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_dir: Path, output_dir: Path, reports_dir: Path, params_path: Path | None = None) -> None:
    log.info("=== Prophet Model Training  v2.0 ===")

    # ── Load hyperparameters (Optuna best params if available, else defaults) ──
    if params_path and params_path.exists():
        params = load_best_params(params_path)
        log.info("Using Optuna best params")
    else:
        params = DEFAULT_PARAMS
        log.info("best_params.json not found — using DEFAULT_PARAMS")

    # ── Load splits ───────────────────────────────────────────────────────────
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df   = pd.read_parquet(data_dir / "val.parquet")
    test_df  = pd.read_parquet(data_dir / "test.parquet")

    log.info(
        "Loaded  train=%d  val=%d  test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Identify series ───────────────────────────────────────────────────────
    series_list = (
        train_df[SERIES_COLS]
        .drop_duplicates()
        .sort_values(SERIES_COLS)
        .values.tolist()
    )
    log.info("Training Prophet on %d series", len(series_list))

    # ── Per-series training loop ──────────────────────────────────────────────
    models: dict[str, Prophet] = {}
    series_metrics: list[dict] = []

    val_all_true:  list[float] = []
    val_all_pred:  list[float] = []
    test_all_true: list[float] = []
    test_all_pred: list[float] = []

    for store_id, product_id in series_list:
        sid = f"{store_id}_{product_id}"

        tr = train_df[
            (train_df["Store ID"] == store_id) &
            (train_df["Product ID"] == product_id)
        ].copy()
        vl = val_df[
            (val_df["Store ID"] == store_id) &
            (val_df["Product ID"] == product_id)
        ].copy()
        te = test_df[
            (test_df["Store ID"] == store_id) &
            (test_df["Product ID"] == product_id)
        ].copy()

        if len(tr) < 30:
            log.warning("Series %s has only %d train rows — skipping", sid, len(tr))
            continue

        # Train + val eval
        model, val_metrics = train_prophet_series(tr, vl, REGRESSORS, sid, params)
        models[sid] = model

        # Test prediction (model already fitted — just predict)
        def prep_predict(df: pd.DataFrame) -> pd.DataFrame:
            out = df[[DATE_COL] + [r for r in REGRESSORS if r in df.columns]].copy()
            out = out.rename(columns={DATE_COL: "ds"})
            out["ds"] = pd.to_datetime(out["ds"])
            for r in REGRESSORS:
                if r in out.columns:
                    out[r] = out[r].fillna(0)
            return out

        test_forecast = model.predict(prep_predict(te))
        test_pred     = test_forecast["yhat"].clip(lower=0).values
        test_true     = te[TARGET_COL].values

        test_metrics = compute_metrics(test_true, test_pred)

        series_metrics.append({
            "series_id":    sid,
            "store_id":     store_id,
            "product_id":   product_id,
            "n_train":      len(tr),
            "val_mae":      val_metrics["mae"],
            "val_rmse":     val_metrics["rmse"],
            "val_mape":     val_metrics["mape"],
            "val_r2":       val_metrics["r2"],
            "test_mae":     test_metrics["mae"],
            "test_rmse":    test_metrics["rmse"],
            "test_mape":    test_metrics["mape"],
            "test_r2":      test_metrics["r2"],
        })

        val_all_true.extend(vl[TARGET_COL].tolist())
        val_all_pred.extend(
            model.predict(prep_predict(vl))["yhat"].clip(lower=0).tolist()
        )
        test_all_true.extend(test_true.tolist())
        test_all_pred.extend(test_pred.tolist())

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    val_agg  = evaluate_aggregate(val_all_true,  val_all_pred,  "validation")
    test_agg = evaluate_aggregate(test_all_true, test_all_pred, "test")

    # ── Save per-series metrics CSV ───────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(series_metrics)
    metrics_path = output_dir / "prophet_series_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Per-series metrics saved to %s", metrics_path)

    # ── Save summary report ───────────────────────────────────────────────────
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model":              "prophet",
        "pipeline_version":   "2.0",
        "hyperparameters":    params,
        "n_series":           len(models),
        "regressors":         REGRESSORS,
        "val_metrics":        val_agg,
        "test_metrics":       test_agg,
        "worst_val_series":   metrics_df.nlargest(3, "val_mae")[["series_id","val_mae"]].to_dict("records"),
        "best_val_series":    metrics_df.nsmallest(3, "val_mae")[["series_id","val_mae"]].to_dict("records"),
        "timestamp":          datetime.utcnow().isoformat() + "Z",
    }
    report_path = reports_dir / "prophet_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved to %s", report_path)

    # MLflow logging (non-fatal — server may be unavailable)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="prophet_supply_chain") as run:
            # Log a representative Prophet model (first series)
            if models:
                first_model = list(models.values())[0]
                mlflow.prophet.log_model(
                    first_model,
                    artifact_path="model",
                    registered_model_name="prophet-supply-chain"
                )
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "prophet")
            mlflow.log_param("pipeline_version", "2.0")
            mlflow.log_param("n_series", len(models))
            mlflow.log_param("regressors", str(REGRESSORS))
            
            # Log metrics
            mlflow.log_metrics({
                "val_mae": val_agg["mae"],
                "val_rmse": val_agg["rmse"],
                "val_mape": val_agg["mape"],
                "val_r2": val_agg["r2"],
                "test_mae": test_agg["mae"],
                "test_rmse": test_agg["rmse"],
                "test_mape": test_agg["mape"],
                "test_r2": test_agg["r2"],
            })
            
            # Log artifacts
            mlflow.log_artifact(str(metrics_path), "metrics")
            mlflow.log_artifact(str(report_path), "reports")
            
            # Log series metrics as a table
            mlflow.log_dict(metrics_df.to_dict(), "series_metrics.json")

        log.info("MLflow logging complete.")
    except Exception as e:
        log.warning("MLflow logging failed — results saved locally. Error: %s", e)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=== Prophet Results ===")
    log.info("Val  — MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
             val_agg["mae"], val_agg["rmse"], val_agg["mape"], val_agg["r2"])
    log.info("Test — MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
             test_agg["mae"], test_agg["rmse"], test_agg["mape"], test_agg["r2"])
    log.info("=== Done ===")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Prophet demand forecasting models")
    parser.add_argument("--data-dir",    type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--params-path", type=Path, default=None)
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.reports_dir, args.params_path)