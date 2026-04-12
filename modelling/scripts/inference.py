#!/usr/bin/env python3
"""
inference.py
Model-agnostic inference using MLflow Production models.
Loads preprocessing artifacts (scaler + series mapping) from GCS.
"""

import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_io import load_preprocessing_artifacts

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://mlflow-952666479463.us-central1.run.app/",
)

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
    "series_enc",
]

NO_SCALE_COLS = [
    "Holiday/Promotion", "Discount", "discount_x_holiday",
    "dow", "month", "is_weekend", "stockout_flag", "reorder_event",
    "Category_enc", "Region_enc", "Seasonality_enc",
    "Lead Time Days", "series_enc",
]


def load_production_model():
    """Load the current production model from MLflow."""
    log.info("Loading Production Model")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    models_to_try = [
        "xgboost-supply-chain",
        "prophet-supply-chain",
    ]

    for model_name in models_to_try:
        try:
            log.info("Attempting to load: %s/Production", model_name)
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
            log.info("Successfully loaded %s from Production", model_name)
            return model, model_name
        except Exception as e:
            log.warning("Failed to load %s: %s", model_name, e)
            continue

    raise RuntimeError("No production model found in MLflow")


def load_preprocessing(local_dir: Path = Path("/tmp/preprocessing")):
    """Load scaler and series mapping from GCS."""
    scaler, series_mapping = load_preprocessing_artifacts(local_dir)
    return scaler, series_mapping


def apply_series_encoding(df: pd.DataFrame, series_mapping: dict) -> pd.DataFrame:
    """Compute series_enc from Store ID + Product ID using the saved mapping."""
    df = df.copy()
    if "Store ID" in df.columns and "Product ID" in df.columns:
        series_id = df["Store ID"].astype(str) + "_" + df["Product ID"].astype(str)
        df["series_enc"] = series_id.map(series_mapping).fillna(-1).astype(int)
    return df


def apply_scaling(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Scale numeric features using the saved scaler."""
    df = df.copy()
    scale_cols = [c for c in df.columns if c in FEATURE_COLS and c not in NO_SCALE_COLS]
    if scale_cols:
        df[scale_cols] = scaler.transform(df[scale_cols])
    return df


def predict_demand(model, input_data, model_name, scaler=None, series_mapping=None):
    """Make predictions using the loaded model with preprocessing."""
    if isinstance(input_data, (dict, list)):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data.copy()

    # Apply series encoding if mapping is available
    if series_mapping is not None:
        input_df = apply_series_encoding(input_df, series_mapping)

    # Select feature columns that exist in the input
    available_cols = [c for c in FEATURE_COLS if c in input_df.columns]
    input_df = input_df[available_cols]

    # Apply scaling for XGBoost models
    if scaler is not None and "xgboost" in model_name:
        input_df = apply_scaling(input_df, scaler)

    log.info("Input shape: %s, columns: %s", input_df.shape, list(input_df.columns))

    predictions = model.predict(input_df)

    # Ensure non-negative predictions
    if isinstance(predictions, np.ndarray):
        predictions = np.maximum(predictions, 0)
    elif hasattr(predictions, "clip"):
        predictions = predictions.clip(lower=0)

    log.info("Predictions: n=%d, sample=%s", len(predictions), predictions[:5])
    return predictions


def get_model_info(model_name):
    """Get information about the production model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            return None

        prod_version = prod_versions[0]
        run = client.get_run(prod_version.run_id)

        return {
            "model_name": model_name,
            "version": prod_version.version,
            "stage": prod_version.stage,
            "run_id": prod_version.run_id,
            "creation_time": run.info.start_time,
            "test_mae": run.data.metrics.get("test_mae"),
            "test_rmse": run.data.metrics.get("test_rmse"),
            "test_r2": run.data.metrics.get("test_r2"),
            "model_type": run.data.params.get("model_type"),
            "pipeline_version": run.data.params.get("pipeline_version"),
        }
    except Exception as e:
        log.error("Error getting model info: %s", e)
        return None


def main():
    """Main inference function."""
    try:
        # Load model
        model, model_name = load_production_model()

        # Load preprocessing artifacts from GCS
        try:
            scaler, series_mapping = load_preprocessing()
            log.info("Loaded preprocessing artifacts from GCS")
        except Exception as e:
            log.warning("Could not load preprocessing artifacts: %s", e)
            scaler, series_mapping = None, None

        # Model info
        model_info = get_model_info(model_name)
        if model_info:
            log.info("Model: %s v%s, Test MAE: %s",
                     model_info["model_name"], model_info["version"],
                     model_info.get("test_mae"))

        # Sample inference
        log.info("Sample Inference with %s", model_name)
        sample_data = {
            "sales_lag_1": [100, 150, 120],
            "sales_lag_7": [100, 150, 120],
            "sales_lag_14": [95, 145, 115],
            "sales_lag_28": [90, 140, 110],
            "sales_roll_mean_7": [98, 148, 118],
            "sales_roll_mean_14": [96, 146, 116],
            "sales_roll_mean_28": [92, 142, 112],
            "sales_roll_std_7": [5.0, 8.0, 4.0],
            "sales_ewm_28": [95, 143, 113],
            "demand_forecast_lag1": [100, 150, 120],
            "price_vs_competitor": [0.95, 1.05, 1.0],
            "effective_price": [10.0, 15.0, 12.0],
            "Holiday/Promotion": [0, 1, 0],
            "Discount": [0.1, 0.15, 0.05],
            "discount_x_holiday": [0.0, 0.15, 0.0],
            "dow": [1, 2, 3],
            "month": [1, 1, 1],
            "is_weekend": [0, 0, 0],
            "Inventory Level": [50, 75, 60],
            "stockout_flag": [0, 0, 0],
            "lead_time_demand": [700, 750, 600],
            "Lead Time Days": [7, 5, 10],
            "reorder_event": [0, 1, 0],
            "Category_enc": [1, 2, 1],
            "Region_enc": [1, 1, 2],
            "Seasonality_enc": [1, 1, 1],
            "y_pred_baseline": [100, 150, 120],
            "Store ID": ["S001", "S002", "S001"],
            "Product ID": ["P0001", "P0002", "P0003"],
        }

        predictions = predict_demand(model, sample_data, model_name, scaler, series_mapping)
        log.info("Sample predictions: %s", predictions)
        return predictions

    except Exception as e:
        log.error("Inference failed: %s", e)
        return None


if __name__ == "__main__":
    main()
