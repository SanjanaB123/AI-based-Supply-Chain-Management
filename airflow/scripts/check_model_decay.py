#!/usr/bin/env python3
"""
check_model_decay.py

Queries MLflow for the Production champion model's test_mae.
If MAE exceeds the configured threshold, the model is considered
decayed and retraining should be triggered.

Returns:
    Path to decay_report.json (str)
    True if decay detected, False otherwise

Environment variables:
    MLFLOW_TRACKING_URI  — MLflow server URL
    MAE_DECAY_THRESHOLD  — float, default 150.0
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://mlflow-952666479463.us-central1.run.app/",
)
MAE_DECAY_THRESHOLD = float(os.getenv("MAE_DECAY_THRESHOLD", "150.0"))

# Candidate model names — script checks both and uses whichever is in Production
MODEL_NAMES = ["xgboost-supply-chain", "prophet-supply-chain"]


def check_model_decay(output_dir: str) -> tuple[str, bool]:
    """
    Query MLflow for the Production model's test_mae.

    Args:
        output_dir: Directory to write decay_report.json.

    Returns:
        (report_path, decay_detected)
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "decay_report.json"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    champion_name = None
    champion_version = None
    current_mae = None

    # Diagnostic logging: list all versions for transparency
    log.info("Checking MLflow model versions...")
    for model_name in MODEL_NAMES:
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                log.info("Model '%s' versions found:", model_name)
                for v in versions:
                    log.info("  - v%s: stage=%s, run_id=%s", v.version, v.current_stage, v.run_id)
            else:
                log.info("Model '%s' has no registered versions.", model_name)
        except Exception as e:
            log.warning("Could not search versions for '%s': %s", model_name, e)

    # Find whichever model has a Production version
    for model_name in MODEL_NAMES:
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            if prod_versions:
                champion_name = model_name
                champion_version = prod_versions[0]
                log.info("Champion detected: %s (version %s in Production)", champion_name, champion_version.version)
                break
        except Exception as exc:
            log.warning("Could not query Production stage for '%s': %s", model_name, exc)

    if champion_version is None:
        log.warning("No Production model found in MLflow — skipping decay check.")
        report = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "mlflow_uri": MLFLOW_TRACKING_URI,
                "mae_threshold": MAE_DECAY_THRESHOLD,
            },
            "summary": {
                "decay_detected": False,
                "reason": "no_production_model_found",
                "champion_model": None,
                "champion_mae": None,
            },
        }
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        return str(report_file), False

    # Fetch test_mae from the run linked to this version
    run_id = champion_version.run_id
    try:
        run_data = client.get_run(run_id).data
        current_mae = run_data.metrics.get("test_mae")
        if current_mae is None:
            # Fallback: try plain "mae"
            current_mae = run_data.metrics.get("mae")
    except Exception as exc:
        log.error("Failed to fetch run metrics for run_id=%s: %s", run_id, exc)

    if current_mae is None:
        log.warning(
            "Production model '%s' v%s has no test_mae metric — skipping decay check.",
            champion_name, champion_version.version,
        )
        report = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "mlflow_uri": MLFLOW_TRACKING_URI,
                "mae_threshold": MAE_DECAY_THRESHOLD,
            },
            "summary": {
                "decay_detected": False,
                "reason": "no_mae_metric_logged",
                "champion_model": champion_name,
                "champion_version": champion_version.version,
                "champion_mae": None,
            },
        }
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        return str(report_file), False

    decay_detected = current_mae > MAE_DECAY_THRESHOLD

    log.info(
        "Model decay check — champion=%s v%s, MAE=%.4f, threshold=%.4f, decay=%s",
        champion_name, champion_version.version,
        current_mae, MAE_DECAY_THRESHOLD, decay_detected,
    )

    report = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "mae_threshold": MAE_DECAY_THRESHOLD,
        },
        "summary": {
            "decay_detected": decay_detected,
            "champion_model": champion_name,
            "champion_version": champion_version.version,
            "champion_mae": current_mae,
            "mae_threshold": MAE_DECAY_THRESHOLD,
            "reason": "mae_above_threshold" if decay_detected else "mae_within_threshold",
        },
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    return str(report_file), decay_detected
