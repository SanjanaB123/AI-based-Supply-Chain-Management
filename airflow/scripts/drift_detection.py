#!/usr/bin/env python3
"""
Drift detection for supply chain feature data.

Compares current feature distribution against a reference (training-period)
distribution using Evidently AI DataDriftPreset.

Reference  = earliest 80% of rows by as_of_date (approximates training data)
Current    = latest 20% of rows by as_of_date

Writes drift_report.json and returns (drift_score, drift_detected) tuple.
"""

import json
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

log = logging.getLogger(__name__)

DEFAULT_DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))

NUMERIC_FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_ewm_28",
    "dow", "month",
    "Price", "Discount",
    "Competitor Pricing",
    "Inventory Level", "Units Ordered",
    "y_pred_baseline",
]


def _split_reference_current(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by as_of_date: first 80% = reference, last 20% = current."""
    df = df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df_sorted = df.sort_values("as_of_date")
    cutoff_idx = int(len(df_sorted) * 0.8)
    reference = df_sorted.iloc[:cutoff_idx]
    current = df_sorted.iloc[cutoff_idx:]
    log.info(
        "Reference rows: %d (up to %s), Current rows: %d (from %s)",
        len(reference),
        reference["as_of_date"].max().date(),
        len(current),
        current["as_of_date"].min().date(),
    )
    return reference, current


def run_drift_detection(
    features_path: str,
    output_dir: str,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> Tuple[str, float, bool]:
    """
    Run Evidently AI drift detection on features.parquet.

    Args:
        features_path: Path to features parquet file.
        output_dir: Directory to write drift_report.json.
        drift_threshold: Fraction of drifted features that triggers retraining.

    Returns:
        (report_path, drift_score, drift_detected)
    """
    log.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)

    available_cols = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
    if not available_cols:
        raise ValueError("No expected feature columns found in the parquet file.")

    reference, current = _split_reference_current(df)

    ref_features = reference[available_cols].reset_index(drop=True)
    cur_features = current[available_cols].reset_index(drop=True)

    log.info("Running Evidently DataDriftPreset on %d features", len(available_cols))
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=cur_features)

    report_dict = report.as_dict()

    drift_results = report_dict.get("metrics", [{}])[0].get("result", {})
    share_drifted = drift_results.get("share_of_drifted_columns", 0.0)
    n_drifted = drift_results.get("number_of_drifted_columns", 0)
    n_total = drift_results.get("number_of_columns", len(available_cols))
    dataset_drift = drift_results.get("dataset_drift", False)

    drift_detected = share_drifted >= drift_threshold

    per_column = {}
    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "ColumnDriftMetric":
            col_name = metric.get("result", {}).get("column_name", "unknown")
            per_column[col_name] = {
                "drift_detected": metric["result"].get("drift_detected", False),
                "stattest": metric["result"].get("stattest", ""),
                "p_value": metric["result"].get("p_value"),
                "drift_score": metric["result"].get("drift_score"),
            }

    output = {
        "metadata": {
            "features_path": features_path,
            "generated_at": pd.Timestamp.now().isoformat(),
            "reference_rows": len(ref_features),
            "current_rows": len(cur_features),
            "features_checked": available_cols,
            "drift_threshold": drift_threshold,
        },
        "summary": {
            "share_of_drifted_columns": share_drifted,
            "number_of_drifted_columns": n_drifted,
            "total_columns": n_total,
            "dataset_drift_evidently": dataset_drift,
            "drift_detected": drift_detected,
        },
        "per_column": per_column,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "drift_report.json"

    with open(report_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info(
        "Drift report saved to %s — %.1f%% of features drifted (threshold %.1f%%), drift_detected=%s",
        report_file,
        share_drifted * 100,
        drift_threshold * 100,
        drift_detected,
    )

    return str(report_file), share_drifted, drift_detected
