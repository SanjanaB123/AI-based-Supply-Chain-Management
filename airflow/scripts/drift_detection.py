#!/usr/bin/env python3
"""
Drift detection for supply chain feature data.

Compares current feature distribution against a reference (training-period)
distribution using Evidently AI.

Reference  = earliest 80% of rows by as_of_date (approximates training data)
Current    = latest 20% of rows by as_of_date

Writes drift_report.json and returns (drift_score, drift_detected) tuple.
"""

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Tuple

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
MAX_SAMPLE_ROWS = 50_000  # cap to avoid OOM on Cloud Run

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


def _try_import_evidently():
    """Try multiple evidently import paths across versions."""
    # evidently >= 0.6
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        log.info("Loaded evidently via evidently.report / evidently.metric_preset")
        return Report, DataDriftPreset
    except ImportError:
        pass

    # evidently 0.4 / 0.5
    try:
        from evidently import Report
        from evidently.metric_preset import DataDriftPreset
        log.info("Loaded evidently via evidently / evidently.metric_preset")
        return Report, DataDriftPreset
    except ImportError:
        pass

    # evidently newer presets path
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        log.info("Loaded evidently via evidently / evidently.presets")
        return Report, DataDriftPreset
    except ImportError:
        pass

    raise ImportError(
        "Could not import evidently Report/DataDriftPreset. "
        "Check that evidently is installed in the Docker image."
    )


def run_drift_detection(
    features_path: str,
    output_dir: str,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> Tuple[str, float, bool]:
    """
    Run Evidently AI drift detection on features.parquet.

    Returns:
        (report_path, drift_score, drift_detected)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "drift_report.json"

    try:
        log.info("Loading features from %s", features_path)
        df = pd.read_parquet(features_path)
        log.info("Loaded %d rows, %d columns. Columns: %s", len(df), len(df.columns), list(df.columns))

        available_cols = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
        log.info("Matched feature columns: %s", available_cols)

        if not available_cols:
            log.warning("No expected feature columns found — writing no-drift report and skipping.")
            _write_no_drift_report(report_file, features_path, drift_threshold, reason="no_matching_columns")
            return str(report_file), 0.0, False

        reference, current = _split_reference_current(df)

        ref_features = reference[available_cols].reset_index(drop=True)
        cur_features = current[available_cols].reset_index(drop=True)

        # Sample to avoid OOM on Cloud Run
        if len(ref_features) > MAX_SAMPLE_ROWS:
            ref_features = ref_features.sample(MAX_SAMPLE_ROWS, random_state=42)
            log.info("Sampled reference to %d rows", MAX_SAMPLE_ROWS)
        if len(cur_features) > MAX_SAMPLE_ROWS:
            cur_features = cur_features.sample(MAX_SAMPLE_ROWS, random_state=42)
            log.info("Sampled current to %d rows", MAX_SAMPLE_ROWS)

        log.info("Importing evidently...")
        Report, DataDriftPreset = _try_import_evidently()

        log.info("Running Evidently DataDriftPreset on %d features", len(available_cols))
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features, current_data=cur_features)

        report_dict = report.as_dict()
        log.info("Evidently report generated successfully")

        drift_results = report_dict.get("metrics", [{}])[0].get("result", {})
        share_drifted = float(drift_results.get("share_of_drifted_columns", 0.0))
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

        with open(report_file, "w") as f:
            json.dump(output, f, indent=2, default=str)

        log.info(
            "Drift report saved — %.1f%% of features drifted (threshold %.1f%%), drift_detected=%s",
            share_drifted * 100,
            drift_threshold * 100,
            drift_detected,
        )
        return str(report_file), share_drifted, drift_detected

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("drift_detection FAILED: %s\n%s", exc, tb)
        # Write error to report file so retrain_trigger can still read it
        error_output = {
            "metadata": {"features_path": features_path, "generated_at": pd.Timestamp.now().isoformat()},
            "summary": {"drift_detected": False, "error": str(exc)},
            "traceback": tb,
            "per_column": {},
        }
        with open(report_file, "w") as f:
            json.dump(error_output, f, indent=2, default=str)
        raise


def _write_no_drift_report(report_file: Path, features_path: str, threshold: float, reason: str) -> None:
    output = {
        "metadata": {
            "features_path": features_path,
            "generated_at": pd.Timestamp.now().isoformat(),
            "drift_threshold": threshold,
        },
        "summary": {
            "share_of_drifted_columns": 0.0,
            "number_of_drifted_columns": 0,
            "total_columns": 0,
            "drift_detected": False,
            "reason": reason,
        },
        "per_column": {},
    }
    with open(report_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
