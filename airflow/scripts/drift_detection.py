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

# BOOTSTRAP: Immediate stdout print to verify script was called and loaded
print("BOOTSTRAP: drift_detection.py module initial load.")

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
    
    n = len(df_sorted)
    if n < 2:
        # Cannot split 1 row into 2 sets
        return df_sorted, pd.DataFrame(columns=df_sorted.columns)

    cutoff_idx = int(n * 0.8)
    
    # Ensure at least one row in current if n >= 2
    if cutoff_idx == n:
        cutoff_idx = n - 1
    elif cutoff_idx == 0:
        cutoff_idx = 1

    reference = df_sorted.iloc[:cutoff_idx]
    current = df_sorted.iloc[cutoff_idx:]
    log.info(
        "Reference rows: %d (up to %s), Current rows: %d (from %s)",
        len(reference),
        reference["as_of_date"].max().date() if not reference.empty else "N/A",
        len(current),
        current["as_of_date"].min().date() if not current.empty else "N/A",
    )
    return reference, current


def _try_import_evidently():
    """Try multiple evidently import paths across versions."""
    # evidently 0.4.x - 0.5.x — primary path
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        log.info("Loaded evidently via evidently.report / evidently.metric_preset (0.4/0.5)")
        return Report, DataDriftPreset
    except ImportError:
        pass

    # evidently 0.6+ presets path
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        log.info("Loaded evidently via evidently / evidently.presets (0.6+)")
        return Report, DataDriftPreset
    except ImportError:
        pass

    import subprocess, sys
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "evidently"],
            capture_output=True, text=True
        )
        log.error("evidently not found. pip show:\n%s", result.stdout or result.stderr)
    except Exception:
        pass

    raise ImportError(
        "Could not import evidently Report/DataDriftPreset. "
        "Check that evidently is installed in the Airflow environment."
    )



def run_drift_detection(
    features_path: str,
    output_dir: str,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> Tuple[str, float, bool]:
    """
    Run Evidently AI drift detection on features.parquet.
    """
    import pandas as pd
    from typing import Tuple
    import json
    import traceback
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "drift_report.json"

    try:
        log.info("Loading features from %s", features_path)
        
        # Optimization: Only read required columns to save memory
        load_cols = NUMERIC_FEATURE_COLS + ["as_of_date"]
        try:
            df = pd.read_parquet(features_path, columns=load_cols)
            log.info("Memory optimization: Read only %d relevant columns", len(load_cols))
        except Exception as e:
            log.warning("Filtered read failed, falling back to full read: %s", e)
            df = pd.read_parquet(features_path)
        
        # Diagnostic: Log memory usage
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"DRIFT_DETECTION_MEMORY_USAGE: {mem_mb:.2f} MB")
        log.info("Loaded %d rows. Memory depth: %.2f MB", len(df), mem_mb)

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

        if ref_features.empty or cur_features.empty:
            log.warning("Reference or Current dataset is empty after filtering — skipping drift detection.")
            _write_no_drift_report(report_file, features_path, drift_threshold, reason="empty_dataset_split")
            return str(report_file), 0.0, False

        log.info("Importing evidently...")
        Report, DataDriftPreset = _try_import_evidently()

        log.info("Instantiating Evidently Report for %d features", len(available_cols))
        # Handle both Metric and Preset style instantiation (defensive)
        try:
            report = Report(metrics=[DataDriftPreset()])
            log.info("Report initialized with metrics=[DataDriftPreset()]")
        except Exception:
            report = Report(presets=[DataDriftPreset()])
            log.info("Report initialized with presets=[DataDriftPreset()]")

        log.info("Running report.run()...")
        report.run(reference_data=ref_features, current_data=cur_features)

        report_dict = report.as_dict()
        log.info("Evidently report generated successfully. Parsing results...")

        # Robust result extraction (structure changed in 0.6+)
        drift_results = {}
        for metric in report_dict.get("metrics", []):
            if "DataDriftPreset" in metric.get("metric", "") or "DatasetDriftMetric" in metric.get("metric", ""):
                drift_results = metric.get("result", {})
                break
        
        # Fallback to direct check if loop didn't find it
        if not drift_results:
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
        # print() goes to stdout → Cloud Logging; log.error() does not in Cloud Run
        print(f"DRIFT_DETECTION_ERROR: {exc}")
        print(f"DRIFT_DETECTION_TRACEBACK:\n{tb}")
        log.error("drift_detection FAILED: %s\n%s", exc, tb)
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
