"""
data_pipeline.py
Airflow 3 DAG: Extract → Transform → Validate → Version → Load

Using TaskFlow API (@task) for improved stability and memory efficiency on Cloud Run.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, task
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.exceptions import AirflowException, AirflowSkipException

sys.path.insert(0, "/opt/airflow")

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
def _get_params():
    import yaml
    PARAMS_PATH = Path(os.getenv("PARAMS_PATH", "/opt/airflow/params.yaml"))
    return yaml.safe_load(open(PARAMS_PATH)) if PARAMS_PATH.exists() else {}

params = _get_params()
HORIZON = int(os.getenv("HORIZON", params.get("horizon", 1)))
ANOMALY_THRESHOLDS = params.get("anomaly_thresholds", {
    "z_score": 5.0,
    "iqr": 3.0,
    "missingness": 0.02,
    "date_gap_days": 1,
})

OUTPUT_BASE_PATH = Path(os.getenv("OUTPUT_BASE_PATH", params.get("output_base_path", "/opt/airflow/data")))
FEAT_DIR = OUTPUT_BASE_PATH / "features"

EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "admin@example.com").split(",")
GCS_BUCKET_NAME  = os.getenv("GCS_BUCKET_NAME", "supply-chain-pipeline")

MONGO_URI             = os.getenv("MONGO_URI")
MONGO_DB              = os.getenv("MONGO_DB", "inventory_forecasting")
MONGO_COLLECTION      = "retail_store_inventory"
MONGO_SNAP_COLLECTION = "inventory_snapshot"


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task
def extract_task():
    from scripts.extract import extract
    return extract(
        uri=MONGO_URI,
        db_name=MONGO_DB,
        collection_name=MONGO_COLLECTION,
        snap_collection_name=MONGO_SNAP_COLLECTION,
    )

@task
def transform_task(extract_results: dict):
    from scripts.transform import transform as run_fe, select_final_cols
    import pandas as pd

    raw_path      = str(extract_results.get("raw_path")).strip('"').strip("'")
    snapshot_path = str(extract_results.get("snapshot_path")).strip('"').strip("'")

    df      = pd.read_parquet(raw_path)
    snap    = pd.read_parquet(snapshot_path)
    df_feat = select_final_cols(run_fe(df, snap, horizon=HORIZON))

    log.info("Feature engineering complete — %d rows, %d columns", len(df_feat), len(df_feat.columns))

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEAT_DIR / "features.parquet"
    df_feat.to_parquet(out_path, index=False)
    log.info("Features written to %s", out_path)
    return str(out_path)


@task
def generate_schema_stats_task(features_path: str):
    from scripts.validate import generate_schema_and_stats
    features_path = str(features_path).strip('"').strip("'")
    outputs_dir   = str(FEAT_DIR / "validation_outputs")
    result        = generate_schema_and_stats(features_path, outputs_dir)
    log.info("Schema and stats generated at %s", result)
    return result


@task
def validate_schema_quality_task(outputs_dir: str):
    log.info("Validating schema quality using outputs in %s", outputs_dir)
    return str(outputs_dir).strip('"').strip("'")


@task
def detect_anomalies_task(features_path: str):
    from scripts.anomaly import generate_anomaly_report, check_anomaly_thresholds
    import json

    features_path = str(features_path).strip('"').strip("'")
    outputs_dir   = str(FEAT_DIR / "anomaly_outputs")
    report_path   = generate_anomaly_report(
        features_path=features_path,
        output_dir=outputs_dir,
        missingness_threshold=ANOMALY_THRESHOLDS.get("missingness", 0.02),
        outlier_z_threshold=ANOMALY_THRESHOLDS.get("z_score", 5.0),
        date_gap_threshold=ANOMALY_THRESHOLDS.get("date_gap_days", 1),
    )
    log.info("Anomaly detection completed. Report saved to %s", report_path)

    if not check_anomaly_thresholds(report_path, max_anomalies=5):
        with open(report_path) as f:
            summary = json.load(f)["summary"]
        raise AirflowException(
            f"Critical anomalies detected: {summary['total_anomaly_types']} total "
            f"({summary['missingness_anomalies']} missingness, "
            f"{summary['outlier_anomalies']} outliers, "
            f"{summary['date_gap_anomalies']} date gaps)"
        )
    return report_path


@task
def bias_slicing_report_task(features_path: str):
    from scripts.bias import generate_bias_report

    features_path = str(features_path).strip('"').strip("'")
    report_path   = generate_bias_report(
        features_path=features_path,
        output_dir=str(FEAT_DIR / "bias_outputs"),
        slice_features=[
            "Holiday/Promotion", "Seasonality_enc", "Category_enc",
            "Region_enc", "Store ID", "Product ID",
        ],
    )
    log.info("Bias analysis completed. Report saved to %s", report_path)
    return report_path


@task(retries=2, retry_delay=timedelta(seconds=30))
def drift_detect_task(features_path: str):
    print("BOOTSTRAP: detect_drift @task invoked.")
    from scripts.drift_detection import run_drift_detection
    import traceback

    try:
        features_path   = str(features_path).strip('"').strip("'")
        drift_threshold = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
        report_path, drift_score, drift_detected = run_drift_detection(
            features_path=features_path,
            output_dir=str(FEAT_DIR / "drift_outputs"),
            drift_threshold=drift_threshold,
        )
        log.info("Drift detection complete — score=%.3f, detected=%s", drift_score, drift_detected)
        return report_path
    except Exception as e:
        print(f"CRITICAL_TASK_ERROR (drift_detect): {e}")
        traceback.print_exc()
        raise


@task(retries=0)
def check_model_decay_task():
    from scripts.check_model_decay import check_model_decay as _check

    output_dir = str(FEAT_DIR / "decay_outputs")
    report_path, decay_detected = _check(output_dir=output_dir)
    log.info("Model decay check complete — decay_detected=%s, report=%s", decay_detected, report_path)
    if not decay_detected:
        raise AirflowSkipException("Model MAE within threshold — retraining not required.")
    return report_path


@task(trigger_rule="one_success", retries=0)
def retrain_trigger_task(drift_report_path: str):
    from scripts.trigger_retraining import trigger_retraining_if_drift

    drift_report_path = str(drift_report_path).strip('"').strip("'")
    triggered = trigger_retraining_if_drift(drift_report_path)
    if not triggered:
        raise AirflowSkipException("No drift detected — retraining not required.")
    log.info("Retraining triggered successfully.")
    return triggered


@task
def version_with_dvc_task(features_path: str):
    import subprocess
    from scripts.upload_to_gcp import upload_to_gcs # verify helper import

    features_path = str(features_path).strip('"').strip("'")
    dvc_root      = Path("/opt/airflow")
    dvc_config    = dvc_root / ".dvc" / "config"
    bucket_name   = os.getenv("GCS_BUCKET_NAME", "").strip()
    github_token  = os.getenv("GITHUB_TOKEN", "").strip()
    github_repo   = os.getenv("GITHUB_REPO", "SanjanaB123/AI-based-Supply-Chain-Management").strip()

    def run_cmd(cmd: list[str], env_extra: dict = None) -> str:
        env = os.environ.copy()
        env["PATH"] = "/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin"
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(cmd, cwd=str(dvc_root), text=True, capture_output=True, env=env)
        if result.returncode != 0:
            raise AirflowException(f"DVC Command failed: {' '.join(cmd)}\nstderr: {result.stderr}")
        return result.stdout.strip()

    if not dvc_config.exists():
        init_cmd = ["dvc", "init", "--no-scm", "-f"]
        run_cmd(init_cmd)

    run_cmd(["dvc", "add", features_path])

    if bucket_name:
        run_cmd(["dvc", "remote", "add", "-d", "storage", f"gs://{bucket_name}/dvc", "-f"])
        run_cmd(["dvc", "push"])

    if github_token:
        dvc_file = features_path + ".dvc"
        commit_msg = f"Update DVC pointer: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
        run_cmd(
            ["python3", "/opt/airflow/scripts/github_push.py", "push", commit_msg, dvc_file],
            env_extra={"GITHUB_TOKEN": github_token, "GITHUB_REPO": github_repo},
        )
    return features_path


@task
def load_task(features_path: str):
    from scripts.upload_to_gcp import upload_to_gcs
    import pandas as pd
    features_path = str(features_path).strip('"').strip("'")
    
    exec_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    prefix = f"monitoring/{exec_date}"

    # Upload main artifact
    upload_to_gcs(file_path=features_path, bucket_name=GCS_BUCKET_NAME, destination_blob_name="features/features.parquet")

    # Upload reports
    for rtype, rname in [("drift_outputs", "drift_report.json"), ("decay_outputs", "decay_report.json")]:
        report_path = FEAT_DIR / rtype / rname
        if report_path.exists():
            upload_to_gcs(file_path=str(report_path), bucket_name=GCS_BUCKET_NAME, destination_blob_name=f"{prefix}/{rname}")
    return features_path


# ── DAG Definition ────────────────────────────────────────────────────────────

@dag(
    dag_id="supply_chain_pipeline",
    description="Extract → Transform → Validate → Version → Load via TaskFlow API",
    start_date=datetime(2026, 2, 21, tzinfo=pendulum.timezone("America/New_York")),
    schedule="0 12 * * *",
    catchup=False,
    default_args={
        "owner": "airflow",
        "email_on_failure": False,
    },
    tags=["supply-chain", "airflow3", "taskflow"],
)
def supply_chain_pipeline():
    # 1. Pipeline execution using functional TaskFlow style
    raw_data_info = extract_task()
    feat_path = transform_task(raw_data_info)

    # 2. Parallel monitoring and validation
    validation_dir = generate_schema_stats_task(feat_path)
    schema_quality_path = validate_schema_quality_task(validation_dir)
    
    anomaly_path = detect_anomalies_task(feat_path)
    bias_path = bias_slicing_report_task(feat_path)
    
    # Drift and Decay
    drift_path = drift_detect_task(feat_path)
    decay_path = check_model_decay_task()

    # 3. Downstream dependencies
    # Retraining (depends on drift/decay)
    retrain_res = retrain_trigger_task(drift_path)
    
    # Versioning and Loading (depends on quality/anomaly)
    # Note: versioning and load run after basic validation
    versioned_path = version_with_dvc_task(feat_path)
    final_load = load_task(versioned_path)

    # 4. Email alerts (using traditional operators for simplicity)
    anomaly_alert = EmailOperator(
        task_id="anomaly_email_alert",
        to=EMAIL_RECIPIENTS,
        subject="Supply Chain Pipeline - Anomaly Alert",
        html_content=f"Anomaly Report: {anomaly_path}",
        trigger_rule="one_failed",
    )
    
    retrain_alert = EmailOperator(
        task_id="retrain_email_alert",
        to=EMAIL_RECIPIENTS,
        subject="Supply Chain Pipeline - Retraining Triggered",
        html_content=f"Retraining triggered on drift. Report: {drift_path}",
        trigger_rule="all_success",
    )

    # Cross-orchestration for alerts
    anomaly_path >> anomaly_alert
    retrain_res >> retrain_alert
    [schema_quality_path, anomaly_path] >> versioned_path >> final_load

# Instantiate DAG
dag = supply_chain_pipeline()
