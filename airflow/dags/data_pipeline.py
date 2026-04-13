"""
data_pipeline.py
Airflow DAG: Extract → Transform → Validate → Version → Load

All task logic lives in scripts/. This file only wires the DAG together.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pymongo

import pendulum
from airflow.sdk import dag
from airflow.operators.python import PythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.exceptions import AirflowException

sys.path.insert(0, "/opt/airflow")
from scripts.extract import extract
from scripts.upload_to_gcp import upload_to_gcs
from scripts.validate import generate_schema_and_stats

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
import yaml
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", "params.yaml"))
params      = yaml.safe_load(open(PARAMS_PATH)) if PARAMS_PATH.exists() else {}

HORIZON    = int(os.getenv("HORIZON", params.get("horizon", 1)))
ANOMALY_THRESHOLDS = params.get("anomaly_thresholds", {
    "z_score":       5.0,
    "iqr":           3.0,
    "missingness":   0.02,
    "date_gap_days": 1,
})

OUTPUT_BASE_PATH = Path(os.getenv("OUTPUT_BASE_PATH", params.get("output_base_path", "/opt/airflow/data")))
FEAT_DIR         = OUTPUT_BASE_PATH / "features"

EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "admin@example.com").split(",")
GCS_BUCKET_NAME  = os.getenv("GCS_BUCKET_NAME", "supply-chain-pipeline")

MONGO_URI             = os.getenv("MONGO_URI")
MONGO_DB              = os.getenv("MONGO_DB", "inventory_forecasting")
MONGO_COLLECTION      = "retail_store_inventory"
MONGO_SNAP_COLLECTION = "inventory_snapshot"


# ── Task functions ────────────────────────────────────────────────────────────

def transform(raw_path: str, snapshot_path: str, horizon: int = 1, **context) -> str:
    import pandas as pd
    from scripts.transform import transform as run_fe, select_final_cols

    raw_path      = str(raw_path).strip('"').strip("'")
    snapshot_path = str(snapshot_path).strip('"').strip("'")

    df      = pd.read_parquet(raw_path)
    snap    = pd.read_parquet(snapshot_path)
    df_feat = select_final_cols(run_fe(df, snap, horizon=horizon))

    log.info("Feature engineering complete — %d rows, %d columns", len(df_feat), len(df_feat.columns))

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEAT_DIR / "features.parquet"
    df_feat.to_parquet(out_path, index=False)
    log.info("Features written to %s", out_path)
    return str(out_path)


def generate_schema_stats(features_path: str, **context) -> str:
    features_path = str(features_path).strip('"').strip("'")
    outputs_dir   = str(FEAT_DIR / "validation_outputs")
    result        = generate_schema_and_stats(features_path, outputs_dir)
    log.info("Schema and stats generated at %s", result)
    return result


def validate_schema_quality(outputs_dir: str, **context) -> str:
    log.info("Validating schema quality using outputs in %s", outputs_dir)
    return str(outputs_dir).strip('"').strip("'")


def detect_anomalies(features_path: str, **context) -> str:
    from scripts.anomaly import generate_anomaly_report, check_anomaly_thresholds

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


def bias_slicing_report(features_path: str, **context) -> str:
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


def version_with_dvc(features_path: str, **context) -> str:
    import subprocess
    features_path = str(features_path).strip('"').strip("'")
    dvc_root      = Path("/opt/airflow")
    dvc_config    = dvc_root / ".dvc" / "config"
    bucket_name   = os.getenv("GCS_BUCKET_NAME", "").strip()
    github_token  = os.getenv("GITHUB_TOKEN", "").strip()
    github_repo   = os.getenv("GITHUB_REPO", "SanjanaB123/AI-based-Supply-Chain-Management").strip()

    log.info("=== version_with_dvc START ===")
    log.info("features_path=%s, bucket=%s, repo=%s, token_set=%s",
             features_path, bucket_name, github_repo, bool(github_token))
    log.info("features file exists: %s", Path(features_path).exists())
    log.info("dvc config exists: %s", dvc_config.exists())

    def run_cmd(cmd: list[str], env_extra: dict = None) -> str:
        log.info("Running command: %s", ' '.join(cmd))
        env = os.environ.copy()
        env["PATH"] = "/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin"
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(cmd, cwd=str(dvc_root), text=True, capture_output=True, env=env)
        if result.returncode != 0:
            log.error("Command FAILED: %s\nReturn code: %d\nSTDOUT:\n%s\nSTDERR:\n%s",
                      ' '.join(cmd), result.returncode, result.stdout, result.stderr)
            raise AirflowException(
                f"Command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        log.info("Command succeeded: %s", ' '.join(cmd))
        return result.stdout.strip()

    if not dvc_config.exists():
        init_cmd = ["dvc", "init", "--no-scm"]
        if (dvc_root / ".dvc").exists():
            init_cmd.append("-f")
        run_cmd(init_cmd)

    run_cmd(["dvc", "add", features_path])

    remotes = run_cmd(["dvc", "remote", "list"])
    if not remotes and bucket_name:
        run_cmd(["dvc", "remote", "add", "-d", "storage", f"gs://{bucket_name}/dvc"])
        remotes = run_cmd(["dvc", "remote", "list"])

    if remotes:
        run_cmd(["dvc", "push"])
        log.info("DVC: tracked and pushed %s", features_path)
    else:
        log.warning("DVC remote not configured — tracked locally only.")

    if not github_token:
        log.warning("GITHUB_TOKEN not set — skipping GitHub push, Actions will NOT trigger.")
        return features_path

    dvc_file      = features_path + ".dvc"
    gitignore     = str(Path(features_path).parent / ".gitignore")
    commit_msg    = f"Update DVC pointer: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
    files_to_push = [dvc_file]
    if Path(gitignore).exists():
        files_to_push.append(gitignore)

    run_cmd(
        ["python3", "/opt/airflow/scripts/github_push.py", "push", commit_msg] + files_to_push,
        env_extra={"GITHUB_TOKEN": github_token, "GITHUB_REPO": github_repo},
    )
    log.info("GitHub push complete — Actions workflow triggered.")
    return features_path


def load(features_path: str, bucket_name: str = GCS_BUCKET_NAME, **context) -> None:
    features_path = str(features_path).strip('"').strip("'")
    destination   = "features/features.parquet"
    log.info("Uploading %s → gs://%s/%s", features_path, bucket_name, destination)
    upload_to_gcs(file_path=features_path, bucket_name=bucket_name, destination_blob_name=destination)
    log.info("Upload complete: gs://%s/%s", bucket_name, destination)


# ── DAG ───────────────────────────────────────────────────────────────────────

default_args = {
    "owner":            "airflow",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="supply_chain_pipeline",
    description="Extract → Transform → Validate → Version → Load supply chain data",
    start_date=datetime(2026, 2, 21, tzinfo=pendulum.timezone("America/New_York")),
    schedule="0 12 * * *",
    catchup=False,
    default_args=default_args,
    tags=["supply-chain", "etl", "ml"],
)
def supply_chain_pipeline():

    extract_task = PythonOperator(
        task_id="extract",
        python_callable=extract,
        op_kwargs={
            "uri":                  MONGO_URI,
            "db_name":              MONGO_DB,
            "collection_name":      MONGO_COLLECTION,
            "snap_collection_name": MONGO_SNAP_COLLECTION,
        },
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=transform,
        op_kwargs={
            "raw_path":      "{{ ti.xcom_pull(task_ids='extract') }}",
            "snapshot_path": "{{ ti.xcom_pull(task_ids='extract', key='snapshot_path') }}",
            "horizon":       HORIZON,
        },
    )

    schema_stats_task = PythonOperator(
        task_id="generate_schema_stats",
        python_callable=generate_schema_stats,
        op_kwargs={"features_path": "{{ ti.xcom_pull(task_ids='transform') }}"},
    )

    validate_schema_task = PythonOperator(
        task_id="validate_schema_quality",
        python_callable=validate_schema_quality,
        op_kwargs={"outputs_dir": "{{ ti.xcom_pull(task_ids='generate_schema_stats') }}"},
    )

    anomaly_detect_task = PythonOperator(
        task_id="detect_anomalies",
        python_callable=detect_anomalies,
        op_kwargs={"features_path": "{{ ti.xcom_pull(task_ids='transform') }}"},
    )

    anomaly_email_alert = EmailOperator(
        task_id="anomaly_email_alert",
        to=EMAIL_RECIPIENTS,
        subject="Supply Chain Pipeline - Anomaly Alert",
        html_content="""
        <h2>Anomaly Detection Alert</h2>
        <p>Critical anomalies detected in the supply chain pipeline.</p>
        <ul>
            <li>Pipeline: {{ dag.dag_id }}</li>
            <li>Execution Date: {{ ds }}</li>
            <li>Anomaly Report: {{ ti.xcom_pull(task_ids='detect_anomalies') }}</li>
        </ul>
        <p><a href="{{ conf.get('webserver', 'base_url') }}">Airflow Dashboard</a></p>
        """,
        trigger_rule="one_failed",
    )

    bias_report_task = PythonOperator(
        task_id="bias_slicing_report",
        python_callable=bias_slicing_report,
        op_kwargs={"features_path": "{{ ti.xcom_pull(task_ids='transform') }}"},
    )

    dvc_version_task = PythonOperator(
        task_id="version_with_dvc",
        python_callable=version_with_dvc,
        op_kwargs={"features_path": "{{ ti.xcom_pull(task_ids='transform') }}"},
    )

    load_task = PythonOperator(
        task_id="load",
        python_callable=load,
        op_kwargs={
            "features_path": "{{ ti.xcom_pull(task_ids='transform') }}",
            "bucket_name":   GCS_BUCKET_NAME,
        },
    )

    # ── Task dependency graph ─────────────────────────────────────────────────
    #
    #   extract
    #     └── transform
    #           ├── schema_stats ──┐
    #           ├── anomaly ───────┴── validate_schema ── dvc_version ── load
    #           │       └── email_alert (on failure only)
    #           └── bias_report
    #
    extract_task >> transform_task
    transform_task >> [schema_stats_task, anomaly_detect_task, bias_report_task]
    [schema_stats_task, anomaly_detect_task] >> validate_schema_task
    validate_schema_task >> dvc_version_task >> load_task
    anomaly_detect_task >> anomaly_email_alert


dag = supply_chain_pipeline()
