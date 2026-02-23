from __future__ import annotations
import sys
import yaml
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from scripts.validate import generate_schema_and_stats
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from airflow.sdk import dag
from airflow.operators.python import PythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.exceptions import AirflowException

log = logging.getLogger(__name__)


# ── Config loading ───────────────────────────────────────────────────────────
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", "params.yaml"))
if PARAMS_PATH.exists():
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)
else:
    params = {}

HORIZON = int(os.getenv("HORIZON", params.get("horizon", 1)))
LAGS = params.get("lags", [1, 7, 14])
ROLLING_WINDOWS = params.get("rolling_windows", [7, 14, 28])
ANOMALY_THRESHOLDS = params.get("anomaly_thresholds", {"z_score": 3.0, "iqr": 1.5, "missingness": 0.02, "date_gap_days": 1})
OUTPUT_BASE_PATH = Path(os.getenv("OUTPUT_BASE_PATH", params.get("output_base_path", "/opt/airflow/data")))

DATA_ROOT  = OUTPUT_BASE_PATH
RAW_DIR    = DATA_ROOT / "raw"
FEAT_DIR   = DATA_ROOT / "features"
SPLIT_DIR  = DATA_ROOT / "processed"

# ── Email configuration ───────────────────────────────────────────────────────────
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "admin@example.com").split(",")

# ── MongoDB defaults ──────────────────────────────────────────────────────────
MONGO_URI        = "mongodb://host.docker.internal:27017"
MONGO_DB         = "supply_chain"
MONGO_COLLECTION = "retail_store_inventory"

# ── Required source columns ───────────────────────────────────────────────────
REQUIRED_COLS = [
    "Date", "Store ID", "Product ID", "Units Sold",
    "Inventory Level", "Price", "Discount",
    "Holiday/Promotion", "Competitor Pricing",
    "Weather Condition", "Seasonality", "Units Ordered",
]

# ── Feature / metadata constants ──────────────────────────────────────────────
FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_ewm_28",
    "dow", "month",
    "Price", "Discount", "Holiday/Promotion",
    "Competitor Pricing", "Weather Condition", "Seasonality",
    "Inventory Level", "Units Ordered",
    "y_pred_baseline", "sample_weight"
]
META_COLS       = ["as_of_date", "series_id", "horizon", "pipeline_version", "created_at"]
LABEL_COLS      = ["y"]
IDENTIFIER_COLS = ["Store ID", "Product ID"]
FINAL_COLS      = META_COLS + IDENTIFIER_COLS + FEATURE_COLS + LABEL_COLS

PIPELINE_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT
# ─────────────────────────────────────────────────────────────────────────────
def extract(
    uri: str = MONGO_URI,
    db_name: str = MONGO_DB,
    collection_name: str = MONGO_COLLECTION,
    **context,
) -> str:
    """Pull raw documents from MongoDB and write to Parquet. Returns file path."""
    log.info("Connecting to MongoDB at %s", uri)
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except ConnectionFailure as exc:
        raise RuntimeError(f"Cannot reach MongoDB: {exc}") from exc

    docs = list(client[db_name][collection_name].find({}, {"_id": 0}))
    client.close()

    if not docs:
        raise ValueError(f"Collection '{collection_name}' is empty or does not exist.")

    df = pd.DataFrame(docs)
    log.info("Extracted %d rows from '%s.%s'", len(df), db_name, collection_name)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    run_id  = context.get("run_id", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    out_path = RAW_DIR / f"raw_{run_id}.parquet"
    df.to_parquet(out_path, index=False)

    log.info("Raw data written to %s", out_path)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────
def transform(raw_path: str, horizon: int = 1, **context) -> str:
    """
    Feature engineering pipeline.
    Reads raw Parquet, returns path to feature Parquet.
    """
    df = pd.read_parquet(raw_path)
    log.info("Loaded %d rows from %s", len(df), raw_path)

    # ── 1. Data-quality checks ────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.drop_duplicates(subset=["Date", "Store ID", "Product ID"])
    dupes_removed = before - len(df)
    if dupes_removed:
        log.warning("Removed %d duplicate rows (Date, Store ID, Product ID)", dupes_removed)

    for col in ["Units Sold", "Inventory Level"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg:
                log.warning("Column '%s' has %d negative values — setting to 0", col, neg)
                df[col] = df[col].clip(lower=0)

    # ── 2. Sort ───────────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    # ── 3. Lag & rolling features (group = same product×store) ────────────────
    gs = df.groupby(["Store ID", "Product ID"])["Units Sold"]

    # Lag features (no leakage: shift > 0)
    df["sales_lag_1"]  = gs.shift(1)
    df["sales_lag_7"]  = gs.shift(7)
    df["sales_lag_14"] = gs.shift(14)

    df["sales_roll_mean_7"]  = gs.transform(
        lambda x: x.rolling(7,  min_periods=3).mean().shift(1)
    )
    df["sales_roll_mean_14"] = gs.transform(
        lambda x: x.rolling(14, min_periods=7).mean().shift(1)
    )
    df["sales_roll_mean_28"] = gs.transform(
        lambda x: x.rolling(28, min_periods=7).mean().shift(1)
    )

    # EWM monthly trend (no hard look-back loss, smoothed)
    df["sales_ewm_28"] = gs.transform(
        lambda x: x.shift(1).ewm(span=28, adjust=False).mean()
    )

    # ── 4. Calendar features ──────────────────────────────────────────────────
    df["dow"]   = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # ── 5. Baseline prediction ─────────────────────────────────────────────────
    # Use lag_7 for horizon=1, lag_1 for horizon=1 as fallback, lag_7 for horizon=7
    if horizon == 1:
        df["y_pred_baseline"] = df["sales_lag_1"].fillna(df["sales_lag_7"])
    else:
        df["y_pred_baseline"] = df["sales_lag_7"]
    
    # ── 6. Sample weights for bias mitigation ───────────────────────────────────
    # Calculate inverse frequency weights by Store ID for bias mitigation
    store_freq = df["Store ID"].value_counts(normalize=True)
    df["sample_weight"] = df["Store ID"].map(lambda x: 1.0 / max(store_freq.get(x, 0), 1e-6))
    df["sample_weight"] = df["sample_weight"].clip(0.1, 10.0)  # Prevent extreme values

    # ── 7. Label: next-step sales ─────────────────────────────────────────────
    df["y"] = gs.shift(-horizon)

    # ── 8. Drop rows with missing lag_14 or missing label ────────────────────
    rows_before_drop = len(df)
    df = df.dropna(subset=["sales_lag_14", "y"])
    rows_dropped = rows_before_drop - len(df)
    if rows_dropped > 0:
        log.info("Dropped %d rows due to missing lags or labels (%.1f%%)", 
                rows_dropped, (rows_dropped / rows_before_drop) * 100)

    # ── 9. Enhanced logging ───────────────────────────────────────────────────
    log.info("Unique series: %d", df["series_id"].nunique())
    log.info("Date range: %s → %s", df["Date"].min(), df["Date"].max())
    
    # Log null percentages for key engineered features
    key_features = ["sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_roll_mean_7", 
                    "sales_roll_mean_14", "sales_roll_mean_28", "y_pred_baseline"]
    for feature in key_features:
        if feature in df.columns:
            null_pct = df[feature].isnull().mean() * 100
            log.info("Feature '%s' null percentage: %.2f%%", feature, null_pct)

    # ── 10. MLOps metadata ───────────────────────────────────────────────────
    df["as_of_date"]        = df["Date"]
    df["series_id"]         = df["Store ID"] + "_" + df["Product ID"]
    df["horizon"]           = horizon
    df["pipeline_version"]  = PIPELINE_VERSION
    df["created_at"]        = datetime.utcnow().isoformat()

    df = df[FINAL_COLS].reset_index(drop=True)
    log.info("Feature engineering complete — %d rows, %d columns", len(df), len(df.columns))

    # ── 11. Write features ─────────────────────────────────────────────────────
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    run_id   = context.get("run_id", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    out_path = FEAT_DIR / f"features_{run_id}.parquet"
    df.to_parquet(out_path, index=False)

    log.info("Features written to %s", out_path)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load(features_path: str, **context) -> None:
    """
    Loads the processed features parquet file. No train/val/test split is performed here.
    """
    df = pd.read_parquet(features_path)
    log.info("Loaded features from %s (rows: %d)", features_path, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    "owner":            "airflow",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="supply_chain_pipeline",
    description="Extract → feature-engineer → split inventory data for ML training",
    start_date=datetime(2026, 2, 21),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["supply-chain", "etl", "ml"],
)
def supply_chain_pipeline():

    extract_task = PythonOperator(
        task_id="extract",
        python_callable=extract,
        op_kwargs={
            "uri":             MONGO_URI,
            "db_name":         MONGO_DB,
            "collection_name": MONGO_COLLECTION,
        },
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=transform,
        op_kwargs={"raw_path": "{{ ti.xcom_pull(task_ids='extract') }}"},
    )


    def generate_schema_stats(features_path: str, **context):
        # Call the validate.py script to generate schema and stats
        sys.path.append(str(Path(__file__).parent.parent / "scripts"))
        outputs_dir = str(FEAT_DIR / "validation_outputs")
        result = generate_schema_and_stats(features_path, outputs_dir)
        log.info("Schema and stats generated at %s", result)
        return result

    def validate_schema_quality(features_path: str, **context):
        # Placeholder: implement schema validation
        log.info("Validating schema quality for %s", features_path)
        return features_path

    def detect_anomalies(features_path: str, **context):
        # Import and use the anomaly detection script
        sys.path.append(str(Path(__file__).parent.parent / "scripts"))
        from anomaly import generate_anomaly_report, check_anomaly_thresholds
        
        outputs_dir = str(FEAT_DIR / "anomaly_outputs")
        report_path = generate_anomaly_report(
            features_path=features_path,
            output_dir=outputs_dir,
            missingness_threshold=ANOMALY_THRESHOLDS.get("missingness", 0.02),
            outlier_z_threshold=ANOMALY_THRESHOLDS.get("z_score", 3.0),
            date_gap_threshold=ANOMALY_THRESHOLDS.get("date_gap_days", 1)
        )
        
        log.info("Anomaly detection completed. Report saved to %s", report_path)
        
        # Check if anomalies exceed acceptable threshold
        if not check_anomaly_thresholds(report_path, max_anomalies=0):
            # Load report to get details for error message
            import json
            with open(report_path, "r") as f:
                report = json.load(f)
            
            summary = report["summary"]
            error_msg = (
                f"Critical anomalies detected: "
                f"{summary['total_anomaly_types']} total "
                f"({summary['missingness_anomalies']} missingness, "
                f"{summary['outlier_anomalies']} outliers, "
                f"{summary['date_gap_anomalies']} date gaps)"
            )
            log.error(error_msg)
            raise AirflowException(error_msg)
        
        return report_path

    def version_with_dvc(features_path: str, **context):
        # Placeholder: implement DVC versioning
        log.info("Versioning with DVC: %s", features_path)
        return features_path

    def bias_slicing_report(features_path: str, **context):
        # Import and use the bias detection script
        sys.path.append(str(Path(__file__).parent.parent / "scripts"))
        from bias import generate_bias_report
        
        outputs_dir = str(FEAT_DIR / "bias_outputs")
        report_path = generate_bias_report(
            features_path=features_path,
            output_dir=outputs_dir,
            slice_features=["Holiday/Promotion", "Weather Condition", "Seasonality", "Store ID", "Product ID"]
        )
        log.info("Bias analysis completed. Report saved to %s", report_path)
        return report_path


    schema_stats_task = PythonOperator(
        task_id="generate_schema_stats",
        python_callable=generate_schema_stats,
        op_kwargs={"features_path": "{{ ti.xcom_pull(task_ids='transform') }}"},
    )

    anomaly_detect_task = PythonOperator(
        task_id="detect_anomalies",
        python_callable=detect_anomalies,
        op_kwargs={"features_path": "{{ ti.xcom_pull(task_ids='transform') }}"},
    )

    # Email alert task for anomaly failures
    anomaly_email_alert = EmailOperator(
        task_id="anomaly_email_alert",
        to=EMAIL_RECIPIENTS,
        subject="Supply Chain Pipeline - Anomaly Alert",
        html_content="""
        <h2>Anomaly Detection Alert</h2>
        <p>The supply chain data pipeline has detected critical anomalies in the processed data.</p>
        <p><strong>Details:</strong></p>
        <ul>
            <li>Pipeline: {{ dag.dag_id }}</li>
            <li>Execution Date: {{ ds }}</li>
            <li>Task: detect_anomalies</li>
            <li>Anomaly Report: {{ ti.xcom_pull(task_ids='detect_anomalies') }}</li>
        </ul>
        <p>Please review the anomaly report and take appropriate action.</p>
        <p>Check the Airflow UI for more details: <a href="{{ conf.get('webserver', 'base_url') }}">Airflow Dashboard</a></p>
        """,
        trigger_rule="one_failed",  # Only send when anomaly detection fails
    )


    validate_schema_task = PythonOperator(
        task_id="validate_schema_quality",
        python_callable=validate_schema_quality,
        op_kwargs={"outputs_dir": "{{ ti.xcom_pull(task_ids='generate_schema_stats') }}"},
    )

    dvc_version_task = PythonOperator(
        task_id="version_with_dvc",
        python_callable=version_with_dvc,
        op_kwargs={"features_path": "{{ ti.xcom_pull(task_ids='validate_schema_quality') }}"},
    )

    bias_report_task = PythonOperator(
        task_id="bias_slicing_report",
        python_callable=bias_slicing_report,
        op_kwargs={"features_path": "{{ ti.xcom_pull(task_ids='transform') }}"},
    )

    # Parallelize: extract -> transform -> [schema_stats, anomaly_detect] -> validate -> dvc_version
    #                                                    \-> bias_report
    #                                                    \-> anomaly_email_alert (on failure)
    extract_task >> transform_task
    transform_task >> [schema_stats_task, anomaly_detect_task, bias_report_task]
    [schema_stats_task, anomaly_detect_task] >> validate_schema_task
    validate_schema_task >> dvc_version_task
    anomaly_detect_task >> anomaly_email_alert


dag = supply_chain_pipeline()