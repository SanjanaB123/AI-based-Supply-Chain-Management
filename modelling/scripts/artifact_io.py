"""
artifact_io.py
Save and load preprocessing artifacts (scaler + series mapping) to/from GCS.

Uses google-cloud-storage directly (not MLflow artifact API) because MLflow's
artifact upload requires separate GCS IAM permissions that the CI runner may
not have, while google-cloud-storage respects GOOGLE_APPLICATION_CREDENTIALS
automatically — the same credential path DVC already uses.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
from google.cloud import storage

log = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "supply-chain-pipeline")
ARTIFACTS_PREFIX = "preprocessing-artifacts"


# ── Save helpers ──────────────────────────────────────────────────────────────

def _upload_blob(local_path: Path, blob_name: str) -> str:
    """Upload a local file to GCS and return the gs:// URI."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{GCS_BUCKET}/{blob_name}"
    log.info("Uploaded %s → %s", local_path, uri)
    return uri


def save_preprocessing_artifacts(
    scaler,
    scaler_local_path: Path,
    series_mapping: dict,
    mapping_local_dir: Path,
) -> dict:
    """
    Upload scaler and series mapping to GCS.

    Parameters
    ----------
    scaler : fitted StandardScaler (already saved locally by caller)
    scaler_local_path : path to the local scaler.pkl (already written by joblib.dump)
    series_mapping : {series_id_string: int} dict from encode_series()
    mapping_local_dir : directory where series_mapping.json will be written locally

    Returns
    -------
    dict with keys "scaler_gcs_uri" and "mapping_gcs_uri"
    """
    # Save mapping locally first
    mapping_local_path = Path(mapping_local_dir) / "series_mapping.json"
    with open(mapping_local_path, "w") as f:
        json.dump(series_mapping, f, indent=2)
    log.info("Series mapping saved locally to %s", mapping_local_path)

    # Upload both to GCS
    scaler_uri = _upload_blob(
        scaler_local_path, f"{ARTIFACTS_PREFIX}/scaler.pkl"
    )
    mapping_uri = _upload_blob(
        mapping_local_path, f"{ARTIFACTS_PREFIX}/series_mapping.json"
    )

    return {"scaler_gcs_uri": scaler_uri, "mapping_gcs_uri": mapping_uri}


# ── Load helpers ──────────────────────────────────────────────────────────────

def _download_blob(blob_name: str, local_path: Path) -> Path:
    """Download a GCS blob to a local path."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    log.info("Downloaded gs://%s/%s → %s", GCS_BUCKET, blob_name, local_path)
    return local_path


def load_preprocessing_artifacts(
    local_dir: Path = Path("/tmp/preprocessing"),
) -> tuple:
    """
    Download scaler and series mapping from GCS.

    Returns
    -------
    (scaler, series_mapping) where scaler is a fitted StandardScaler
    and series_mapping is a {series_id: int} dict.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = _download_blob(
        f"{ARTIFACTS_PREFIX}/scaler.pkl", local_dir / "scaler.pkl"
    )
    mapping_path = _download_blob(
        f"{ARTIFACTS_PREFIX}/series_mapping.json",
        local_dir / "series_mapping.json",
    )

    scaler = joblib.load(scaler_path)
    with open(mapping_path) as f:
        series_mapping = json.load(f)

    log.info(
        "Loaded preprocessing artifacts: scaler (%d features), mapping (%d series)",
        len(scaler.mean_) if hasattr(scaler, "mean_") else -1,
        len(series_mapping),
    )
    return scaler, series_mapping
