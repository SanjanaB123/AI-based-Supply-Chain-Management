"""
scripts/extract.py
Pulls raw data from MongoDB Atlas.
Includes a fingerprint check — skips full pull if data hasn't changed.
Fingerprint is stored as a small JSON file in GCS.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from google.cloud import storage
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MONGO_URI             = os.getenv("MONGO_URI")
MONGO_DB              = os.getenv("MONGO_DB", "inventory_forecasting")
MONGO_COLLECTION      = "retail_store_inventory"
MONGO_SNAP_COLLECTION = "inventory_snapshot"

GCS_BUCKET_NAME       = os.getenv("GCS_BUCKET_NAME", "supply-chain-pipeline")
FINGERPRINT_BLOB      = "metadata/data_fingerprint.json"   # tiny file in GCS

RAW_DIR               = Path(os.getenv("OUTPUT_BASE_PATH", "/opt/airflow/data")) / "raw"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_mongo_fingerprint(client: MongoClient, db_name: str, collection_name: str) -> dict:
    """
    Returns a cheap fingerprint of the MongoDB collection.
    Uses only max(Date) and document count — no full data pull needed.
    """
    collection = client[db_name][collection_name]
    count      = collection.count_documents({})

    # Get the latest date in the collection without pulling all rows
    latest_doc = collection.find_one(
        {}, {"Date": 1, "_id": 0}, sort=[("Date", -1)]
    )
    max_date = str(latest_doc.get("Date", "")) if latest_doc else ""

    return {"count": count, "max_date": max_date}


def _load_fingerprint_from_gcs(bucket_name: str) -> dict | None:
    """Loads the last saved fingerprint JSON from GCS. Returns None if not found."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(FINGERPRINT_BLOB)
        if not blob.exists():
            return None
        data = json.loads(blob.download_as_text())
        log.info("Loaded fingerprint from GCS: %s", data)
        return data
    except Exception as exc:
        log.warning("Could not load fingerprint from GCS: %s", exc)
        return None


def _save_fingerprint_to_gcs(bucket_name: str, fingerprint: dict) -> None:
    """Saves the current fingerprint JSON to GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(FINGERPRINT_BLOB)
        blob.upload_from_string(json.dumps(fingerprint), content_type="application/json")
        log.info("Saved fingerprint to GCS: %s", fingerprint)
    except Exception as exc:
        log.warning("Could not save fingerprint to GCS: %s", exc)


# ── Main extract function ─────────────────────────────────────────────────────

def extract(
    uri:                  str = MONGO_URI,
    db_name:              str = MONGO_DB,
    collection_name:      str = MONGO_COLLECTION,
    snap_collection_name: str = MONGO_SNAP_COLLECTION,
    **context,
) -> str:
    """
    Pull raw documents from MongoDB and write to Parquet.

    Fingerprint check:
      - Compares max(Date) + count against last saved fingerprint in GCS
      - If unchanged → skips full pull, returns path to existing raw parquet
      - If changed   → pulls fresh data, saves new fingerprint

    Returns retail raw path as default XCom.
    Pushes snapshot path under XCom key 'snapshot_path'.
    """
    log.info("Connecting to MongoDB at %s", uri)
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except ConnectionFailure as exc:
        raise RuntimeError(f"Cannot reach MongoDB: {exc}") from exc

    # ── Fingerprint check ─────────────────────────────────────────────────────
    current_fingerprint = _get_mongo_fingerprint(client, db_name, collection_name)
    last_fingerprint    = _load_fingerprint_from_gcs(GCS_BUCKET_NAME)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if last_fingerprint and current_fingerprint == last_fingerprint:
        # Data hasn't changed — find the most recent existing raw parquet and reuse it
        existing_files = sorted(RAW_DIR.glob("raw_*.parquet"), reverse=True)
        if existing_files:
            raw_path      = existing_files[0]
            snap_files    = sorted(RAW_DIR.glob("snapshot_*.parquet"), reverse=True)
            snap_path     = snap_files[0] if snap_files else None

            log.info(
                "Data unchanged (count=%d, max_date=%s) — reusing %s",
                current_fingerprint["count"], current_fingerprint["max_date"], raw_path,
            )

            if snap_path:
                context["ti"].xcom_push(key="snapshot_path", value=str(snap_path))
            client.close()
            return str(raw_path)
        else:
            log.info("No existing raw files found despite matching fingerprint — pulling fresh.")

    # ── Full pull from MongoDB ────────────────────────────────────────────────
    log.info(
        "Data changed or first run (count=%d, max_date=%s) — pulling from MongoDB.",
        current_fingerprint["count"], current_fingerprint["max_date"],
    )

    # Retail time-series
    docs = list(client[db_name][collection_name].find({}, {"_id": 0}))
    if not docs:
        raise ValueError(f"Collection '{collection_name}' is empty or does not exist.")

    df = pd.DataFrame(docs)
    log.info("Extracted %d rows from '%s.%s'", len(df), db_name, collection_name)

    run_id   = context.get("run_id", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    raw_path = RAW_DIR / f"raw_{run_id}.parquet"
    df.to_parquet(raw_path, index=False)
    log.info("Raw retail data written to %s", raw_path)

    # Inventory snapshot
    snap_docs = list(client[db_name][snap_collection_name].find({}, {"_id": 0}))
    if not snap_docs:
        raise ValueError(
            f"Snapshot collection '{snap_collection_name}' is empty or does not exist."
        )

    snap_df   = pd.DataFrame(snap_docs)
    snap_path = RAW_DIR / f"snapshot_{run_id}.parquet"
    snap_df.to_parquet(snap_path, index=False)
    log.info("Snapshot data written to %s (%d rows)", snap_path, len(snap_df))

    # Save updated fingerprint to GCS
    _save_fingerprint_to_gcs(GCS_BUCKET_NAME, current_fingerprint)

    context["ti"].xcom_push(key="snapshot_path", value=str(snap_path))
    client.close()
    return str(raw_path)