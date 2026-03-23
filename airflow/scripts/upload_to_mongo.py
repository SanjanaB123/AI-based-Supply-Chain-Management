#!/usr/bin/env python3
"""
Upload a file (CSV or JSON) to a MongoDB collection.

Usage:
    python upload_to_mongo.py --file <path_to_file> [options]

Examples:
    # Upload a CSV to the default collection
    python upload_to_mongo.py --file data/raw/retail_store_inventory.csv

    # Upload to a specific collection and database
    python upload_to_mongo.py --file data/raw/retail_store_inventory.csv \
        --db supply_chain --collection inventory

    # Upload a JSON file
    python upload_to_mongo.py --file data/raw/orders.json --collection orders
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, ConnectionFailure


# ──────────────────────────────────────────────
# Configuration defaults (override via CLI args)
# ──────────────────────────────────────────────
DEFAULT_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB         = os.getenv("MONGO_DB",  "supply_chain")
DEFAULT_BATCH_SIZE = 1000  # number of docs per bulk-insert batch


def load_file(file_path: Path) -> list[dict]:
    """Load a CSV or JSON file and return a list of dicts (documents)."""
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        print(f"[INFO] Reading CSV: {file_path}")
        df = pd.read_csv(file_path)
        # Replace NaN with None so MongoDB stores null instead of NaN strings
        df = df.where(pd.notnull(df), None)
        return df.to_dict(orient="records")

    elif suffix == ".json":
        print(f"[INFO] Reading JSON: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept either a top-level list or a dict wrapping a list
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try to find the first list value in the dict
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]  # single document
        else:
            raise ValueError("JSON file must contain an array or object at the top level.")

    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Only .csv and .json are supported.")


def upload_to_mongo(
    documents: list[dict],
    uri: str,
    db_name: str,
    collection_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    drop_first: bool = False,
) -> None:
    """Connect to MongoDB and insert documents in batches."""

    print(f"[INFO] Connecting to MongoDB at {uri} ...")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # verify connection
    except ConnectionFailure as e:
        print(f"[ERROR] Cannot connect to MongoDB: {e}")
        sys.exit(1)

    db         = client[db_name]
    collection = db[collection_name]

    if drop_first:
        print(f"[WARN] Dropping existing collection '{collection_name}' before upload ...")
        collection.drop()

    total    = len(documents)
    inserted = 0

    print(f"[INFO] Uploading {total:,} documents to '{db_name}.{collection_name}' in batches of {batch_size} ...")

    for start in range(0, total, batch_size):
        batch = documents[start : start + batch_size]
        try:
            result = collection.insert_many(batch, ordered=False)
            inserted += len(result.inserted_ids)
        except BulkWriteError as bwe:
            # Some docs may still have been inserted
            inserted += bwe.details.get("nInserted", 0)
            print(f"[WARN] Bulk write error on batch starting at {start}: {bwe.details['writeErrors'][0]['errmsg']}")

        progress = min(start + batch_size, total)
        print(f"  ... {progress:,}/{total:,} processed", end="\r")

    print(f"\n[SUCCESS] Inserted {inserted:,} / {total:,} documents into '{db_name}.{collection_name}'.")
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a CSV or JSON file to a MongoDB collection."
    )
    parser.add_argument(
        "--file", "-f", required=True,
        help="Path to the CSV or JSON file to upload."
    )
    parser.add_argument(
        "--uri", default=DEFAULT_URI,
        help=f"MongoDB connection URI (default: {DEFAULT_URI})."
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"Target database name (default: {DEFAULT_DB})."
    )
    parser.add_argument(
        "--collection", "-c", default=None,
        help="Target collection name. Defaults to the file name (without extension)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Number of documents per insert batch (default: {DEFAULT_BATCH_SIZE})."
    )
    parser.add_argument(
        "--drop", action="store_true",
        help="Drop the collection before uploading (replaces existing data)."
    )

    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    # Default collection name = filename without extension
    collection_name = args.collection or file_path.stem

    documents = load_file(file_path)

    if not documents:
        print("[WARN] No documents found in the file. Nothing to upload.")
        sys.exit(0)

    upload_to_mongo(
        documents=documents,
        uri=args.uri,
        db_name=args.db,
        collection_name=collection_name,
        batch_size=args.batch_size,
        drop_first=args.drop,
    )


if __name__ == "__main__":
    main()
