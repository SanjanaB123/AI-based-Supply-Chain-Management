#!/usr/bin/env python3
"""
rollback_model.py
Rolls back a model in MLflow by demoting the current Production version
to Archived and promoting the most recent Archived version to Production.

Usage:
    python rollback_model.py --model-name xgboost-supply-chain
    python rollback_model.py --model-name xgboost-supply-chain --target-version 3
"""

import argparse
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://mlflow-952666479463.us-central1.run.app/",
)


def rollback(model_name: str, target_version: int | None = None) -> bool:
    """
    Rollback logic:
    1. Find the current Production version.
    2. Find the rollback target (latest Archived, or a specific version).
    3. Demote current Production -> Archived.
    4. Promote target -> Production.
    5. Print summary with version numbers and MAE metrics.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # ── Step 1: find current production version ──────────────────────
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        print(f"ERROR: No Production version found for '{model_name}'. Nothing to roll back.")
        return False

    current_prod = prod_versions[0]
    print(f"Current Production: version {current_prod.version}")

    # ── Step 2: find rollback target ─────────────────────────────────
    if target_version is not None:
        try:
            rollback_target = client.get_model_version(model_name, str(target_version))
        except Exception as exc:
            print(f"ERROR: Could not fetch version {target_version}: {exc}")
            return False

        if rollback_target.current_stage != "Archived":
            print(
                f"WARNING: Version {target_version} is in stage "
                f"'{rollback_target.current_stage}', not Archived."
            )
    else:
        archived = client.get_latest_versions(model_name, stages=["Archived"])
        if not archived:
            print(f"ERROR: No Archived versions found for '{model_name}'. Cannot roll back.")
            return False
        rollback_target = max(archived, key=lambda v: int(v.version))

    if rollback_target.version == current_prod.version:
        print("ERROR: Rollback target is the same as current Production. Nothing to do.")
        return False

    print(f"Rollback target:   version {rollback_target.version}")

    # ── Step 3: demote current production ────────────────────────────
    client.transition_model_version_stage(
        name=model_name,
        version=current_prod.version,
        stage="Archived",
    )

    # ── Step 4: promote rollback target ──────────────────────────────
    client.transition_model_version_stage(
        name=model_name,
        version=rollback_target.version,
        stage="Production",
    )

    # ── Step 5: summary ──────────────────────────────────────────────
    print(f"\nRollback complete for '{model_name}':")
    print(f"  Demoted:  version {current_prod.version} (Production -> Archived)")
    print(f"  Promoted: version {rollback_target.version} (Archived -> Production)")

    for label, ver in [("Old prod", current_prod), ("Restored", rollback_target)]:
        try:
            run = client.get_run(ver.run_id)
            mae = run.data.metrics.get("test_mae", "N/A")
            print(f"  {label} test MAE: {mae}")
        except Exception:
            pass

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rollback MLflow model to a previous version"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Registered model name (e.g. xgboost-supply-chain)",
    )
    parser.add_argument(
        "--target-version",
        type=int,
        default=None,
        help="Specific version number to restore (default: latest Archived)",
    )
    args = parser.parse_args()

    success = rollback(args.model_name, args.target_version)
    sys.exit(0 if success else 1)
