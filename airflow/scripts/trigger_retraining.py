#!/usr/bin/env python3
"""
Trigger retraining via GitHub Actions workflow dispatch when drift is detected.

Reads drift_report.json produced by drift_detection.py, and if drift is
detected fires a workflow_dispatch event on ml_pipeline.yml in the configured
GitHub repository.

Environment variables:
    GITHUB_TOKEN   — personal access token with `repo` + `actions` scope
    GITHUB_REPO    — owner/repo, e.g. "myorg/AI-based-Supply-Chain-Management"
    GITHUB_BRANCH  — branch to dispatch on (default: "main")
"""

import json
import logging
import os

import requests

log = logging.getLogger(__name__)

WORKFLOW_FILE = "ml_pipeline.yml"


def trigger_retraining_if_drift(report_path: str) -> bool:
    """
    Read drift report and fire GitHub workflow dispatch if drift detected.

    Args:
        report_path: Path to drift_report.json written by drift_detection.py.

    Returns:
        True if retraining was triggered, False otherwise.
    """
    with open(report_path, "r") as f:
        report = json.load(f)

    drift_detected = report["summary"]["drift_detected"]
    share_drifted = report["summary"]["share_of_drifted_columns"]

    if not drift_detected:
        log.info(
            "No drift detected (%.1f%% of features drifted). Skipping retraining.",
            share_drifted * 100,
        )
        return False

    log.warning(
        "Drift detected: %.1f%% of features drifted. Triggering retraining.",
        share_drifted * 100,
    )

    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO", "")
    branch = os.getenv("GITHUB_BRANCH", "main")

    if not token:
        log.error("GITHUB_TOKEN not set — cannot trigger retraining workflow.")
        return False
    if not repo:
        log.error("GITHUB_REPO not set — cannot trigger retraining workflow.")
        return False

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    payload = {
        "ref": branch,
        "inputs": {
            "reason": f"drift_detected:{share_drifted:.3f}",
        },
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 204:
            log.info(
                "Retraining workflow dispatched successfully on %s/%s (branch: %s).",
                repo, WORKFLOW_FILE, branch,
            )
            return True
        else:
            log.error(
                "Failed to dispatch retraining workflow: HTTP %d — %s",
                response.status_code,
                response.text,
            )
            return False
    except requests.exceptions.RequestException as exc:
        log.error("Network error triggering retraining: %s", exc)
        return False
