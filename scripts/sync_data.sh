#!/bin/bash

# Configuration
FEATURES_PATH=$1
COMMIT_MESSAGE=${2:-"Update tracked data via DVC: $(date +'%Y-%m-%d %H:%M:%S')"}

# Ensure we are in the project root for git/dvc commands
cd /opt/airflow || exit 1

if [ -z "$FEATURES_PATH" ]; then
    echo "Usage: $0 <features_path> [commit_message]"
    exit 1
fi

echo "Starting DVC and Git sync for: $FEATURES_PATH"

# 0.1 Handle DVC Initialization
if [ ! -f ".dvc/config" ]; then
    echo "DVC not initialized. Initializing..."
    dvc init --no-scm -f
fi

# 0.2 Handle DVC Remote configuration
REMOTES=$(dvc remote list)
if [ -z "$REMOTES" ]; then
    if [ -n "$GCS_BUCKET_NAME" ] && [ -f "/opt/airflow/gcp-key.json" ]; then
        echo "Configuring DVC GCS remote..."
        dvc remote add -d storage "gs://${GCS_BUCKET_NAME}/dvc"
        # Use --local for the credential path so it doesn't get pushed to GitHub
        dvc remote modify --local storage credentialpath "/opt/airflow/gcp-key.json"
    else
        echo "Warning: GCS_BUCKET_NAME or gcp-key.json missing."
    fi
fi

# 1. DVC Add
echo "Step 1: dvc add $FEATURES_PATH"
dvc add "$FEATURES_PATH"
if [ $? -ne 0 ]; then
    echo "Error: dvc add failed"
    exit 1
fi

# 2. DVC Push
echo "Step 2: dvc push"
dvc push
if [ $? -ne 0 ]; then
    echo "Error: dvc push failed"
    exit 1
fi

# 3. GitHub API Push (instead of git commands)
DVC_FILE="${FEATURES_PATH}.dvc"
DVC_CONFIG=".dvc/config"
echo "Step 3: Pushing $DVC_FILE and $DVC_CONFIG to GitHub via API"
python3 /opt/airflow/scripts/github_push.py "push" "$COMMIT_MESSAGE" "$DVC_FILE" "$DVC_CONFIG"

if [ $? -ne 0 ]; then
    echo "Error: GitHub API push failed"
    exit 1
fi

echo "Sync completed successfully via GitHub API!"
exit 0
