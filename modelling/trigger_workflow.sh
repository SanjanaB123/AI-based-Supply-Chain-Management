#!/bin/bash

# trigger_workflow.sh
Bash script to trigger GitHub Actions workflow

set -e

# Load environment variables from .env file
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Loaded environment from .env file"
else
    echo " .env file not found"
fi

# Configuration
REPO_OWNER="SanjanaB123"
REPO_NAME="model_ci_cd"
WORKFLOW_FILE="ml_pipeline.yml"

# Get GitHub token from environment or prompt
if [ -z "$GITHUB_TOKEN" ]; then
    echo "GITHUB_TOKEN environment variable not found"
    echo "Set it with: export GITHUB_TOKEN=your_token"
    echo "Or create a Personal Access Token at: https://github.com/settings/tokens"
    exit 1
fi

echo "Triggering workflow: $WORKFLOW_FILE"
echo "Repository: $REPO_OWNER/$REPO_NAME"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# Trigger workflow via GitHub API
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/actions/workflows/$WORKFLOW_FILE/dispatches \
  -d '{"ref":"main"}'

echo
echo "Workflow triggered!"
echo "Watch progress: https://github.com/$REPO_OWNER/$REPO_NAME/actions"
