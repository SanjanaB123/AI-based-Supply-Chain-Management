#!/usr/bin/env python3
"""
trigger_github_workflow.py
Script to trigger GitHub Actions workflow via API
"""

import requests
import json
import sys
import os
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("You can still use by setting GITHUB_TOKEN environment variable manually")

# Configuration
REPO_OWNER = "SanjanaB123"
REPO_NAME = "model_ci_cd"
WORKFLOW_FILE = "ml_pipeline.yml"

def get_github_token():
    """Get GitHub token from environment or prompt user"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print(" environment variable not found")
        print("Set it with: export GITHUB_TOKEN=your_token")
        print("Or create a Personal Access Token at: https://github.com/settings/tokens")
        sys.exit(1)
    return token

def trigger_workflow():
    """Trigger the ML pipeline workflow via GitHub API"""
    token = get_github_token()
    
    # API endpoint for workflow dispatch
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    # Payload for workflow dispatch (can be empty for basic trigger)
    payload = {
        "ref": "main"
    }
    
    try:
        print(f"Triggering workflow: {WORKFLOW_FILE}")
        print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 204:
            print("Workflow triggered successfully!")
            print(f"Watch progress: https://github.com/{REPO_OWNER}/{REPO_NAME}/actions")
        else:
            print(f"Failed to trigger workflow")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        sys.exit(1)

def check_workflow_status():
    """Check current ML pipeline workflow runs only"""
    token = get_github_token()
    
    # Get runs for specific workflow only
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/runs"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            runs = response.json().get("workflow_runs", [])
            if runs:
                latest = runs[0]  # GitHub API returns most recent first
                status = latest["status"]
                conclusion = latest.get("conclusion", "running")
                created_at = latest["created_at"]
                
                print(f"Latest ML Pipeline run:")
                print(f"   Status: {status}")
                print(f"   Conclusion: {conclusion}")
                print(f"   Created: {created_at}")
                print(f"   URL: {latest['html_url']}")
            else:
                print("No ML Pipeline workflow runs found")
        else:
            print(f"Failed to check status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_workflow_status()
    else:
        trigger_workflow()
