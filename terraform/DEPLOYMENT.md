# Airflow Pipeline Deployment to GCP Cloud Run

## Table of Contents
- [1. Overview](#1-overview)
- [2. Current System Architecture](#2-current-system-architecture)
- [3. Deployment Architecture](#3-deployment-architecture)
- [4. Technology Stack](#4-technology-stack)
- [5. Why Cloud Run (Not a VM)](#5-why-cloud-run-not-a-vm)
- [6. How the System Runs After Deployment](#6-how-the-system-runs-after-deployment)
- [7. What We Build](#7-what-we-build)
- [8. Implementation Steps](#8-implementation-steps)
- [9. Replication Guide (Fresh Environment)](#9-replication-guide-fresh-environment)
- [10. Submission Guidelines Compliance](#10-submission-guidelines-compliance)

---

## 1. Overview

This document describes the deployment of our Airflow-based data pipeline from a local Docker Compose setup to Google Cloud Platform (GCP) using Cloud Run, Terraform, and GitHub Actions.

**Project:** AI-based Supply Chain Management (MLOps)
**GCP Project ID:** `mlops-project-488302`
**GCP Region:** `us-central1`
**Deployment Service:** Cloud Run
**Infrastructure Automation:** Terraform
**CI/CD:** GitHub Actions

---

## 2. Current System Architecture

### What exists today

The system has two main pipelines that work together:

### Pipeline 1: Airflow Data Pipeline (runs locally)

Started with `docker-compose up` on a developer's laptop. Creates 5 Docker containers:

| Container | Role |
|-----------|------|
| Postgres | Stores Airflow's internal metadata (task status, logs, etc.) |
| Airflow API Server | The web UI at `localhost:8080` |
| Airflow Scheduler | Decides WHEN to run DAG tasks |
| DAG Processor | Reads DAG Python files and parses task dependencies |
| Triggerer | Handles async/deferred tasks |

**What the DAG does (daily at 12:00 PM UTC):**
```
Extract (MongoDB)
  -> Transform & Feature Engineering
    -> Schema Validation (Great Expectations)
    -> Anomaly Detection (z-score, IQR, missingness, date gaps)
    -> Bias Analysis (across stores, products, weather, seasons)
  -> Quality Gate
  -> DVC Versioning (pushes to GCS)
  -> Upload to GCS
  -> Email Alert (if anomalies found)
```

### Pipeline 2: ML Model Training (runs on GitHub Actions)

Triggered when `.dvc` files change (meaning new data was versioned):

```
Pull data via DVC
  -> Split train/val/test (chronological, with 14-day gaps)
  -> Tune XGBoost hyperparameters (Optuna)
  -> Tune Prophet hyperparameters (Optuna)
  -> Train XGBoost model
  -> Train Prophet model
  -> Select best model (5% improvement gate)
  -> Promote winner to MLflow Production
  -> Bias detection on best model
  -> Sensitivity analysis
```

### How they connect

```
Airflow DAG (data pipeline)
    |
    | produces new features.parquet
    | DVC versions it, pushes .dvc file to GitHub
    |
    v
GitHub detects .dvc file change
    |
    | triggers ml_pipeline.yml
    |
    v
GitHub Actions (model pipeline)
    |
    | trains models, logs to MLflow
    | promotes best model to Production
    |
    v
MLflow Server (on Cloud Run)
    |
    | serves production model info
    v
Inference (future: API on Cloud Run)
```

### External services already in use
- **MongoDB** (cloud) - raw inventory data
- **GCS Bucket** (`supply-chain-pipeline`) - features, DVC remote, data storage
- **MLflow Server** (`https://mlflow-833456981899.us-central1.run.app/`) - experiment tracking, model registry
- **GitHub Actions** - ML pipeline CI/CD

---

## 3. Deployment Architecture

### What changes, what stays the same

| Component | Before (Local) | After (Cloud) | Changes? |
|-----------|----------------|---------------|----------|
| Airflow DAG | Runs on laptop | Runs on Cloud Run | **Moves to cloud** |
| Postgres (Airflow DB) | Docker container | Cloud SQL (managed) | **Moves to cloud** |
| Data pipeline logic | Python scripts | Same scripts, baked in Docker image | **No change** |
| DVC | Tracks data in GCS | Same | **No change** |
| ML Pipeline | GitHub Actions | Same | **No change** |
| MLflow Server | Cloud Run | Same | **No change** |
| GCS Bucket | Stores features | Same | **No change** |
| Terraform | Doesn't exist | Creates all cloud infra | **New** |
| Deploy workflow | Doesn't exist | Builds & deploys Airflow | **New** |

### Cloud architecture diagram

```
                          +---------------+
                          |   INTERNET    |
                          +-------+-------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                        GCP PROJECT                                 |
|                   (mlops-project-488302)                           |
|                                                                    |
|  +-------------------------------------------------------------+  |
|  |                   CLOUD RUN SERVICES                         |  |
|  |                                                              |  |
|  |  +------------------+      +-------------------+             |  |
|  |  | WEBSERVER        |      | SCHEDULER          |            |  |
|  |  | (api-server)     |<-----| (always-on)        |            |  |
|  |  | Port 8080        |      | picks & runs       |            |  |
|  |  | PUBLIC access    |      | your DAG tasks     |            |  |
|  |  +------------------+      +-------------------+             |  |
|  |                                                              |  |
|  |  +------------------+      +-------------------+             |  |
|  |  | DAG PROCESSOR    |      | TRIGGERER          |            |  |
|  |  | (always-on)      |      | (always-on)        |            |  |
|  |  | parses DAG files |      | handles deferred   |            |  |
|  |  +------------------+      | tasks              |            |  |
|  |                            +-------------------+             |  |
|  +-------------------------------------------------------------+  |
|           |              |                |                        |
|           v              v                v                        |
|  +------------------+  +---------------------+                    |
|  | CLOUD SQL        |  | SECRET MANAGER      |                    |
|  | (PostgreSQL 15)  |  |                     |                    |
|  | Airflow metadata |  | - MONGO_URI         |                    |
|  | Managed backups  |  | - DB password        |                    |
|  +------------------+  | - Fernet key         |                    |
|                        | - JWT secret          |                    |
|                        | - GitHub token        |                    |
|                        +---------------------+                    |
|                                                                    |
|  +------------------+  +---------------------+                    |
|  | ARTIFACT REGISTRY|  | GCS BUCKET          |                    |
|  | (Docker images)  |  | (pipeline data)     |                    |
|  +------------------+  +---------------------+                    |
|                                                                    |
|       All managed by TERRAFORM (infrastructure as code)           |
+-------------------------------------------------------------------+
```

### Terraform modules breakdown

| Module | What it creates | Why |
|--------|----------------|-----|
| `networking` | VPC, subnet, VPC connector | Private connectivity between Cloud Run and Cloud SQL |
| `cloud-sql` | PostgreSQL 15 instance, database, user | Replaces local Postgres container |
| `artifact-registry` | Docker image repository | Stores custom Airflow Docker images |
| `secret-manager` | Secret entries for all sensitive values | Secure storage (no hardcoded passwords) |
| `iam` | Service account + role bindings | Permissions for Cloud Run to access SQL, GCS, secrets |
| `cloud-run` | 4 services + 1 init job | The actual Airflow deployment |

---

## 4. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Cloud Provider | GCP | Main cloud platform |
| Compute | Cloud Run | Runs Airflow containers |
| Database | Cloud SQL (PostgreSQL 15) | Airflow metadata |
| Container Registry | Artifact Registry | Docker image storage |
| Secrets | Secret Manager | Secure credential storage |
| Networking | VPC + Serverless VPC Connector | Private Cloud SQL access |
| Infrastructure as Code | Terraform | Creates and manages all GCP resources |
| CI/CD (deployment) | GitHub Actions | Automated build and deploy on code push |
| CI/CD (ML pipeline) | GitHub Actions | Model training on data changes |
| Data Pipeline | Apache Airflow 3.0.3 | DAG orchestration |
| Data Versioning | DVC | Feature data versioning in GCS |
| Experiment Tracking | MLflow | Model registry and metrics |
| Data Storage | GCS | Features, DVC remote, reports |
| Raw Data | MongoDB | Source inventory data |

---

## 5. Why Cloud Run (Not a VM)

### VM approach (lift-and-shift)
- Just run `docker-compose up` on a GCE VM
- Fast to set up (~30 minutes)
- **Downsides:** Manual OS patching, no auto-healing, you pay 24/7, no built-in CI/CD, manual scaling, you become the sysadmin

### Cloud Run approach (what we chose)
- Fully managed containers, no OS to manage
- Auto-healing (crashed containers restart automatically)
- Infrastructure as Code (Terraform) - reproducible, version-controlled
- Built-in logging and monitoring (Cloud Logging, Cloud Monitoring)
- Clean CI/CD story (GitHub Actions -> build -> deploy, no SSH needed)
- Managed database (Cloud SQL) with automated backups
- **Downsides:** More upfront setup, DAG changes require image rebuild

### Decision rationale
The project is an MLOps showcase. A VM with `docker-compose up` bypasses both Terraform and GitHub Actions. Cloud Run is architecturally aligned with our chosen stack and demonstrates production-grade patterns.

---

## 6. How the System Runs After Deployment

### Two separate automated processes

**Process 1: Daily pipeline (runs automatically, no human needed)**
```
Cloud Run Scheduler (always running)
    |
    | "It's 12:00 PM UTC, time to run the DAG"
    v
DAG executes: Extract -> Transform -> Validate -> DVC -> GCS
    |
    | DVC pushes new .dvc file to GitHub
    v
GitHub Actions (ml_pipeline.yml) triggers automatically
    |
    | Trains XGBoost + Prophet -> Selects best -> MLflow Production
    v
Done. Repeats tomorrow.
```

**Process 2: Code deployment (runs only when you push code)**
```
Developer pushes to main branch (airflow/ or terraform/ changes)
    |
    v
GitHub Actions (deploy-airflow.yml) triggers
    |
    | Builds Docker image -> Pushes to Artifact Registry
    | Runs Terraform -> Updates Cloud Run services
    v
Cloud Run services updated with new code
```

### Key insight
After the initial deployment, **nobody needs to do anything**. The data pipeline runs daily on schedule, trains models when data changes, and the best model is promoted to production — all automatically.

---

## 7. What We Build

### New files

| File | Purpose |
|------|---------|
| `airflow/Dockerfile` | Custom Airflow image with DAGs baked in |
| `airflow/requirements.txt` | Python packages for Airflow |
| `airflow/params.yaml` | Pipeline parameters (defaults) |
| `terraform/main.tf` | Root Terraform config |
| `terraform/variables.tf` | Input variables |
| `terraform/outputs.tf` | Output values (URLs) |
| `terraform/versions.tf` | Provider versions |
| `terraform/terraform.tfvars` | Actual values (project ID, region) |
| `terraform/modules/networking/*` | VPC + connector |
| `terraform/modules/cloud-sql/*` | Managed Postgres |
| `terraform/modules/artifact-registry/*` | Docker image repo |
| `terraform/modules/secret-manager/*` | Secrets storage |
| `terraform/modules/iam/*` | Service account + permissions |
| `terraform/modules/cloud-run/*` | Cloud Run services + init job |
| `.github/workflows/deploy-airflow.yml` | CI/CD deployment pipeline |

### Modified files

| File | Change |
|------|--------|
| `airflow/scripts/upload_to_gcp.py` | Use Application Default Credentials (works on both Cloud Run and local) |

---

## 8. Implementation Steps

### Step 1: Airflow Dockerfile
Create a custom Docker image based on `apache/airflow:3.0.3` that includes:
- All pip dependencies (pandas, pymongo, great-expectations, etc.)
- DAG files baked in (`/opt/airflow/dags/`)
- Script files baked in (`/opt/airflow/scripts/`)
- Default params.yaml baked in (`/opt/airflow/params.yaml`)

### Step 2: Fix GCS authentication
Update `upload_to_gcp.py` to use Application Default Credentials (ADC) instead of a key file. ADC works automatically on Cloud Run (via the attached service account) and locally (via `gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`).

### Step 3: Create default params.yaml
Create the pipeline parameters file with sensible defaults so it can be baked into the Docker image.

### Step 4: Terraform infrastructure
Create all Terraform modules to provision:
- VPC + Serverless VPC Connector
- Cloud SQL PostgreSQL instance
- Artifact Registry repository
- Secret Manager secrets
- Service account with appropriate roles
- 4 Cloud Run services + 1 Cloud Run Job (init)

### Step 5: GitHub Actions deployment workflow
Create a workflow that:
1. Builds the custom Airflow Docker image
2. Pushes it to Artifact Registry
3. Runs `terraform apply` to create/update infrastructure
4. Executes the `airflow-init` Cloud Run Job (DB migration + admin user creation)
5. Smoke tests the webserver URL

### Step 6: One-time GCP setup
Manual steps to prepare GCP:
- Create Terraform state bucket
- Add GitHub repository secrets
- Enable required GCP APIs

---

## 9. Replication Guide (Fresh Environment)

### Prerequisites
- Google Cloud account with billing enabled
- GitHub account
- `gcloud` CLI installed
- `terraform` CLI installed
- Git installed

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/SanjanaB123/AI-based-Supply-Chain-Management.git
   cd AI-based-Supply-Chain-Management
   ```

2. **Set up GCP**
   ```bash
   gcloud auth login
   gcloud config set project mlops-project-488302

   # Enable required APIs
   gcloud services enable \
     run.googleapis.com \
     sqladmin.googleapis.com \
     artifactregistry.googleapis.com \
     secretmanager.googleapis.com \
     vpcaccess.googleapis.com \
     compute.googleapis.com \
     servicenetworking.googleapis.com

   # Create Terraform state bucket
   gsutil mb -l us-central1 gs://mlops-project-488302-tfstate
   ```

3. **Add GitHub secrets**
   In your GitHub repo → Settings → Secrets and variables → Actions:
   - `GCP_SA_KEY` — service account JSON key
   - `TF_VAR_mongo_uri` — MongoDB connection string
   - `TF_VAR_db_password` — Cloud SQL password (you choose)
   - `TF_VAR_airflow_fernet_key` — Airflow encryption key
   - `TF_VAR_airflow_jwt_secret` — Airflow JWT secret
   - `TF_VAR_github_token` — GitHub token for DVC push

4. **Push to main branch**
   ```bash
   git push origin main
   ```

5. **GitHub Actions deploys automatically**
   - Go to GitHub → Actions tab → watch the `Deploy Airflow` workflow
   - Once complete, the Airflow URL is printed in the workflow output

6. **Access Airflow**
   - Open the Cloud Run URL from the workflow output
   - Login with the configured admin credentials
   - Your DAG (`supply_chain_pipeline`) is visible
   - Click "Trigger" to run it manually, or wait for the daily schedule

---

## 10. Submission Guidelines Compliance

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Specify cloud provider + service | GCP + Cloud Run | Done |
| Deployment automation scripts | Terraform + GitHub Actions | Done |
| Pull latest model from registry | MLflow model registry (existing) | Done |
| Auto-deploy on code push | GitHub Actions deploy workflow | Done |
| Monitor deployment status | Cloud Run logs + GitHub Actions status | Done |
| Connection to repository (CI/CD) | GitHub Actions triggered on push to main | Done |
| Detailed replication steps | Section 9 of this document | Done |
| Environment configuration files | Dockerfile, Terraform configs, docker-compose | Done |
| Logs and monitoring | Cloud Run built-in logging + Airflow logs | Done |
| Model monitoring & data drift | To be implemented (separate work) | Pending |
| Trigger retraining on drift | To be implemented (separate work) | Pending |
| Notifications on retraining | To be implemented (separate work) | Pending |
| Video demo on fresh environment | To be recorded after deployment works | Pending |
