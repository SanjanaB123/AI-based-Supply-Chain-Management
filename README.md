# Stratos - AI-Powered Supply Chain Intelligence

A production-grade MLOps platform for retail demand forecasting and inventory management, built with FastAPI, React, LangGraph agents, and deployed on GCP + Vercel.

**Live Demo**: [Vercel Frontend](https://your-app.vercel.app) | **Backend**: Cloud Run | **ML Tracking**: MLflow on Cloud Run

## Architecture

```
Frontend (Vercel)  -->  Backend API (Cloud Run)  -->  MCP Server (Cloud Run)
     React/Vite          FastAPI + Claude AI          MongoDB + MLflow
     Clerk Auth          LangGraph Agents             XGBoost/Prophet
     Tailwind CSS        Gemini Chat (orders)         Demand Forecasting
```

### Services

| Service | Tech | Deployed To |
|---------|------|-------------|
| **Frontend** | React 19, TypeScript, Vite, Tailwind, Clerk Auth | Vercel |
| **Backend API** | FastAPI, SetFit classifier, LangGraph, Claude | GCP Cloud Run |
| **MCP Server** | FastMCP, PyMongo, MLflow, XGBoost/Prophet | GCP Cloud Run |
| **ML Training** | XGBoost, Prophet, Optuna, SHAP | GitHub Actions + Docker |
| **Data Pipeline** | Apache Airflow 3.0, Great Expectations, DVC | Docker Compose / Cloud Run |
| **Model Registry** | MLflow 2.22 | GCP Cloud Run |
| **Database** | MongoDB Atlas | MongoDB Cloud |
| **Storage** | Google Cloud Storage | GCP |
| **Infrastructure** | Terraform (6 modules) | GCP |

## Project Structure

```
AI-based-Supply-Chain-Management/
├── frontend/                 # React/TypeScript SPA
│   ├── src/
│   │   ├── pages/            # Dashboard, Analytics, Inventory, Risk, AI Assistant
│   │   ├── components/       # Chat panels, charts, data tables
│   │   ├── hooks/            # useGeminiChat, useChatAssistant, useInventoryData
│   │   └── lib/              # API client, config
│   ├── vercel.json           # SPA rewrite rules
│   └── package.json
├── backend/                  # FastAPI backend
│   ├── main.py               # App entrypoint with lifespan, CORS, rate limiting
│   ├── routes_inventory.py   # Inventory dashboard endpoints
│   ├── routes_chat.py        # LangGraph + MCP chat (Claude Haiku)
│   ├── routes_gemini_chat.py # AI Assistant with tool calling (Claude Sonnet)
│   ├── routes_email.py       # Email alert endpoints
│   ├── clerk_auth.py         # Clerk JWT verification
│   ├── mcp/                  # MCP inference server
│   │   └── server.py         # FastMCP tools: inventory, forecasting, restocking
│   ├── config/               # Agent configs, prompts, SetFit classifier model
│   └── Dockerfile
├── modelling/                # ML training pipeline
│   ├── models/               # XGBoost, Prophet, Optuna tuning
│   ├── scripts/              # Data splitting, bias detection, SHAP, inference
│   └── Dockerfile            # Multi-stage build with CmdStan for Prophet
├── airflow/                  # Data pipeline
│   ├── dags/                 # Airflow DAG definitions
│   ├── scripts/              # ETL: extract, transform, validate, anomaly, bias
│   └── tests/                # 42 unit tests
├── terraform/                # GCP infrastructure as code
│   ├── main.tf               # Root module
│   └── modules/              # artifact-registry, cloud-run, cloud-sql, iam, networking, secret-manager
├── .github/workflows/
│   ├── ml_pipeline.yml       # ML training CI/CD (triggered on data changes)
│   ├── deploy-airflow.yml    # Airflow deployment to Cloud Run
│   ├── rollback.yml          # Model rollback via MLflow
│   └── frontend-ci.yml       # Frontend lint + typecheck + build
├── docker-compose.yaml       # Airflow local stack (Postgres, scheduler, API server)
└── params.yaml               # Pipeline parameters and thresholds
```

## Deployment

### Prerequisites

- GCP account with billing enabled
- MongoDB Atlas cluster
- Clerk account (authentication)
- Anthropic API key
- Vercel account

### 1. Frontend (Vercel)

```bash
# Import repo on vercel.com, set root directory to "frontend"
# Set environment variables:
VITE_CLERK_PUBLISHABLE_KEY=pk_test_...
VITE_API_BASE_URL=https://backend-XXXXX.run.app
```

Vercel auto-detects Vite and deploys. The `vercel.json` handles SPA routing.

### 2. Backend API (Cloud Run)

```bash
cd backend
gcloud run deploy backend \
  --source . \
  --region us-central1 \
  --port 8000 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2 \
  --set-env-vars \
    ANTHROPIC_API_KEY="...",\
    MONGO_URI="mongodb+srv://...",\
    MCP_SERVER_URL="https://mcp-XXXXX.run.app/mcp",\
    CLERK_JWKS_URL="https://....clerk.accounts.dev/.well-known/jwks.json",\
    MLFLOW_TRACKING_URI="https://mlflow-XXXXX.run.app/",\
    ALLOWED_ORIGINS="*",\
    DEV_BYPASS="false"
```

The Dockerfile copies a pre-trained SetFit classifier model (no build-time training needed).

### 3. MCP Server (Cloud Run)

```bash
cd backend/mcp
gcloud run deploy mcp \
  --source . \
  --region us-central1 \
  --port 8000 \
  --allow-unauthenticated \
  --memory 4Gi \
  --set-env-vars \
    MONGO_URI="mongodb+srv://...",\
    MLFLOW_TRACKING_URI="https://mlflow-XXXXX.run.app/",\
    GCS_BUCKET_NAME="supply-chain-pipeline",\
    GOOGLE_APPLICATION_CREDENTIALS="/app/gcp-key.json"
```

### 4. MLflow (Cloud Run)

Deployed as a standalone Cloud Run service for experiment tracking and model registry.

### 5. Infrastructure (Terraform)

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

Provisions: Artifact Registry, Cloud Run services, Cloud SQL, IAM roles, networking, Secret Manager.

### 6. Airflow Pipeline (Docker Compose)

```bash
docker compose up -d
# Access Airflow UI at http://localhost:8080 (airflow/airflow)
# Enable the supply_chain_pipeline DAG
```

## Local Development

### Backend

```bash
cd backend
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000 --reload-exclude 'venv/*'
```

### MCP Server (separate terminal)

```bash
cd backend/mcp
pip install -r requirements.txt
PORT=8001 python server.py
```

Set `MCP_SERVER_URL=http://localhost:8001/mcp` in `.env` for local dev.

### Frontend

```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

Create `frontend/.env`:
```
VITE_CLERK_PUBLISHABLE_KEY=pk_test_...
VITE_API_BASE_URL=http://localhost:8000
```

Ensure backend CORS allows port 5173: set `ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000` in root `.env`.

## Environment Variables

### Root `.env` (backend reads this)

| Variable | Description |
|----------|-------------|
| `MONGO_URI` | MongoDB Atlas connection string |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `MCP_SERVER_URL` | MCP server URL (local: `http://localhost:8001/mcp`) |
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `CLERK_JWKS_URL` | Clerk JWKS endpoint for JWT verification |
| `ALLOWED_ORIGINS` | CORS origins (comma-separated) |
| `DEV_BYPASS` | Skip Clerk auth locally (`true`/`false`) |
| `GCS_BUCKET_NAME` | GCS bucket for data and model artifacts |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account key |

### Frontend `.env`

| Variable | Description |
|----------|-------------|
| `VITE_CLERK_PUBLISHABLE_KEY` | Clerk publishable key |
| `VITE_API_BASE_URL` | Backend API URL |

## AI Chat System

The platform has two chat interfaces:

### Floating Chat (`/api/chat`) - LangGraph + MCP
- Uses a **SetFit classifier** to route queries to specialized agents (inventory, forecasting)
- Agents use **LangGraph** with **Claude Haiku** and **MCP tools** for real-time data access
- Tools: `summarize_inventory`, `get_product_at_store`, `check_restocking_needs`, `smart_predict_demand`

### AI Assistant Page (`/api/gemini-chat`) - Direct Tool Calling
- Uses **Claude Sonnet** with native tool calling (no LangGraph)
- Supports inventory checks, low stock alerts, reorder suggestions, order placement, and order history
- Conversation history persisted in MongoDB with multi-conversation support

## ML Pipeline

### Two-Model Strategy
- **XGBoost**: Global model trained on 29 features with Bayesian tuning (50 Optuna trials)
- **Prophet**: Per-series models with external regressors and tuning (30 trials)
- Best model auto-promoted to MLflow `@champion` alias with 5% improvement gate

### Data Quality
- Schema validation with Great Expectations
- Anomaly detection: missingness, z-score outliers, date gaps
- Bias detection at training and inference time across multiple slices

### CI/CD
- **ML Pipeline** (`ml_pipeline.yml`): Triggered on DVC file changes. Runs: data split, Optuna tuning, training, bias check, SHAP analysis, model promotion.
- **Rollback** (`rollback.yml`): Manual workflow to restore previous model version.
- **Airflow Deploy** (`deploy-airflow.yml`): Builds and deploys Airflow to Cloud Run via Terraform.
- **Frontend CI** (`frontend-ci.yml`): TypeCheck + lint + build on PRs.

## Testing

```bash
# Backend unit tests (Airflow pipeline)
cd airflow && python3 -m pytest tests/ -v   # 42 tests

# Frontend
cd frontend && npm run lint && npm run typecheck && npm run build
```

## License

Apache License 2.0
