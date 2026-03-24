# ML Supply Chain Forecasting CI/CD Pipeline

## Overview

This project implements a complete machine learning CI/CD pipeline for supply chain demand forecasting with automated model selection, MLflow tracking, and intelligent deployment.

## Features

- **Automated Model Training**: XGBoost and Prophet models with hyperparameter optimization
- **MLflow Integration**: Full experiment tracking, model registry, and artifact logging
- **Smart Model Selection**: Automated winner selection based on test MAE performance
- **Intelligent Deployment**: 5% improvement gate prevents bad model deployments
- **Model-Agnostic Inference**: Always loads current production model
- **Cloud Integration**: Google Cloud Storage via DVC for data versioning
- **GitHub Actions**: Complete CI/CD pipeline with automated runs
- **Critical Bug Fixes**: Proper MLflow environment, no target leakage, robust error handling

## Project Structure

```
ci_cd-main/
├── .github/workflows/
│   └── ml_pipeline.yml          # GitHub Actions CI/CD pipeline
├── modelling/
│   ├── xgboost_model.py         # XGBoost model with MLflow logging
│   ├── prophet_model.py         # Prophet model with MLflow logging
│   └── lstm_model.py            # LSTM model (PyTorch) - deprecated
├── data/
│   ├── features/                # Processed features (DVC tracked)
│   ├── splits/                  # Train/val/test splits
│   └── models/                  # Trained models
├── reports/                     # Model reports and comparisons
├── optuna_studies.db           # Hyperparameter optimization studies
├── select_model.py             # Automated model selection and promotion
├── inference.py                # Model-agnostic inference
├── trigger_github_workflow.py  # API workflow trigger
├── trigger_workflow.sh         # Bash workflow trigger
├── data_splitting.py          # Chronological data splitting with leakage prevention
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (local)
└── .dvc/                       # DVC configuration
```

## Setup

### Prerequisites

- Python 3.12+
- Google Cloud Account (for data storage)
- GitHub Account with Personal Access Token
- MLflow Server (already deployed)

### Environment Setup

1. **Clone and Setup**
```bash
git clone <repository-url>
cd ci_cd-main
```

2. **Create Environment File**
```bash
# Create .env file (add to .gitignore)
echo "GITHUB_TOKEN=your_github_pat_here" > .env
echo "MLFLOW_TRACKING_URI=https://mlflow-833456981899.us-central1.run.app/" >> .env
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup DVC and Google Cloud**
```bash
# Configure DVC with Google Cloud Storage
dvc remote add -d myremote gs://your-bucket-name
dvc push
```

5. **GitHub Secrets**
Add these secrets to your GitHub repository:
- `GCP_SA_KEY`: Google Cloud service account JSON
- `GITHUB_TOKEN`: Personal access token (optional, for API triggers)

## Running the Pipeline

### Option 1: GitHub Actions (Recommended)

```bash
# Trigger via API
python trigger_github_workflow.py

# Or trigger via Bash
./trigger_workflow.sh

# Check status of ML pipeline only
python trigger_github_workflow.py status
```

### Option 2: Local Development

```bash
# Train models locally
python modelling/xgboost_model.py --data-dir data/splits --output-dir data/models/xgboost --reports-dir reports
python modelling/prophet_model.py --data-dir data/splits --output-dir data/models/prophet --reports-dir reports

# Select best model
python select_model.py

# Make predictions
python inference.py
```

## Model Selection Process

### Automated Selection Logic

1. **Train Models**: Both XGBoost and Prophet train with MLflow logging
2. **Compare Performance**: Test MAE is compared from JSON reports
3. **Select Winner**: Lower MAE wins
4. **Promotion Gate**: Only promotes if 5% better than current production
5. **Deploy**: Winner moved to "Production" stage in MLflow

### Model Registry Stages

- **None**: Newly trained models
- **Production**: Active serving model (single winner)
- **Archived**: Old production models and non-winners

### Example Selection Output

```
Model Selection Analysis
==================================================
XGBoost  test MAE: 15.2341
Prophet  test MAE: 17.8923
Winner: XGBoost (MAE=15.2341, +14.86% vs Prophet)

Promoting to MLflow Production
==================================================
Promoted xgboost-supply-chain version 3 to Production
Production MAE: 15.2341
Archived old version 2
```

## Inference

### Model-Agnostic Predictions

The inference script automatically loads whatever model is currently in Production:

```python
from inference import load_production_model, predict_demand

# Load production model (automatically detects winner)
model, model_name = load_production_model()

# Make predictions
predictions = predict_demand(model, input_data, model_name)
```

### Sample Inference

```bash
# Set environment
export MLFLOW_TRACKING_URI=https://mlflow-833456981899.us-central1.run.app/

# Run inference
python inference.py
```

Output:
```
Loading Production Model
========================================
Attempting to load: xgboost-supply-chain/Production
Successfully loaded xgboost-supply-chain from Production

Making Predictions with xgboost-supply-chain
========================================
Input shape: (100, 15)
Predictions shape: 100
Sample predictions: [145.23, 167.89, 123.45, 189.67, 156.78]
```

## MLflow Integration

### Experiment Tracking

- **Experiment Name**: `supply-chain-forecasting`
- **Models Registered**: `xgboost-supply-chain`, `prophet-supply-chain`
- **Logged Artifacts**: Models, reports, metrics, feature importance

### Access MLflow

MLflow Server: https://mlflow-833456981899.us-central1.run.app/

- **Experiments Tab**: View training runs and compare metrics
- **Models Tab**: See model registry and production versions
- **Artifacts**: Download trained models and reports

## Safety Features & Critical Fixes

### Recent Critical Bug Fixes

1. **MLflow Environment Fixed**: All model training steps now have `MLFLOW_TRACKING_URI`
   - Before: Only 2 of 6 steps had MLflow server connection
   - After: All steps properly log to your MLflow server

2. **Target Leakage Prevention**: Removed `sample_weight` from features
   - Before: Models learned from sample weights (indirect target info)
   - After: Sample weights extracted separately for model fitting only

3. **Robust Error Handling**: Fixed `best_model.txt` file handling
   - Before: Silent failures when file missing
   - After: Proper error checking and GitHub environment variables

4. **Clean Status Monitoring**: Workflow status checks only ML pipeline
   - Before: All repo workflows (noisy)
   - After: Specific ML pipeline workflow only

### Deployment Gates

- **5% Improvement Gate**: New models must be 5% better than current production
- **Automatic Rollback**: Old models archived (not deleted) for easy rollback
- **Performance Monitoring**: Full metrics tracking and comparison

### Error Handling

- **Graceful Failures**: Pipeline continues if individual models fail
- **Detailed Logging**: Comprehensive error messages and debugging info
- **Validation**: Input validation and data quality checks

## Model Performance

### Current Models

| Model | Test MAE | Test RMSE | Test R² | Features |
|-------|----------|-----------|---------|----------|
| XGBoost | ~15.2 | ~24.3 | ~0.60 | 15 engineered features |
| Prophet | ~17.7 | ~24.8 | ~0.58 | Time series + regressors |

### Feature Engineering

- **Lag Features**: sales_lag_7, sales_lag_14, sales_lag_28
- **Rolling Means**: sales_roll_mean_7, sales_roll_mean_14, sales_roll_mean_28
- **Calendar Features**: day of week, month, holidays, promotions
- **Supply Chain**: lead time, discounts, categories, regions

### Data Splitting Safety

- **Chronological Split**: No random splitting, strict date boundaries
- **14-Day Gaps**: Prevents lag feature leakage across train/val/test
- **Walk-Forward Validation**: Time-aware cross-validation
- **No Target Leakage**: Sample weights handled separately from features

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
    paths:
      - '**.dvc'
  workflow_dispatch:

jobs:
  ml_pipeline:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
      - name: Set up Python 3.12
      - name: Install dependencies
      - name: Authenticate to Google Cloud
      - name: Configure DVC
      - name: Pull data from GCS
      - name: Split data
      - name: Tune XGBoost hyperparameters
      - name: Tune Prophet hyperparameters
      - name: Train XGBoost model
      - name: Train Prophet model
      - name: Select best model and promote to production
      - name: Upload models and reports
      - name: Upload Optuna studies
```

### Pipeline Triggers

- **Automatic**: On push to main branch affecting .dvc files
- **Manual**: Via GitHub Actions "workflow_dispatch"
- **API**: Via `trigger_github_workflow.py` script

## Monitoring and Maintenance

### Model Monitoring

1. **MLflow Dashboard**: Track experiments and model performance
2. **Production Metrics**: Monitor inference performance and drift
3. **Automated Reports**: Model comparison reports saved to `reports/`

### Maintenance Tasks

- **Data Updates**: Push new data via DVC (`dvc push`)
- **Model Retraining**: Trigger pipeline weekly or on data changes
- **Performance Review**: Check MLflow for model drift

## Deployment Architecture

```
GitHub Actions → Model Training → MLflow Registry → Production Model → Inference API
     ↓                    ↓                ↓                    ↓
   CI/CD Pipeline    XGBoost/Prophet    Model Selection    Model-Agnostic
   Automation        Hyperparameter     Performance Gate    Inference
                      Optimization      5% Improvement      Loading
```

## Configuration

### Environment Variables

```bash
# Required
MLFLOW_TRACKING_URI=https://mlflow-833456981899.us-central1.run.app/

# Optional (for API triggers)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# DVC Remote (in .dvc/config)
[remote "myremote"]
    url = gs://your-bucket-name
```

### Model Parameters

**XGBoost Hyperparameters** (optimized via Optuna):
- n_estimators: 100-1000
- max_depth: 3-10
- learning_rate: 0.01-0.3
- subsample: 0.6-1.0

**Prophet Hyperparameters**:
- changepoint_prior_scale: 0.001-0.5
- seasonality_prior_scale: 0.01-10
- holidays_prior_scale: 0.01-10
- seasonality_mode: ['additive', 'multiplicative']

## Troubleshooting

### Common Issues

1. **MLflow Connection Error**
   - Check `MLFLOW_TRACKING_URI` environment variable
   - Verify MLflow server is accessible

2. **DVC Authentication**
   - Ensure Google Cloud credentials are properly set
   - Check service account permissions

3. **Model Selection Error**
   - Verify both model reports exist in `reports/`
   - Check JSON report structure and MAE keys

4. **GitHub Actions Failures**
   - Check secrets are properly configured
   - Verify GCP service account permissions

### Debug Mode

```bash
# Enable verbose logging
export MLFLOW_TRACKING_URI=https://mlflow-833456981899.us-central1.run.app/
python -v select_model.py
```

## API Reference

### Model Selection Script

```python
from select_model import select_best_model, promote_to_production

# Select best model
best_model, best_mae = select_best_model()

# Promote to production
promote_to_production(best_model, best_mae)
```

### Inference Script

```python
from inference import load_production_model, predict_demand

# Load production model
model, model_name = load_production_model()

# Make predictions
predictions = predict_demand(model, input_data, model_name)
```

### Workflow Trigger Script

```python
from trigger_github_workflow import trigger_workflow, check_workflow_status

# Trigger ML pipeline
trigger_workflow()

# Check ML pipeline status only
check_workflow_status()
```

## Success Metrics

### Pipeline Performance

- **Automated Training**: Models train automatically on data changes
- **Smart Selection**: Best model automatically selected and deployed
- **Safety Gates**: Bad models prevented from reaching production
- **Full Tracking**: Complete experiment tracking and model registry
- **Model-Agnostic**: Inference works with any production model
- **Critical Fixes**: All major bugs resolved for production safety

### Business Impact

- **Improved Accuracy**: Automated model selection improves performance
- **Faster Deployment**: CI/CD reduces deployment time from days to minutes
- **Risk Reduction**: Gates prevent bad model deployments
- **Visibility**: Full MLflow tracking provides model transparency

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **MLflow**: For excellent experiment tracking and model registry
- **Optuna**: For hyperparameter optimization
- **DVC**: For data versioning and cloud storage
- **GitHub Actions**: For CI/CD automation
- **XGBoost & Prophet**: For powerful modeling capabilities

---

Ready to deploy intelligent ML models automatically?

MLflow Server: https://mlflow-833456981899.us-central1.run.app/
GitHub Actions: https://github.com/SanjanaB123/model_ci_cd/actions

---

### Recent Critical Fixes Applied

This README reflects the current production-ready state with all critical bugs resolved:

1. **MLflow Environment**: All training steps now properly connect to your MLflow server
2. **Target Leakage Prevention**: Sample weights removed from feature set
3. **Error Handling**: Robust file handling and environment variable management  
4. **Clean Monitoring**: Workflow status checks specific pipeline only

The pipeline is now production-safe and ready for automated ML deployments!
