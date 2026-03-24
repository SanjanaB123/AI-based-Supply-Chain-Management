#!/usr/bin/env python3
"""
inference.py
Model-agnostic inference using MLflow Production models
"""

import mlflow
import os
import pandas as pd
import json
from pathlib import Path

# Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", 
    "https://mlflow-833456981899.us-central1.run.app/"
)

def load_production_model():
    """Load the current production model from MLflow"""
    print("Loading Production Model")
    print("=" * 40)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Try XGBoost first, then Prophet
    models_to_try = [
        "xgboost-supply-chain",
        "prophet-supply-chain"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"Attempting to load: {model_name}/Production")
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
            print(f"Successfully loaded {model_name} from Production")
            return model, model_name
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    
    raise Exception("No production model found in MLflow")

def predict_demand(model, input_data, model_name):
    """Make predictions using the loaded model"""
    print(f"\nMaking Predictions with {model_name}")
    print("=" * 40)
    
    # Ensure input_data is a DataFrame
    if isinstance(input_data, (dict, list)):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data.copy()
    
    print(f"Input shape: {input_df.shape}")
    print(f"Input columns: {list(input_df.columns)}")
    
    # Make predictions
    predictions = model.predict(input_df)
    
    # Ensure non-negative predictions (demand can't be negative)
    if hasattr(predictions, 'clip'):
        predictions = predictions.clip(lower=0)
    elif isinstance(predictions, (list, np.ndarray)):
        if isinstance(predictions, np.ndarray):
            predictions = np.maximum(predictions, 0)
        else:
            predictions = [max(0, p) for p in predictions]
    
    print(f"Predictions shape: {len(predictions)}")
    print(f"Sample predictions: {predictions[:5] if len(predictions) > 5 else predictions}")
    
    return predictions

def get_model_info(model_name):
    """Get information about the production model"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get latest production version
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            return None
        
        prod_version = prod_versions[0]
        run_id = prod_version.run_id
        run = client.get_run(run_id)
        
        info = {
            "model_name": model_name,
            "version": prod_version.version,
            "stage": prod_version.stage,
            "run_id": run_id,
            "creation_time": run.info.start_time,
            "test_mae": run.data.metrics.get("test_mae"),
            "test_rmse": run.data.metrics.get("test_rmse"),
            "test_r2": run.data.metrics.get("test_r2"),
            "model_type": run.data.params.get("model_type"),
            "pipeline_version": run.data.params.get("pipeline_version"),
        }
        
        return info
        
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None

def main():
    """Main inference function"""
    try:
        # Load production model
        model, model_name = load_production_model()
        
        # Get model information
        model_info = get_model_info(model_name)
        if model_info:
            print(f"\nModel Information:")
            print(f"   Name: {model_info['model_name']}")
            print(f"   Version: {model_info['version']}")
            print(f"   Type: {model_info['model_type']}")
            print(f"   Pipeline Version: {model_info['pipeline_version']}")
            print(f"   Test MAE: {model_info['test_mae']:.4f}")
            print(f"   Test RMSE: {model_info['test_rmse']:.4f}")
            print(f"   Test R²: {model_info['test_r2']:.4f}")
        
        # Example usage with sample data
        print(f"\nSample Inference with {model_name}:")
        print("=" * 50)
        
        # Create sample input data (adjust columns as needed)
        sample_data = {
            'sales_lag_7': [100, 150, 120],
            'sales_lag_14': [95, 145, 115],
            'sales_lag_28': [90, 140, 110],
            'sales_roll_mean_7': [98, 148, 118],
            'sales_roll_mean_14': [96, 146, 116],
            'sales_roll_mean_28': [92, 142, 112],
            'dow': [1, 2, 3],
            'month': [1, 1, 1],
            'is_weekend': [0, 0, 0],
            'Holiday/Promotion': [0, 1, 0],
            'Discount': [0.1, 0.15, 0.05],
            'Inventory Level': [50, 75, 60],
            'Lead Time Days': [7, 5, 10],
            'Category_enc': [1, 2, 1],
            'Region_enc': [1, 1, 2],
            'Seasonality_enc': [1, 1, 1],
        }
        
        # Make predictions
        predictions = predict_demand(model, sample_data, model_name)
        
        print(f"Input shape: {len(sample_data['sales_lag_7'])}")
        print(f"Predictions shape: {len(predictions)}")
        print(f"Sample predictions: {predictions}")
        
        return predictions
        
    except Exception as e:
        print(f"Inference failed: {e}")
        return None

if __name__ == "__main__":
    main()
