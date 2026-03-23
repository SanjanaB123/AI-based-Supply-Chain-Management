#!/usr/bin/env python3
"""
Bias detection and mitigation for supply chain data pipeline.

Analyzes performance across different slices (Category, Region, Weather, Seasonality, Promotion)
using baseline predictions and generates mitigation recommendations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return None
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def analyze_slice_performance(
    df: pd.DataFrame, 
    slice_col: str, 
    target_col: str = "y", 
    pred_col: str = "y_pred_baseline"
) -> Dict[str, Any]:
    """Analyze baseline performance for each slice of a categorical feature."""
    results = {}
    
    if slice_col not in df.columns:
        log.warning(f"Slice column '{slice_col}' not found in dataframe")
        return results
    
    # Overall performance
    overall_mask = df[target_col].notna() & df[pred_col].notna()
    overall_mae = calculate_mae(df.loc[overall_mask, target_col], df.loc[overall_mask, pred_col])
    overall_mape = calculate_mape(df.loc[overall_mask, target_col], df.loc[overall_mask, pred_col])
    
    results["_overall"] = {
        "sample_count": len(df[overall_mask]),
        "mae": float(overall_mae) if not np.isnan(overall_mae) else None,
        "mape": float(overall_mape) if overall_mape is not None and not np.isnan(overall_mape) else None
    }
    
    # Performance per slice
    for slice_value in df[slice_col].unique():
        if pd.isna(slice_value):
            slice_key = "NULL"
        else:
            slice_key = str(slice_value)
            
        slice_data = df[df[slice_col] == slice_value]
        slice_mask = slice_data[target_col].notna() & slice_data[pred_col].notna()
        
        if len(slice_data[slice_mask]) == 0:
            continue
            
        slice_mae = calculate_mae(
            slice_data.loc[slice_mask, target_col], 
            slice_data.loc[slice_mask, pred_col]
        )
        slice_mape = calculate_mape(
            slice_data.loc[slice_mask, target_col], 
            slice_data.loc[slice_mask, pred_col]
        )
        
        # Calculate bias metrics
        mae_ratio = slice_mae / overall_mae if overall_mae > 0 else float('inf')
        sample_ratio = len(slice_data) / len(df)
        
        results[slice_key] = {
            "sample_count": len(slice_data),
            "sample_ratio": float(sample_ratio),
            "mae": float(slice_mae) if not np.isnan(slice_mae) else None,
            "mape": float(slice_mape) if slice_mape is not None and not np.isnan(slice_mape) else None,
            "mae_ratio": float(mae_ratio) if not np.isnan(mae_ratio) and mae_ratio != float('inf') else None,
            "bias_flag": mae_ratio > 1.2  # Flag if MAE is 20% worse than overall
        }
    
    return results


def generate_mitigation_plan(slice_results: Dict[str, Any], min_samples: int = 100) -> Dict[str, Any]:
    """Generate mitigation recommendations based on slice analysis."""
    mitigation_plan = {}
    
    for slice_col, results in slice_results.items():
        if slice_col == "_overall" or not results:
            continue
            
        plan = {
            "slice_column": slice_col,
            "recommendations": [],
            "sample_weights": {},
            "priority": "low"
        }
        
        overall_mae = results.get("_overall", {}).get("mae", 0)
        high_bias_slices = []
        low_sample_slices = []
        
        for slice_key, metrics in results.items():
            if slice_key == "_overall":
                continue
                
            sample_count = metrics.get("sample_count", 0)
            mae_ratio = metrics.get("mae_ratio", 1.0)
            bias_flag = metrics.get("bias_flag", False)
            
            # Check for low sample size
            if sample_count < min_samples:
                low_sample_slices.append(slice_key)
                plan["recommendations"].append(
                    f"Low sample count for {slice_key} ({sample_count}). Consider data collection or oversampling."
                )
            
            # Check for high bias
            if bias_flag:
                high_bias_slices.append(slice_key)
                plan["recommendations"].append(
                    f"High bias detected for {slice_key} (MAE ratio: {mae_ratio:.2f}). Consider feature engineering."
                )
            
            # Calculate sample weights (inverse frequency weighting)
            if sample_count > 0:
                weight = 1.0 / max(sample_count / len(results), 0.01)  # Cap minimum frequency
                plan["sample_weights"][slice_key] = float(weight)
        
        # Determine priority
        if high_bias_slices and low_sample_slices:
            plan["priority"] = "high"
        elif high_bias_slices or low_sample_slices:
            plan["priority"] = "medium"
        
        mitigation_plan[slice_col] = plan
    
    return mitigation_plan


def generate_bias_report(
    features_path: str,
    output_dir: str,
    slice_features: List[str] = None
) -> str:
    """
    Generate comprehensive bias analysis report.
    
    Args:
        features_path: Path to features parquet file
        output_dir: Directory to save bias report
        slice_features: List of columns to slice by (default: common categorical features)
        
    Returns:
        Path to generated bias report JSON file
    """
    log.info(f"Starting bias analysis for {features_path}")
    
    # Load data
    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df)} rows for bias analysis")
    
    # Default slice features if not provided
    if slice_features is None:
        slice_features = ["Holiday/Promotion", "Weather Condition", "Seasonality", "Store ID", "Product ID"]
    
    # Filter to available columns
    available_slices = [col for col in slice_features if col in df.columns]
    log.info(f"Analyzing bias across {len(available_slices)} features: {available_slices}")
    
    # Analyze each slice feature
    slice_results = {}
    for slice_col in available_slices:
        log.info(f"Analyzing slice: {slice_col}")
        slice_results[slice_col] = analyze_slice_performance(df, slice_col)
    
    # Generate mitigation plan
    mitigation_plan = generate_mitigation_plan(slice_results)
    
    # Compile report
    report = {
        "metadata": {
            "features_path": features_path,
            "generated_at": pd.Timestamp.now().isoformat(),
            "total_rows": len(df),
            "slice_features_analyzed": available_slices,
            "baseline_model": "sales_lag_7"
        },
        "slice_performance": slice_results,
        "mitigation_plan": mitigation_plan,
        "summary": {
            "total_slices_analyzed": sum(len(results) - 1 for results in slice_results.values()),  # -1 for _overall
            "high_bias_slices": sum(
                1 for results in slice_results.values() 
                for metrics in results.values() 
                if metrics.get("bias_flag", False)
            ),
            "low_sample_slices": sum(
                1 for results in slice_results.values() 
                for metrics in results.values() 
                if metrics.get("sample_count", 0) < 100
            )
        }
    }
    
    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / "bias_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save mitigation plan separately
    mitigation_file = output_path / "bias_mitigation_plan.json"
    with open(mitigation_file, "w") as f:
        json.dump(mitigation_plan, f, indent=2, default=str)
    
    log.info(f"Bias report saved to {report_file}")
    log.info(f"Mitigation plan saved to {mitigation_file}")
    
    return str(report_file)


def calculate_sample_weights(
    df: pd.DataFrame, 
    weight_column: str = "Store ID", 
    min_weight: float = 0.1,
    max_weight: float = 10.0
) -> pd.Series:
    """
    Calculate inverse frequency sample weights for bias mitigation.
    
    Args:
        df: Input dataframe
        weight_column: Column to calculate weights for
        min_weight: Minimum weight to prevent extreme values
        max_weight: Maximum weight to prevent extreme values
        
    Returns:
        Series of sample weights
    """
    if weight_column not in df.columns:
        log.warning(f"Weight column '{weight_column}' not found, returning uniform weights")
        return pd.Series(1.0, index=df.index)
    
    # Calculate frequencies
    freq = df[weight_column].value_counts(normalize=True)
    
    # Calculate inverse frequency weights
    weights = df[weight_column].map(lambda x: 1.0 / max(freq.get(x, 0), 1e-6))
    
    # Clip weights to prevent extreme values
    weights = weights.clip(min_weight, max_weight)
    
    log.info(f"Calculated sample weights for {weight_column}: min={weights.min():.3f}, max={weights.max():.3f}")
    
    return weights
