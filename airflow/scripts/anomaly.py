#!/usr/bin/env python3
"""
Anomaly detection for supply chain data pipeline.

Detects:
- Missingness spikes: features with null% > threshold
- Outliers: sales or y values beyond P99.9 or z-score > threshold  
- Date gaps: missing days per series > threshold

Writes anomaly_report.json with flags and counts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# Default thresholds
DEFAULT_MISSINGNESS_THRESHOLD = 0.02  # 2%
DEFAULT_OUTLIER_Z_SCORE = 3.0
DEFAULT_DATE_GAP_THRESHOLD = 1  # days
NULL_PERCENTILE = 99.9


def detect_missingness_spikes(
    df: pd.DataFrame, threshold: float = DEFAULT_MISSINGNESS_THRESHOLD
) -> Dict[str, Any]:
    """Detect features with missingness above threshold."""
    missingness = {}
    anomalies = {}
    
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        missingness[col] = {
            "null_percentage": float(null_pct),
            "null_count": int(df[col].isnull().sum()),
            "total_count": len(df)
        }
        
        if null_pct > threshold:
            anomalies[col] = {
                "type": "missingness_spike",
                "null_percentage": float(null_pct),
                "threshold": threshold,
                "severity": "high" if null_pct > 0.1 else "medium"
            }
    
    return {
        "missingness_summary": missingness,
        "missingness_anomalies": anomalies
    }


def detect_outliers(
    df: pd.DataFrame, 
    z_threshold: float = DEFAULT_OUTLIER_Z_SCORE,
    percentile_threshold: float = NULL_PERCENTILE
) -> Dict[str, Any]:
    """Detect outliers in sales and target variables."""
    anomalies = {}
    outlier_summary = {}
    
    # Check numeric columns that could have outliers
    numeric_cols = ["Units Sold", "y", "Price", "Inventory Level"]
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
            
        # Remove null values for outlier detection
        clean_series = df[col].dropna()
        
        if len(clean_series) == 0:
            continue
            
        # Z-score based outliers
        z_scores = np.abs(stats.zscore(clean_series))
        z_outliers = z_scores > z_threshold
        
        # Percentile based outliers
        p_high = np.percentile(clean_series, percentile_threshold)
        p_low = np.percentile(clean_series, 100 - percentile_threshold)
        percentile_outliers = (clean_series > p_high) | (clean_series < p_low)
        
        outlier_count = np.sum(z_outliers | percentile_outliers)
        outlier_pct = outlier_count / len(clean_series)
        
        outlier_summary[col] = {
            "outlier_count": int(outlier_count),
            "outlier_percentage": float(outlier_pct),
            "z_score_outliers": int(np.sum(z_outliers)),
            "percentile_outliers": int(np.sum(percentile_outliers)),
            "total_values": len(clean_series)
        }
        
        if outlier_pct > 0.01:  # More than 1% outliers
            anomalies[col] = {
                "type": "outlier_spike",
                "outlier_percentage": float(outlier_pct),
                "z_threshold": z_threshold,
                "percentile_threshold": percentile_threshold,
                "severity": "high" if outlier_pct > 0.05 else "medium"
            }
    
    return {
        "outlier_summary": outlier_summary,
        "outlier_anomalies": anomalies
    }


def detect_date_gaps(
    df: pd.DataFrame, 
    date_col: str = "as_of_date",
    series_cols: List[str] = ["Store ID", "Product ID"],
    gap_threshold: int = DEFAULT_DATE_GAP_THRESHOLD
) -> Dict[str, Any]:
    """Detect missing days in time series per store-product combination."""
    anomalies = {}
    gap_summary = {}
    
    if date_col not in df.columns:
        log.warning(f"Date column '{date_col}' not found")
        return {"date_gap_summary": {}, "date_gap_anomalies": {}}
    
    # Convert to datetime if not already
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Group by series identifier
    for series_id, group in df.groupby(series_cols):
        # Sort by date
        group = group.sort_values(date_col)
        
        # Calculate date differences
        date_diffs = group[date_col].diff().dt.days.dropna()
        
        # Find gaps larger than threshold
        large_gaps = date_diffs[date_diffs > gap_threshold]
        
        series_key = "_".join([str(series_id) if isinstance(series_id, tuple) else str(series_id)])
        
        gap_summary[series_key] = {
            "gap_count": len(large_gaps),
            "max_gap_days": int(large_gaps.max()) if len(large_gaps) > 0 else 0,
            "total_days": len(group),
            "date_range": {
                "start": group[date_col].min().isoformat(),
                "end": group[date_col].max().isoformat()
            }
        }
        
        if len(large_gaps) > 0:
            anomalies[series_key] = {
                "type": "date_gap",
                "gap_count": len(large_gaps),
                "max_gap_days": int(large_gaps.max()),
                "gap_threshold": gap_threshold,
                "severity": "high" if len(large_gaps) > 5 else "medium"
            }
    
    return {
        "date_gap_summary": gap_summary,
        "date_gap_anomalies": anomalies
    }


def generate_anomaly_report(
    features_path: str,
    output_dir: str,
    missingness_threshold: float = DEFAULT_MISSINGNESS_THRESHOLD,
    outlier_z_threshold: float = DEFAULT_OUTLIER_Z_SCORE,
    date_gap_threshold: int = DEFAULT_DATE_GAP_THRESHOLD
) -> str:
    """
    Generate comprehensive anomaly report for the feature dataset.
    
    Args:
        features_path: Path to features parquet file
        output_dir: Directory to save anomaly report
        missingness_threshold: Threshold for missingness anomalies (percentage)
        outlier_z_threshold: Z-score threshold for outlier detection
        date_gap_threshold: Maximum allowed gap in days
        
    Returns:
        Path to generated anomaly report JSON file
    """
    log.info(f"Starting anomaly detection for {features_path}")
    
    # Load data
    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Run anomaly detection
    missingness_results = detect_missingness_spikes(df, missingness_threshold)
    outlier_results = detect_outliers(df, outlier_z_threshold)
    date_gap_results = detect_date_gaps(df, gap_threshold=date_gap_threshold)
    
    # Compile report
    report = {
        "metadata": {
            "features_path": features_path,
            "generated_at": pd.Timestamp.now().isoformat(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "thresholds": {
                "missingness": missingness_threshold,
                "outlier_z_score": outlier_z_threshold,
                "date_gap_days": date_gap_threshold
            }
        },
        "missingness": missingness_results,
        "outliers": outlier_results,
        "date_gaps": date_gap_results
    }
    
    # Summary counts
    total_anomalies = (
        len(missingness_results["missingness_anomalies"]) +
        len(outlier_results["outlier_anomalies"]) +
        len(date_gap_results["date_gap_anomalies"])
    )
    
    report["summary"] = {
        "total_anomaly_types": total_anomalies,
        "missingness_anomalies": len(missingness_results["missingness_anomalies"]),
        "outlier_anomalies": len(outlier_results["outlier_anomalies"]),
        "date_gap_anomalies": len(date_gap_results["date_gap_anomalies"]),
        "has_critical_anomalies": total_anomalies > 0
    }
    
    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / "anomaly_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    log.info(f"Anomaly report saved to {report_file}")
    log.info(f"Found {total_anomalies} anomaly types")
    
    return str(report_file)


def check_anomaly_thresholds(report_path: str, max_anomalies: int = 0) -> bool:
    """
    Check if anomalies exceed acceptable threshold.
    
    Args:
        report_path: Path to anomaly report JSON
        max_anomalies: Maximum allowed anomalies (default 0 for strict checking)
        
    Returns:
        True if anomalies are within acceptable limits, False otherwise
    """
    with open(report_path, "r") as f:
        report = json.load(f)
    
    total_anomalies = report["summary"]["total_anomaly_types"]
    
    if total_anomalies > max_anomalies:
        log.warning(
            f"Anomalies detected: {total_anomalies} (threshold: {max_anomalies})"
        )
        return False
    
    log.info(f"Anomaly check passed: {total_anomalies} anomalies found")
    return True
