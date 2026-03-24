"""
bias_detection.py
Evaluates the production model across meaningful data slices and flags
disparities. Generates a JSON report and PNG visualisation logged to MLflow.

Run after training:
    python bias_detection.py --model-name xgboost-supply-chain --data-dir data/splits
"""
from __future__ import annotations
import argparse, json, logging, os, joblib
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_splitting import get_X_y, FEATURE_COLS, LABEL_COL

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Columns to slice on + human-readable decode maps (update to match your encodings)
SLICE_COLS = {
    "Category_enc":    {0: "Electronics", 1: "Clothing", 2: "Food", 3: "Home"},
    "Region_enc":      {0: "North", 1: "South", 2: "East", 3: "West"},
    "Seasonality_enc": {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"},
}

DISPARITY_THRESHOLD = 0.25   # flag any slice whose MAE exceeds overall MAE by >25%
REPORTS_DIR = Path("reports")


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "n": len(y_true)}


def evaluate_slices(
    model,
    test_df:    pd.DataFrame,
    model_name: str,
    scaler,
    slice_cols: dict = SLICE_COLS,
) -> dict:
    X_test, y_test = get_X_y(test_df)
    
    # Apply scaling if scaler available and model needs it
    if scaler is not None and "prophet" not in model_name.lower():
        from data_splitting import SCALE_COLS
        scale_cols = [c for c in SCALE_COLS if c in X_test.columns]
        X_test_scaled = X_test.copy()
        X_test_scaled[scale_cols] = scaler.transform(X_test_scaled[scale_cols])
        log.info("Applied scaling to %d columns for XGBoost model", len(scale_cols))
    else:
        log.info("No scaling needed for Prophet model or scaler unavailable")
        X_test_scaled = X_test
    
    y_pred = model.predict(X_test_scaled)

    overall = metrics(y_test, y_pred)
    log.info("Overall — MAE=%.4f  RMSE=%.4f  R²=%.4f  n=%d", overall["mae"], overall["rmse"], overall["r2"], overall["n"])

    results = {"overall": overall, "slices": {}, "flagged": []}

    for col, label_map in slice_cols.items():
        if col not in test_df.columns:
            log.warning("Slice column '%s' not in test data — skipping.", col)
            continue

        col_results = {}
        for code, name in label_map.items():
            mask = test_df[col] == code
            if mask.sum() < 10:   # too few rows — skip
                continue
            m = metrics(y_test[mask], y_pred[mask])
            col_results[name] = m

            # Flag if disparity vs overall MAE exceeds threshold
            if overall["mae"] > 0:
                disparity = (m["mae"] - overall["mae"]) / overall["mae"]
                if disparity > DISPARITY_THRESHOLD:
                    results["flagged"].append({
                        "slice_col": col,
                        "slice_val": name,
                        "mae":       m["mae"],
                        "overall_mae": overall["mae"],
                        "disparity_pct": round(disparity * 100, 1),
                    })
                    log.warning(
                        "BIAS FLAGGED — %s=%s  MAE=%.4f (+%.1f%% vs overall)",
                        col, name, m["mae"], disparity * 100,
                    )

        results["slices"][col] = col_results

    return results


def plot_slice_maes(results: dict, output_path: Path) -> None:
    """Bar chart of MAE per slice group, with overall baseline marked."""
    overall_mae = results["overall"]["mae"]
    fig, axes = plt.subplots(1, len(results["slices"]), figsize=(5 * len(results["slices"]), 5))
    if len(results["slices"]) == 1:
        axes = [axes]

    for ax, (col, slices) in zip(axes, results["slices"].items()):
        names = list(slices.keys())
        maes  = [slices[n]["mae"] for n in names]
        colors = [
            "#E24B4A" if m > overall_mae * (1 + DISPARITY_THRESHOLD) else "#1D9E75"
            for m in maes
        ]
        ax.bar(names, maes, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(overall_mae, color="#888780", linestyle="--", linewidth=1, label=f"Overall MAE={overall_mae:.2f}")
        ax.set_title(col.replace("_enc", "").replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Test MAE")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)

    fig.suptitle("MAE by data slice  (red = disparity > 25%)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Slice plot saved to %s", output_path)


def suggest_mitigations(flagged: list[dict]) -> list[str]:
    """Return human-readable mitigation suggestions for each flagged slice."""
    suggestions = []
    for f in flagged:
        suggestions.append(
            f"{f['slice_col']}={f['slice_val']} (MAE +{f['disparity_pct']}%): "
            f"Consider (1) up-weighting this slice via sample_weight, "
            f"(2) training a slice-specific sub-model, or "
            f"(3) adding slice-specific lag/rolling features."
        )
    return suggestions


def run_bias_detection(model_name: str, data_dir: str, stage: str = "Production", scaler_path: str = None) -> dict:
    # Set MLflow tracking URI explicitly
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model from MLflow
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
    
    # Load from local disk only — no MLflow download
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
        log.info("Loaded scaler from local path: %s", scaler_path)
    else:
        log.warning("Scaler not found at %s — results will be unscaled", scaler_path)

    test_df = pd.read_parquet(Path(data_dir) / "test.parquet")
    results = evaluate_slices(model, test_df, model_name, scaler)
    results["mitigations"] = suggest_mitigations(results["flagged"])
    results["model_name"]  = model_name
    results["bias_threshold_pct"] = DISPARITY_THRESHOLD * 100

    # Save JSON report
    report_path = REPORTS_DIR / "bias_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    log.info("Bias report saved to %s", report_path)

    # Save and log plot
    plot_path = REPORTS_DIR / "bias_slices.png"
    plot_slice_maes(results, plot_path)

    # Log everything to MLflow
    with mlflow.start_run(run_name=f"bias-detection-{model_name}", nested=True):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("bias_threshold_pct", DISPARITY_THRESHOLD * 100)
        mlflow.log_metric("overall_mae",  results["overall"]["mae"])
        mlflow.log_metric("overall_rmse", results["overall"]["rmse"])
        mlflow.log_metric("overall_r2",   results["overall"]["r2"])
        mlflow.log_metric("n_slices_flagged", len(results["flagged"]))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(plot_path))

        # Set a tag so CI can read it
        mlflow.set_tag("bias_check_passed", str(len(results["flagged"]) == 0))

    # Exit non-zero if flagged — lets the CI step fail and block deployment
    if results["flagged"]:
        log.warning(
            "%d slice(s) exceed the disparity threshold. "
            "Review reports/bias_report.json and apply mitigations.",
            len(results["flagged"]),
        )
    else:
        log.info("Bias check PASSED — no slices exceed the %.0f%% disparity threshold.",
                 DISPARITY_THRESHOLD * 100)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="xgboost-supply-chain")
    parser.add_argument("--data-dir",   default="data/splits")
    parser.add_argument("--stage",      default="Production")
    parser.add_argument("--scaler-path", default=None,
                        help="Path to local scaler.pkl file (avoids MLflow download)")
    parser.add_argument("--fail-on-bias", action="store_true",
                        help="Exit with code 1 if any slice is flagged (blocks CI deployment)")
    args = parser.parse_args()

    results = run_bias_detection(args.model_name, args.data_dir, args.stage, args.scaler_path)
    if args.fail_on_bias and results["flagged"]:
        raise SystemExit(1)