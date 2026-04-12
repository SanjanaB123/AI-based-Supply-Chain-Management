"""
sensitivity_analysis.py
Generates SHAP feature importance and Optuna hyperparameter sensitivity plots.
Both are logged to MLflow and saved to reports/ for the CI artifact upload.

Run after training:
    python sensitivity_analysis.py --model-name xgboost-supply-chain
"""
from __future__ import annotations
import argparse, json, logging, os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from data_splitting import get_X_y, FEATURE_COLS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPORTS_DIR = Path("reports")


# ── SHAP ──────────────────────────────────────────────────────────────────────

def run_shap_analysis(model_name: str, data_dir: str, stage: str = "Production") -> None:
    # Set MLflow tracking URI explicitly
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model   = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
    test_df = pd.read_parquet(Path(data_dir) / "test.parquet")

    X_test, _ = get_X_y(test_df)

    # Sample for speed — SHAP on the full test set can be slow for XGBoost with many rows
    sample = X_test.sample(min(500, len(X_test)), random_state=42)

    # Use the underlying sklearn-compatible model if available (faster TreeExplainer)
    try:
        raw_model = model._model_impl.python_model.model   # MLflow pyfunc wrapper
        explainer = shap.TreeExplainer(raw_model)
        sv = explainer.shap_values(sample)
    except AttributeError:
        # Fallback to KernelExplainer (model-agnostic, slower)
        log.warning("TreeExplainer unavailable — using KernelExplainer (slower).")
        explainer = shap.KernelExplainer(model.predict, shap.sample(sample, 50))
        sv = explainer.shap_values(sample)

    # ── Bar summary (mean |SHAP|) ──────────────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8, max(5, len(FEATURE_COLS) * 0.28)))
    top = importance_df.head(20)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#1D9E75")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature importance — {model_name}\n(top 20, mean absolute SHAP)", fontsize=11)
    plt.tight_layout()
    bar_path = REPORTS_DIR / "shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Beeswarm plot ─────────────────────────────────────────────────────────
    shap.summary_plot(sv, sample, show=False, max_display=20)
    beeswarm_path = REPORTS_DIR / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save importance table as JSON
    importance_path = REPORTS_DIR / "feature_importance.json"
    importance_path.write_text(importance_df.to_json(orient="records", indent=2))

    # Log to MLflow
    with mlflow.start_run(run_name=f"shap-{model_name}", nested=True):
        mlflow.log_artifact(str(bar_path))
        mlflow.log_artifact(str(beeswarm_path))
        mlflow.log_artifact(str(importance_path))
        # Log top-3 features as metrics for quick dashboard view
        for i, row in importance_df.head(3).iterrows():
            mlflow.log_metric(f"top{i+1}_feature_shap", row["mean_abs_shap"])

    log.info("SHAP analysis complete. Plots saved to reports/")


# ── Hyperparameter sensitivity ────────────────────────────────────────────────

def run_hyperparameter_sensitivity(study_name: str, model_key: str) -> None:
    """
    Reads the Optuna study and plots how each hyperparameter correlates
    with validation MAE. Requires the optuna_studies.db to be present.
    """
    import optuna
    from optuna.visualization.matplotlib import (
        plot_param_importances,
        plot_parallel_coordinate,
    )

    storage = "sqlite:///optuna_studies.db"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        log.warning("Could not load Optuna study '%s': %s — skipping sensitivity.", study_name, e)
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Parameter importance (how much each HP moves the objective)
    fig = plot_param_importances(study)
    imp_path = REPORTS_DIR / f"{model_key}_hp_importance.png"
    fig.savefig(imp_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Parallel coordinate (shows joint HP effect)
    fig = plot_parallel_coordinate(study)
    pc_path = REPORTS_DIR / f"{model_key}_hp_parallel.png"
    fig.savefig(pc_path, dpi=150, bbox_inches="tight")
    plt.close()

    with mlflow.start_run(run_name=f"hp-sensitivity-{model_key}", nested=True):
        mlflow.log_artifact(str(imp_path))
        mlflow.log_artifact(str(pc_path))

    log.info("Hyperparameter sensitivity plots saved for %s.", model_key)


# ── Model comparison visualisation ───────────────────────────────────────────

def plot_model_comparison(reports_dir: Path = REPORTS_DIR) -> None:
    """
    Bar chart comparing XGBoost vs Prophet (and LSTM if present) on
    test MAE / RMSE / R². This is the 'results visualisation' the guidelines require.
    """
    model_keys = ["xgboost", "prophet", "lstm"]
    records = []
    for key in model_keys:
        p = reports_dir / f"{key}_report.json"
        if p.exists():
            r = json.loads(p.read_text())
            records.append({
                "model":     key.upper(),
                "Test MAE":  r.get("test_mae",  0),
                "Test RMSE": r.get("test_rmse", 0),
                "Test R²":   r.get("test_r2",   0),
            })

    if not records:
        log.warning("No model reports found — skipping comparison plot.")
        return

    df   = pd.DataFrame(records).set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#1D9E75", "#3B8BD4", "#E8593C"]

    for ax, metric in zip(axes, ["Test MAE", "Test RMSE", "Test R²"]):
        df[metric].plot(kind="bar", ax=ax, color=colors[:len(df)], edgecolor="white")
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        # Annotate bars
        for bar in ax.patches:
            ax.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("Model comparison — test set performance", fontsize=13)
    plt.tight_layout()
    out = REPORTS_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    with mlflow.start_run(run_name="model-comparison", nested=True):
        mlflow.log_artifact(str(out))

    log.info("Model comparison chart saved to %s", out)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",  default="xgboost-supply-chain")
    parser.add_argument("--study-name",  default="xgboost-supply-chain-hpo")
    parser.add_argument("--model-key",   default="xgboost")
    parser.add_argument("--data-dir",    default="data/splits")
    parser.add_argument("--stage",       default="Production")
    args = parser.parse_args()

    run_shap_analysis(args.model_name, args.data_dir, args.stage)
    run_hyperparameter_sensitivity(args.study_name, args.model_key)
    plot_model_comparison()