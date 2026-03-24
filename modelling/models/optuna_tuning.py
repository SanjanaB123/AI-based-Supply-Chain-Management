"""
optuna_tuning.py  —  pipeline version 2.0  (local machine)
Hyperparameter tuning for XGBoost and Prophet using Optuna.

Strategy
--------
  XGBoost : Optuna TPE sampler, 50 trials, objective = mean MAE across
            all 5 walk-forward folds on the train set.
  Prophet  : Optuna TPE sampler, 20 trials (slower to train), objective =
            mean MAE across a 3-series sample to keep runtime reasonable.
            Full per-series refit happens after best params are found.

Storage
-------
  SQLite backend — studies persist across runs so tuning can be resumed.
  File: optuna_studies.db  (created in working directory)

Usage
-----
    python optuna_tuning.py                      # tune both models
    python optuna_tuning.py --model xgboost      # XGBoost only
    python optuna_tuning.py --model prophet      # Prophet only
    python optuna_tuning.py --model xgboost --trials 100
    python optuna_tuning.py --model xgboost --fresh   # delete old study and restart
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise ImportError("Install optuna: pip install optuna") from e

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("Install xgboost: pip install xgboost") from e

try:
    from prophet import Prophet
except ImportError as e:
    raise ImportError("Install prophet: pip install prophet") from e

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_roll_std_7", "sales_ewm_28",
    "demand_forecast_lag1",
    "price_vs_competitor", "effective_price",
    "Holiday/Promotion", "Discount", "discount_x_holiday",
    "dow", "month", "is_weekend",
    "Inventory Level", "stockout_flag", "lead_time_demand",
    "Lead Time Days", "reorder_event",
    "Category_enc", "Region_enc", "Seasonality_enc",
    "y_pred_baseline", "series_enc",
]

NO_SCALE_COLS = [
    "Holiday/Promotion", "Discount", "discount_x_holiday",
    "dow", "month", "is_weekend", "stockout_flag", "reorder_event",
    "Category_enc", "Region_enc", "Seasonality_enc",
    "Lead Time Days", "series_enc",
]

PROPHET_REGRESSORS = [
    "Holiday/Promotion", "Discount", "price_vs_competitor",
    "effective_price", "discount_x_holiday",
    "Inventory Level", "stockout_flag", "lead_time_demand",
    "demand_forecast_lag1",
]

# Series used for Prophet tuning (using 5 representative series keeps it fast)
PROPHET_TUNE_SERIES = [
    "S001_P0001", "S001_P0007", "S002_P0012",
    "S003_P0006", "S004_P0020",
]

TARGET_COL  = "y"
DATE_COL    = "as_of_date"
WEIGHT_COL  = "sample_weight"

DEFAULT_DATA_DIR    = Path("data/splits")
DEFAULT_OUTPUT_DIR  = Path("data/models")
DEFAULT_REPORTS_DIR = Path("reports")
STORAGE_PATH        = "sqlite:///optuna_studies.db"

# Walk-forward config (mirrors data_splitting.py)
N_FOLDS     = 5
VAL_MONTHS  = 2
GAP_DAYS    = 14


# ── Shared helpers ────────────────────────────────────────────────────────────
def encode_series(train: pd.DataFrame, *others: pd.DataFrame) -> tuple:
    if "series_id" not in train.columns:
        train = train.copy()
        train["series_id"] = (
            train["Store ID"].astype(str) + "_" + train["Product ID"].astype(str)
        )
    mapping = {sid: i for i, sid in enumerate(sorted(train["series_id"].unique()))}
    results = []
    for df in (train, *others):
        df = df.copy()
        if "series_id" not in df.columns:
            df["series_id"] = (
                df["Store ID"].astype(str) + "_" + df["Product ID"].astype(str)
            )
        df["series_enc"] = df["series_id"].map(mapping).fillna(-1).astype(int)
        results.append(df)
    return tuple(results)


def scale(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple:
    scale_cols = [c for c in X_train.columns if c not in NO_SCALE_COLS]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val[scale_cols]   = scaler.transform(X_val[scale_cols])
    return X_train, X_val, scaler


def make_folds(
    df: pd.DataFrame,
    n_splits:   int = N_FOLDS,
    val_months: int = VAL_MONTHS,
    gap_days:   int = GAP_DAYS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate (train_fold, val_fold) pairs for walk-forward CV."""
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    min_date        = df[DATE_COL].min()
    max_date        = df[DATE_COL].max()
    total_days      = (max_date - min_date).days
    val_window_days = val_months * 30
    step_days       = (total_days - val_window_days - gap_days) // n_splits

    folds = []
    for i in range(n_splits):
        train_end  = min_date + pd.Timedelta(days=step_days * (i + 1))
        val_start  = train_end + pd.Timedelta(days=gap_days)
        val_end    = val_start + pd.Timedelta(days=val_window_days)
        if val_end > max_date:
            val_end = max_date
        if val_start >= max_date:
            break

        tr = df[df[DATE_COL] <= train_end]
        vl = df[(df[DATE_COL] > val_start) & (df[DATE_COL] <= val_end)]

        if len(tr) > 0 and len(vl) > 0:
            folds.append((tr, vl))

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST TUNING
# ─────────────────────────────────────────────────────────────────────────────
def xgboost_objective(trial: optuna.Trial, train_df: pd.DataFrame) -> float:
    """
    Optuna objective for XGBoost.
    Trains on each walk-forward fold with the trial's hyperparameters.
    Returns mean MAE across all folds (Optuna minimises this).
    """
    params = {
        "objective":            "reg:squarederror",
        "eval_metric":          "mae",
        "random_state":         42,
        "n_jobs":               -1,
        # ── Parameters being tuned ──────────────────────────────────────────
        "max_depth":            trial.suggest_int("max_depth", 3, 10),
        "learning_rate":        trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":         trial.suggest_int("n_estimators", 100, 800),
        "subsample":            trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":     trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":     trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":            trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":           trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma":                trial.suggest_float("gamma", 0.0, 5.0),
    }
    early_stopping_rounds = 30

    folds = make_folds(train_df)
    fold_maes = []

    for fold_idx, (tr, vl) in enumerate(folds):
        tr, vl = encode_series(tr, vl)

        available = [c for c in FEATURE_COLS if c in tr.columns]
        X_tr = tr[available]
        y_tr = tr[TARGET_COL]
        w_tr = tr[WEIGHT_COL] if WEIGHT_COL in tr.columns else None
        X_vl = vl[available]
        y_vl = vl[TARGET_COL]

        X_tr, X_vl, _ = scale(X_tr, X_vl)

        fit_params = {k: v for k, v in params.items()
                      if k not in ("eval_metric",)}
        model = xgb.XGBRegressor(
            **fit_params,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False,
        )

        y_pred = model.predict(X_vl).clip(min=0)
        mae    = mean_absolute_error(y_vl, y_pred)
        fold_maes.append(mae)

        # Optuna pruning — stop unpromising trials early
        trial.report(mae, fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(fold_maes))


def tune_xgboost(
    train_df:    pd.DataFrame,
    n_trials:    int  = 50,
    fresh:       bool = False,
    output_dir:  Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict:
    log.info("=== XGBoost Optuna Tuning ===")
    log.info("Trials: %d   Folds per trial: %d   Storage: %s",
             n_trials, N_FOLDS, STORAGE_PATH)

    study_name = "xgboost_supply_chain_v2"

    if fresh:
        try:
            optuna.delete_study(study_name=study_name, storage=STORAGE_PATH)
            log.info("Deleted existing study '%s'", study_name)
        except Exception:
            pass

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE_PATH,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    completed_so_far = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed_so_far)

    if remaining == 0:
        log.info("Study already has %d completed trials — skipping tuning.",
                 completed_so_far)
    else:
        log.info("Running %d new trials (%d already completed) ...", remaining, completed_so_far)
        study.optimize(
            lambda trial: xgboost_objective(trial, train_df),
            n_trials=remaining,
            show_progress_bar=True,
        )

    best_params = study.best_params
    best_mae    = study.best_value

    log.info("Best MAE  : %.4f", best_mae)
    log.info("Best params:")
    for k, v in best_params.items():
        log.info("  %-25s = %s", k, v)

    # ── Save best params ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "xgboost" / "best_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)

    best_params_full = {
        **best_params,
        "objective":    "reg:squarederror",
        "eval_metric":  "mae",
        "random_state": 42,
        "n_jobs":       -1,
    }
    with open(params_path, "w") as f:
        json.dump(best_params_full, f, indent=2)
    log.info("Best params saved to %s", params_path)

    # ── Save tuning report ────────────────────────────────────────────────────
    reports_dir.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()

    report = {
        "model":             "xgboost",
        "study_name":        study_name,
        "n_trials_total":    len(study.trials),
        "n_trials_complete": len([t for t in study.trials
                                  if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_trials_pruned":   len([t for t in study.trials
                                  if t.state == optuna.trial.TrialState.PRUNED]),
        "best_mae":          round(best_mae, 4),
        "best_params":       best_params_full,
        "top_5_trials":      (
            trials_df[trials_df["state"] == "COMPLETE"]
            .nsmallest(5, "value")[["number", "value"]]
            .rename(columns={"number": "trial", "value": "mae"})
            .round(4)
            .to_dict("records")
        ),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    report_path = reports_dir / "xgboost_tuning_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Tuning report saved to %s", report_path)
    log.info("=== XGBoost Tuning Done ===")
    return best_params_full


# ─────────────────────────────────────────────────────────────────────────────
# PROPHET TUNING
# ─────────────────────────────────────────────────────────────────────────────
def prophet_objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> float:
    """
    Optuna objective for Prophet.
    Evaluates on PROPHET_TUNE_SERIES only to keep each trial fast (~5 series).
    Returns mean MAE across all tuning series.
    """
    params = {
        "changepoint_prior_scale":   trial.suggest_float(
            "changepoint_prior_scale", 0.001, 0.5, log=True
        ),
        "seasonality_prior_scale":   trial.suggest_float(
            "seasonality_prior_scale", 0.01, 20.0, log=True
        ),
        "seasonality_mode":          trial.suggest_categorical(
            "seasonality_mode", ["additive", "multiplicative"]
        ),
        "changepoint_range":         trial.suggest_float(
            "changepoint_range", 0.7, 0.95
        ),
        "n_changepoints":            trial.suggest_int(
            "n_changepoints", 10, 40
        ),
    }

    # Prep function: rename columns for Prophet
    def prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df[[DATE_COL, TARGET_COL] +
                 [r for r in PROPHET_REGRESSORS if r in df.columns]].copy()
        out = out.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
        out["ds"] = pd.to_datetime(out["ds"])
        out["y"]  = out["y"].clip(lower=0)
        for r in PROPHET_REGRESSORS:
            if r in out.columns:
                out[r] = out[r].fillna(0)
        return out

    # Add series_id if missing
    for df in [train_df, val_df]:
        if "series_id" not in df.columns:
            df["series_id"] = (
                df["Store ID"].astype(str) + "_" + df["Product ID"].astype(str)
            )

    series_maes = []

    for sid in PROPHET_TUNE_SERIES:
        tr = train_df[train_df["series_id"] == sid]
        vl = val_df[val_df["series_id"] == sid]

        if len(tr) < 30 or len(vl) == 0:
            continue

        tr_p = prep(tr)
        vl_p = prep(vl)

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95,
            **params,
        )

        for r in PROPHET_REGRESSORS:
            if r in tr_p.columns:
                model.add_regressor(r)

        model.fit(tr_p)

        forecast = model.predict(vl_p.drop(columns=["y"]))
        y_pred   = forecast["yhat"].clip(lower=0).values
        y_true   = vl_p["y"].values

        mae = mean_absolute_error(y_true, y_pred)
        series_maes.append(mae)

    if not series_maes:
        raise optuna.exceptions.TrialPruned()

    return float(np.mean(series_maes))


def tune_prophet(
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    n_trials:    int  = 20,
    fresh:       bool = False,
    output_dir:  Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict:
    log.info("=== Prophet Optuna Tuning ===")
    log.info("Trials: %d   Tuning on %d series: %s",
             n_trials, len(PROPHET_TUNE_SERIES), PROPHET_TUNE_SERIES)

    study_name = "prophet_supply_chain_v2"

    if fresh:
        try:
            optuna.delete_study(study_name=study_name, storage=STORAGE_PATH)
            log.info("Deleted existing study '%s'", study_name)
        except Exception:
            pass

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE_PATH,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    completed_so_far = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed_so_far)

    if remaining == 0:
        log.info("Study already has %d completed trials — skipping tuning.",
                 completed_so_far)
    else:
        log.info("Running %d new trials (%d already completed) ...", remaining, completed_so_far)
        study.optimize(
            lambda trial: prophet_objective(trial, train_df, val_df),
            n_trials=remaining,
            show_progress_bar=True,
        )

    best_params = study.best_params
    best_mae    = study.best_value

    log.info("Best MAE  : %.4f", best_mae)
    log.info("Best params:")
    for k, v in best_params.items():
        log.info("  %-30s = %s", k, v)

    # ── Save best params ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "prophet" / "best_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info("Best params saved to %s", params_path)

    # ── Save tuning report ────────────────────────────────────────────────────
    reports_dir.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()

    report = {
        "model":             "prophet",
        "study_name":        study_name,
        "tuning_series":     PROPHET_TUNE_SERIES,
        "n_trials_total":    len(study.trials),
        "n_trials_complete": len([t for t in study.trials
                                  if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_mae":          round(best_mae, 4),
        "best_params":       best_params,
        "top_5_trials":      (
            trials_df[trials_df["state"] == "COMPLETE"]
            .nsmallest(5, "value")[["number", "value"]]
            .rename(columns={"number": "trial", "value": "mae"})
            .round(4)
            .to_dict("records")
        ),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    report_path = reports_dir / "prophet_tuning_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Tuning report saved to %s", report_path)
    log.info("=== Prophet Tuning Done ===")
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(
    model:       str,
    data_dir:    Path,
    output_dir:  Path,
    reports_dir: Path,
    xgb_trials:  int,
    prophet_trials: int,
    fresh:       bool,
) -> None:
    log.info("=== Optuna Hyperparameter Tuning  v2.0 ===")
    log.info("Model: %s   Storage: %s", model, STORAGE_PATH)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df   = pd.read_parquet(data_dir / "val.parquet")
    log.info("Loaded  train=%d  val=%d rows", len(train_df), len(val_df))

    if model in ("xgboost", "both"):
        tune_xgboost(
            train_df    = train_df,
            n_trials    = xgb_trials,
            fresh       = fresh,
            output_dir  = output_dir,
            reports_dir = reports_dir,
        )

    if model in ("prophet", "both"):
        tune_prophet(
            train_df    = train_df,
            val_df      = val_df,
            n_trials    = prophet_trials,
            fresh       = fresh,
            output_dir  = output_dir,
            reports_dir = reports_dir,
        )

    log.info("")
    log.info("=== All tuning complete ===")
    log.info("Best params saved to:")
    if model in ("xgboost", "both"):
        log.info("  %s", output_dir / "xgboost" / "best_params.json")
    if model in ("prophet", "both"):
        log.info("  %s", output_dir / "prophet"  / "best_params.json")
    log.info("")
    log.info("To retrain with best params, pass --params-path to the model script,")
    log.info("or load best_params.json directly in xgboost_model.py / prophet_model.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for XGBoost and Prophet"
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "prophet", "both"],
        default="both",
        help="Which model to tune (default: both)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train/val/test parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write best_params.json files",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory to write tuning report JSON files",
    )
    parser.add_argument(
        "--xgb-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for XGBoost (default: 50)",
    )
    parser.add_argument(
        "--prophet-trials",
        type=int,
        default=20,
        help="Number of Optuna trials for Prophet (default: 20)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing Optuna study and start fresh",
    )
    args = parser.parse_args()

    main(
        model          = args.model,
        data_dir       = args.data_dir,
        output_dir     = args.output_dir,
        reports_dir    = args.reports_dir,
        xgb_trials     = args.xgb_trials,
        prophet_trials = args.prophet_trials,
        fresh          = args.fresh,
    )