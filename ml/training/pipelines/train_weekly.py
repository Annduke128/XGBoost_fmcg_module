"""Weekly training pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.shared.features.encode import make_categorical
from ml.shared.features.feature_defs import ALL_FEATURES, CAT_COLS
from ml.shared.schema import validate_columns
from ml.shared.utils.metrics import mae, wape
from ml.training.data.lead_time import assign_lead_time
from ml.training.data.split import time_split
from ml.training.data.stockout import impute_stockout
from ml.training.features.build_features import build_all_features
from ml.training.models.baselines import seasonal_naive
from ml.training.models.model_registry import rotate_versions, save_model
from ml.training.models.train_xgb import train_xgb_poisson
from ml.training.models.tune_optuna import run_optuna_study


@dataclass
class TrainWeeklyConfig:
    """Configuration for weekly training pipeline."""

    data_path: str = "data/weekly_sales.parquet"
    model_dir: str = "artifacts/models"
    val_weeks: int = 4
    test_weeks: int = 4
    max_model_versions: int = 8
    n_optuna_trials: int = 50
    run_tuning: bool = True


def run_train_weekly(cfg: TrainWeeklyConfig | None = None) -> dict:
    """Execute full weekly training pipeline.

    Steps:
        1. Load & validate data
        2. Stockout censor & impute
        3. Compute lead time
        4. Build features
        5. Time split
        6. Encode categoricals
        7. Baseline benchmark
        8. Optuna tuning (optional)
        9. Train final XGBoost Poisson
        10. Save artifact & rotate

    Returns:
        Dict with metrics, version, and model info.
    """
    if cfg is None:
        cfg = TrainWeeklyConfig()

    # 1. Load data
    df = pd.read_parquet(cfg.data_path)
    df = validate_columns(df)
    df["week"] = pd.to_datetime(df["week"])

    # 2. Stockout impute
    df = impute_stockout(df, ["sku_id", "branch_id"])

    # 3. Lead time — compute ema_sales_8w if not present
    if "ema_sales_8w" not in df.columns:
        df = df.sort_values(["sku_id", "branch_id", "week"])
        df["ema_sales_8w"] = (
            df.groupby(["sku_id", "branch_id"])["units"]
            .shift(1)
            .ewm(span=8, adjust=False)
            .mean()
        )
    df = assign_lead_time(df)

    # 4. Features
    df = build_all_features(df)
    df = df.dropna(subset=["lag_1"])  # drop rows without lag history

    # 5. Split
    train, val, test = time_split(df, cfg.val_weeks, cfg.test_weeks)

    # 6. Encode
    available_features = [c for c in ALL_FEATURES if c in df.columns]
    available_cats = [c for c in CAT_COLS if c in train.columns]
    train = make_categorical(train, available_cats)
    val = make_categorical(val, available_cats)

    X_train, y_train = train[available_features], train["units"].values.astype(float)
    X_val, y_val = val[available_features], val["units"].values.astype(float)

    # 7. Baseline
    baseline_preds = seasonal_naive(train, val)
    baseline_wape = wape(y_val, baseline_preds)

    # 8. Tuning (optional)
    best_params = None
    if cfg.run_tuning:
        best_params = run_optuna_study(
            X_train, y_train, train["week"], n_trials=cfg.n_optuna_trials
        )

    # 9. Train final model
    model = train_xgb_poisson(X_train, y_train, X_val, y_val, params=best_params)
    preds = model.predict(X_val)
    model_wape = wape(y_val, preds)
    model_mae = mae(y_val, preds)

    # 10. Save
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    metadata = {
        "version": version,
        "baseline_wape": float(baseline_wape),
        "model_wape": float(model_wape),
        "model_mae": float(model_mae),
        "best_params": best_params,
        "feature_cols": available_features,
        "n_train": len(train),
        "n_val": len(val),
    }
    model_dir = Path(cfg.model_dir)
    save_model(model, metadata, model_dir, version)
    rotate_versions(model_dir, cfg.max_model_versions)

    return metadata


if __name__ == "__main__":
    result = run_train_weekly()
    print(
        f"Training complete: WAPE={result['model_wape']:.4f} "
        f"(baseline={result['baseline_wape']:.4f})"
    )
