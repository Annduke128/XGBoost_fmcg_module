"""Weekly training pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.shared.features.encode import make_categorical
from ml.shared.features.feature_defs import ALL_FEATURES, CAT_COLS
from ml.shared.schema import validate_columns
from ml.shared.utils.metrics import mae, wape
from ml.training.data.lead_time import assign_lead_time
from ml.training.data.promo_depth import compute_promo_depth
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
    daily_data_path: str | None = None  # Optional daily data for daily features
    model_dir: str = "artifacts/models"
    factors_dir: str = "artifacts/factors"
    val_weeks: int = 4
    test_weeks: int = 4
    max_model_versions: int = 8
    n_optuna_trials: int = 50
    run_tuning: bool = True
    holiday_csv: str | None = None
    use_daily_features: bool = False
    use_growth_features: bool = False
    use_kalman_factors: bool = False


def _add_daily_features(
    df: pd.DataFrame, daily_df: pd.DataFrame, holiday_csv: str | None = None
) -> pd.DataFrame:
    """Compute daily-level features and merge into weekly DataFrame."""
    from ml.training.features.daily_features import (
        add_daily_time_features,
        aggregate_daily_to_weekly,
    )
    from ml.training.data.daily_aggregate import (
        add_daily_lag_rolling_ema,
        aggregate_lag_features_to_weekly,
    )

    # Daily time features → weekly aggregation
    daily_df = add_daily_time_features(daily_df, holiday_csv=holiday_csv)
    weekly_time = aggregate_daily_to_weekly(daily_df)

    # Daily lag/rolling/EMA → weekly last values
    daily_df = add_daily_lag_rolling_ema(daily_df)
    # Need week column for aggregation
    daily_df["week"] = (
        pd.to_datetime(daily_df["date"]).dt.to_period("W").apply(lambda p: p.start_time)
    )
    weekly_lag = aggregate_lag_features_to_weekly(daily_df)

    # Merge daily-derived features into weekly DataFrame
    merge_keys = ["sku_id", "branch_id", "week"]
    df = df.merge(weekly_time, on=merge_keys, how="left", suffixes=("", "_daily"))
    df = df.merge(weekly_lag, on=merge_keys, how="left", suffixes=("", "_lag"))

    return df


def _add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add growth features if category/sub_category columns exist."""
    from ml.training.features.growth_features import build_growth_features

    if "category" in df.columns and "sub_category" in df.columns:
        df = build_growth_features(df)
    return df


def _apply_kalman_factors(df: pd.DataFrame, factors_dir: str) -> pd.DataFrame:
    """Apply Kalman seasonal and promo factors if state files exist."""
    from ml.training.factors.seasonal_kalman import (
        apply_seasonal_factors,
        create_seasonal_store,
    )
    from ml.training.factors.promo_kalman import (
        apply_promo_factors,
        create_promo_store,
    )
    from ml.training.factors.kalman_filter import KalmanFactorStore

    factors_path = Path(factors_dir)

    # Seasonal factor
    seasonal_path = factors_path / "seasonal_factors.json"
    if seasonal_path.exists():
        seasonal_store = KalmanFactorStore.load(seasonal_path)
    else:
        seasonal_store = create_seasonal_store()
    df = apply_seasonal_factors(df, seasonal_store)

    # Promo factor
    promo_path = factors_path / "promo_factors.json"
    if promo_path.exists():
        promo_store = KalmanFactorStore.load(promo_path)
    else:
        promo_store = create_promo_store()
    df = apply_promo_factors(df, promo_store)

    return df


def _update_kalman_factors(
    val: pd.DataFrame, preds: np.ndarray, factors_dir: str
) -> None:
    """Update Kalman factors with validation predictions vs actuals."""
    from ml.training.factors.seasonal_kalman import (
        create_seasonal_store,
        update_seasonal_factors,
    )
    from ml.training.factors.promo_kalman import (
        create_promo_store,
        update_promo_factors,
    )
    from ml.training.factors.kalman_filter import KalmanFactorStore

    factors_path = Path(factors_dir)
    factors_path.mkdir(parents=True, exist_ok=True)

    val_with_preds = val.copy()
    val_with_preds["pred_base"] = preds

    # Update seasonal
    seasonal_path = factors_path / "seasonal_factors.json"
    if seasonal_path.exists():
        seasonal_store = KalmanFactorStore.load(seasonal_path)
    else:
        seasonal_store = create_seasonal_store()
    update_seasonal_factors(seasonal_store, val_with_preds)
    seasonal_store.save(seasonal_path)

    # Update promo
    promo_path = factors_path / "promo_factors.json"
    if promo_path.exists():
        promo_store = KalmanFactorStore.load(promo_path)
    else:
        promo_store = create_promo_store()
    update_promo_factors(promo_store, val_with_preds)
    promo_store.save(promo_path)


def run_train_weekly(cfg: TrainWeeklyConfig | None = None) -> dict:
    """Execute full weekly training pipeline.

    Steps:
        1. Load & validate data
        2. Stockout censor & impute
        3. Compute lead time
        4. Build weekly features (time/lag/rolling/EMA)
        5. [Optional] Add daily-aggregated features
        6. [Optional] Add growth features
        7. Time split
        8. [Optional] Apply Kalman factors
        9. Encode categoricals
        10. Baseline benchmark
        11. Optuna tuning (optional)
        12. Train final XGBoost Poisson
        13. [Optional] Update Kalman factors
        14. Save artifact & rotate

    Returns:
        Dict with metrics, version, and model info.
    """
    if cfg is None:
        cfg = TrainWeeklyConfig()

    # 1. Load data
    df = pd.read_parquet(cfg.data_path)
    df = compute_promo_depth(df)
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

    # 4. Weekly features
    df = build_all_features(df)

    # 5. Daily features (optional)
    if cfg.use_daily_features and cfg.daily_data_path:
        daily_df = pd.read_parquet(cfg.daily_data_path)
        df = _add_daily_features(df, daily_df, holiday_csv=cfg.holiday_csv)

    # 6. Growth features (optional)
    if cfg.use_growth_features:
        df = _add_growth_features(df)

    df = df.dropna(subset=["lag_1"])  # drop rows without lag history

    # 7. Split
    train, val, test = time_split(df, cfg.val_weeks, cfg.test_weeks)

    # 8. Kalman factors (optional)
    if cfg.use_kalman_factors:
        train = _apply_kalman_factors(train, cfg.factors_dir)
        val = _apply_kalman_factors(val, cfg.factors_dir)

    # 9. Encode
    available_features = [c for c in ALL_FEATURES if c in df.columns]
    available_cats = [c for c in CAT_COLS if c in train.columns]
    train = make_categorical(train, available_cats)
    val = make_categorical(val, available_cats)

    X_train, y_train = train[available_features], train["units"].values.astype(float)
    X_val, y_val = val[available_features], val["units"].values.astype(float)

    # 10. Baseline
    baseline_preds = seasonal_naive(train, val)
    baseline_wape = wape(y_val, baseline_preds)

    # 11. Tuning (optional)
    best_params = None
    if cfg.run_tuning:
        best_params = run_optuna_study(
            X_train, y_train, train["week"], n_trials=cfg.n_optuna_trials
        )

    # 12. Train final model
    model = train_xgb_poisson(X_train, y_train, X_val, y_val, params=best_params)
    preds = model.predict(X_val)
    model_wape = wape(y_val, preds)
    model_mae = mae(y_val, preds)

    # 13. Update Kalman factors (optional)
    if cfg.use_kalman_factors:
        _update_kalman_factors(val, preds, cfg.factors_dir)

    # 14. Save
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
        "use_daily_features": cfg.use_daily_features,
        "use_growth_features": cfg.use_growth_features,
        "use_kalman_factors": cfg.use_kalman_factors,
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
