"""Weekly forecast pipeline — recursive multi-step rollout.

Generates demand forecasts for requested horizons (default 1, 2, 4 weeks)
and scenarios (A = no promo, B = 50 % discount) using a trained XGBoost
model loaded from the model registry.

Key design:
    * ``_rollout_forecast`` steps sequentially h=1..max(horizons) so that
      each step's prediction feeds as ``lag_1`` into the next step.
    * Intermediate steps (e.g. h=3) are computed internally for correct
      lag propagation but excluded from output unless explicitly requested.
    * Predictions are clipped >= 0 (Poisson target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ml.shared.features.encode import make_categorical
from ml.shared.features.feature_defs import ALL_FEATURES, CAT_COLS
from ml.shared.schema import validate_columns
from ml.training.data.lead_time import assign_lead_time
from ml.training.data.promo_depth import compute_promo_depth
from ml.training.data.scenario import build_scenarios
from ml.training.features.build_features import build_all_features
from ml.training.models.model_registry import load_latest, load_metadata


@dataclass
class ForecastConfig:
    """Configuration for weekly forecast pipeline."""

    data_path: str = "data/weekly_sales.parquet"
    daily_data_path: str | None = None
    model_dir: str = "artifacts/models"
    factors_dir: str = "artifacts/factors"
    output_path: str = "artifacts/forecast/weekly_forecast.parquet"
    horizons: list[int] = field(default_factory=lambda: [1, 2, 4])
    scenarios: list[str] = field(default_factory=lambda: ["A", "B"])
    holiday_csv: str | None = None
    use_daily_features: bool = False
    use_growth_features: bool = False
    use_kalman_factors: bool = False


def _ensure_ema_sales(df: pd.DataFrame) -> pd.DataFrame:
    if "ema_sales_8w" in df.columns:
        return df
    df = df.sort_values(["sku_id", "branch_id", "week"]).copy()
    df["ema_sales_8w"] = (
        df.groupby(["sku_id", "branch_id"])["units"]
        .shift(1)
        .ewm(span=8, adjust=False)
        .mean()
    )
    return df


def _add_extended_features(
    df: pd.DataFrame, cfg: ForecastConfig, daily_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add daily/growth/Kalman features if enabled."""
    # Daily features
    if cfg.use_daily_features and daily_df is not None:
        from ml.training.features.daily_features import (
            add_daily_time_features,
            aggregate_daily_to_weekly,
        )
        from ml.training.data.daily_aggregate import (
            add_daily_lag_rolling_ema,
            aggregate_lag_features_to_weekly,
        )

        daily_proc = add_daily_time_features(daily_df, holiday_csv=cfg.holiday_csv)
        weekly_time = aggregate_daily_to_weekly(daily_proc)

        daily_proc = add_daily_lag_rolling_ema(daily_proc)
        daily_proc["week"] = (
            pd.to_datetime(daily_proc["date"])
            .dt.to_period("W")
            .apply(lambda p: p.start_time)
        )
        weekly_lag = aggregate_lag_features_to_weekly(daily_proc)

        merge_keys = ["sku_id", "branch_id", "week"]
        df = df.merge(weekly_time, on=merge_keys, how="left", suffixes=("", "_daily"))
        df = df.merge(weekly_lag, on=merge_keys, how="left", suffixes=("", "_lag"))

    # Growth features
    if cfg.use_growth_features:
        if "category" in df.columns and "sub_category" in df.columns:
            from ml.training.features.growth_features import build_growth_features

            df = build_growth_features(df)

    # Kalman factors
    if cfg.use_kalman_factors:
        from ml.training.factors.seasonal_kalman import (
            apply_seasonal_factors,
            create_seasonal_store,
        )
        from ml.training.factors.promo_kalman import (
            apply_promo_factors,
            create_promo_store,
        )
        from ml.training.factors.kalman_filter import KalmanFactorStore

        factors_path = Path(cfg.factors_dir)

        seasonal_path = factors_path / "seasonal_factors.json"
        if seasonal_path.exists():
            seasonal_store = KalmanFactorStore.load(seasonal_path)
        else:
            seasonal_store = create_seasonal_store()
        df = apply_seasonal_factors(df, seasonal_store)

        promo_path = factors_path / "promo_factors.json"
        if promo_path.exists():
            promo_store = KalmanFactorStore.load(promo_path)
        else:
            promo_store = create_promo_store()
        df = apply_promo_factors(df, promo_store)

    return df


def _recompute_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute lag_1/2/4, roll_4_mean, ema_4 from units per group.

    Used after appending a predicted row so subsequent horizons see
    updated lag features that incorporate prior predictions.
    """
    df = df.sort_values(["sku_id", "branch_id", "week"]).copy()
    grp = df.groupby(["sku_id", "branch_id"])["units"]

    for lag in [1, 2, 4]:
        df[f"lag_{lag}"] = grp.shift(lag)

    shifted = grp.shift(1)
    df["roll_4_mean"] = shifted.rolling(4, min_periods=1).mean()
    df["ema_4"] = shifted.ewm(span=4, adjust=False).mean()

    return df


def _rollout_forecast(
    model: object,
    df: pd.DataFrame,
    horizons: list[int],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Recursive multi-step forecast: step 1 → max(horizons).

    At each step h, a placeholder row for the target week T+h is
    appended to the working history so that ``_recompute_lags`` yields
    correct lag values (e.g. lag_1 at T+h = units at T+h-1, which is
    the prediction from the previous step). Intermediate steps (e.g.
    h=3) are computed for correct lag propagation but only requested
    horizons appear in the output.
    """
    _EMPTY_COLS = ["sku_id", "branch_id", "week", "horizon", "forecast_units"]

    latest_week = df["week"].max()
    latest_rows = df[df["week"] == latest_week].copy()
    if latest_rows.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)

    max_h = max(horizons)
    horizon_set = set(horizons)
    collected: list[pd.DataFrame] = []

    # Working copy: we append predicted rows step by step
    working = df.copy()

    for h in range(1, max_h + 1):
        target_week = latest_week + pd.Timedelta(weeks=h)

        # Template: copy non-lag features from the current latest row
        template = working[working["week"] == working["week"].max()].copy()
        template["week"] = target_week
        template["units"] = 0.0  # placeholder — lags don't depend on own row

        # Temporarily append and recompute lags so the target row gets
        # lag_1 = units[T+h-1] (the previous prediction)
        temp = pd.concat([working, template], ignore_index=True)
        temp = _recompute_lags(temp)

        # Predict from the target row's (now correct) lag features
        pred_rows = temp[temp["week"] == target_week].copy()
        avail = [c for c in feature_cols if c in pred_rows.columns]
        X_pred = pred_rows[avail]
        preds = np.clip(model.predict(X_pred), a_min=0, a_max=None)

        # Collect output only for requested horizons
        if h in horizon_set:
            out = pred_rows[["sku_id", "branch_id"]].copy()
            out["week"] = target_week
            out["horizon"] = h
            out["forecast_units"] = preds
            collected.append(out)

        # Finalize: set units = prediction for future lag computation
        template["units"] = preds
        working = pd.concat([working, template], ignore_index=True)

    if not collected:
        return pd.DataFrame(columns=_EMPTY_COLS)
    return pd.concat(collected, ignore_index=True)


def run_forecast(cfg: ForecastConfig | None = None) -> pd.DataFrame:
    """Run weekly forecast pipeline for all scenarios and horizons."""
    if cfg is None:
        cfg = ForecastConfig()

    model = load_latest(cfg.model_dir)
    metadata = load_metadata(cfg.model_dir)
    feature_cols = metadata.get("feature_cols", ALL_FEATURES)

    df = pd.read_parquet(cfg.data_path)
    df = compute_promo_depth(df)
    df = validate_columns(df)
    df["week"] = pd.to_datetime(df["week"])

    # Load daily data if needed
    daily_df = None
    if cfg.use_daily_features and cfg.daily_data_path:
        daily_df = pd.read_parquet(cfg.daily_data_path)

    scenarios = build_scenarios(df)
    selected = {k: v for k, v in scenarios.items() if k in cfg.scenarios}

    outputs: list[pd.DataFrame] = []
    for scenario_key, scenario_df in selected.items():
        scenario_df = _ensure_ema_sales(scenario_df)
        scenario_df = assign_lead_time(scenario_df)
        scenario_df = build_all_features(scenario_df)

        # Extended features
        scenario_df = _add_extended_features(scenario_df, cfg, daily_df)

        scenario_df = scenario_df.dropna(subset=["lag_1"])

        available_features = [c for c in feature_cols if c in scenario_df.columns]
        available_cats = [c for c in CAT_COLS if c in scenario_df.columns]
        scenario_df = make_categorical(scenario_df, available_cats)

        preds = _rollout_forecast(model, scenario_df, cfg.horizons, available_features)
        if not preds.empty:
            preds["scenario"] = scenario_key
            outputs.append(preds)

    result = (
        pd.concat(outputs, ignore_index=True)
        if outputs
        else pd.DataFrame(
            columns=[
                "sku_id",
                "branch_id",
                "week",
                "horizon",
                "scenario",
                "forecast_units",
            ]
        )
    )

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


if __name__ == "__main__":
    run_forecast()
