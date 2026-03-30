"""Weekly forecast pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ml.shared.features.encode import make_categorical
from ml.shared.features.feature_defs import ALL_FEATURES, CAT_COLS
from ml.shared.schema import validate_columns
from ml.training.data.lead_time import assign_lead_time
from ml.training.data.scenario import build_scenarios
from ml.training.features.build_features import build_all_features
from ml.training.models.model_registry import load_latest, load_metadata


@dataclass
class ForecastConfig:
    """Configuration for weekly forecast pipeline."""

    data_path: str = "data/weekly_sales.parquet"
    model_dir: str = "artifacts/models"
    output_path: str = "artifacts/forecast/weekly_forecast.parquet"
    horizons: list[int] = field(default_factory=lambda: [1, 2, 4])
    scenarios: list[str] = field(default_factory=lambda: ["A", "B"])


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


def _predict_for_horizon(
    model: object,
    df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    latest_week = df["week"].max()
    target_week = latest_week + pd.Timedelta(weeks=horizon)
    latest_rows = df[df["week"] == latest_week].copy()
    if latest_rows.empty:
        return pd.DataFrame(
            columns=["sku_id", "branch_id", "week", "horizon", "forecast_units"]
        )
    X_pred = latest_rows[feature_cols]
    preds = model.predict(X_pred)
    latest_rows = latest_rows[["sku_id", "branch_id"]].copy()
    latest_rows["week"] = target_week
    latest_rows["horizon"] = horizon
    latest_rows["forecast_units"] = np.clip(preds, a_min=0, a_max=None)
    return latest_rows


def run_forecast(cfg: ForecastConfig | None = None) -> pd.DataFrame:
    """Run weekly forecast pipeline for all scenarios and horizons."""
    if cfg is None:
        cfg = ForecastConfig()

    model = load_latest(cfg.model_dir)
    metadata = load_metadata(cfg.model_dir)
    feature_cols = metadata.get("feature_cols", ALL_FEATURES)

    df = pd.read_parquet(cfg.data_path)
    df = validate_columns(df)
    df["week"] = pd.to_datetime(df["week"])

    scenarios = build_scenarios(df)
    selected = {k: v for k, v in scenarios.items() if k in cfg.scenarios}

    outputs: list[pd.DataFrame] = []
    for scenario_key, scenario_df in selected.items():
        scenario_df = _ensure_ema_sales(scenario_df)
        scenario_df = assign_lead_time(scenario_df)
        scenario_df = build_all_features(scenario_df)
        scenario_df = scenario_df.dropna(subset=["lag_1"])

        available_features = [c for c in feature_cols if c in scenario_df.columns]
        available_cats = [c for c in CAT_COLS if c in scenario_df.columns]
        scenario_df = make_categorical(scenario_df, available_cats)

        for horizon in cfg.horizons:
            preds = _predict_for_horizon(
                model, scenario_df, horizon, available_features
            )
            if preds.empty:
                continue
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
