"""Daily lag/rolling/EMA features aggregated to weekly.

Computes daily-level lag, rolling stats, and EMA (all on shift(1)+ to avoid
leakage), then aggregates to weekly by taking the last value in each week.
"""

from __future__ import annotations

import pandas as pd


def add_daily_lag_rolling_ema(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "units",
) -> pd.DataFrame:
    """Add daily lag/rolling/EMA features. All shift(1)+ for no leakage."""
    if group_cols is None:
        group_cols = ["sku_id", "branch_id"]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(group_cols + ["date"])

    grp = df.groupby(group_cols)[target_col]
    shifted = grp.shift(1)

    # Lags (shift first)
    for lag in [1, 3, 7, 14, 28]:
        df[f"lag_{lag}d"] = grp.shift(lag)

    # Rolling means (on shifted series)
    for window in [3, 7, 14, 28]:
        df[f"roll_{window}d_mean"] = shifted.rolling(window, min_periods=window).mean()

    # Rolling stds (on shifted series)
    for window in [7, 14, 28]:
        df[f"roll_{window}d_std"] = shifted.rolling(window, min_periods=window).std()

    # EMAs (on shifted series)
    df["ema_14d"] = shifted.ewm(span=14, adjust=False).mean()
    df["ema_30d"] = shifted.ewm(span=30, adjust=False).mean()

    return df


def aggregate_lag_features_to_weekly(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate daily lag/rolling/EMA features to weekly using last value."""
    if group_cols is None:
        group_cols = ["sku_id", "branch_id"]

    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(group_cols + ["date"])

    feature_cols = [
        "lag_1d",
        "lag_3d",
        "lag_7d",
        "lag_14d",
        "lag_28d",
        "roll_3d_mean",
        "roll_7d_mean",
        "roll_14d_mean",
        "roll_28d_mean",
        "roll_7d_std",
        "roll_14d_std",
        "roll_28d_std",
        "ema_14d",
        "ema_30d",
    ]

    existing_features = [col for col in feature_cols if col in df.columns]
    grouped = df.groupby(group_cols + ["week"], sort=False)[existing_features].last()
    weekly = grouped.reset_index()

    rename_map = {col: f"{col}_last" for col in existing_features}
    weekly = weekly.rename(columns=rename_map)

    return weekly
