"""Feature engineering pipeline for FMCG weekly forecast."""

from __future__ import annotations
import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic time/seasonal features from week column."""
    df = df.copy()
    df["weekofyear"] = df["week"].dt.isocalendar().week.astype(int)
    df["month"] = df["week"].dt.month
    df["sin_woy"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["cos_woy"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "units",
) -> pd.DataFrame:
    """Add lag, rolling mean, and EMA features.

    All computed on shift(1)+ to prevent leakage.
    """
    if group_cols is None:
        group_cols = ["sku_id", "branch_id"]

    df = df.sort_values(["sku_id", "branch_id", "week"]).copy()
    grp = df.groupby(group_cols)[target_col]

    # Lags
    for lag in [1, 2, 4]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling mean on shifted series (past only)
    shifted = grp.shift(1)
    df["roll_4_mean"] = shifted.rolling(4, min_periods=1).mean()

    # EMA on shifted series
    df["ema_4"] = shifted.ewm(span=4, adjust=False).mean()

    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    df = add_time_features(df)
    df = add_lag_features(df)
    return df
