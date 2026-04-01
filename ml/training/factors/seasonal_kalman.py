"""Seasonal Kalman factor — self-learning adjustment per week-of-year.

Maintains a KalmanFactorStore keyed by week-of-year (woy_1 .. woy_52).
Observation = actual / pred_base for each week.
Factor is applied: pred_final *= seasonal_factor[woy]
"""

from __future__ import annotations

import pandas as pd

from ml.training.factors.kalman_filter import KalmanConfig, KalmanFactorStore


def create_seasonal_store(config: KalmanConfig | None = None) -> KalmanFactorStore:
    """Create a new seasonal factor store with optional custom config."""
    return KalmanFactorStore(config=config)


def update_seasonal_factors(
    store: KalmanFactorStore,
    actuals: pd.DataFrame,
    pred_col: str = "pred_base",
    actual_col: str = "units",
    week_col: str = "week",
) -> KalmanFactorStore:
    """Update seasonal factors with new actual vs predicted data.

    Args:
        store: KalmanFactorStore for seasonal factors
        actuals: DataFrame with week, actual_col, pred_col columns
        pred_col: Column name for base predictions
        actual_col: Column name for actual values
        week_col: Column name for week datetime

    Returns:
        Updated store
    """
    df = actuals.copy()
    df["_woy"] = pd.to_datetime(df[week_col]).dt.isocalendar().week.astype(int)

    # Aggregate by week-of-year across all SKU-branches
    woy_agg = (
        df.groupby("_woy")
        .agg(
            actual_sum=(actual_col, "sum"),
            pred_sum=(pred_col, "sum"),
        )
        .reset_index()
    )

    for _, row in woy_agg.iterrows():
        woy = int(row["_woy"])
        key = f"woy_{woy}"
        if row["pred_sum"] > 0:
            obs = row["actual_sum"] / row["pred_sum"]
            store.update(key, obs)

    return store


def apply_seasonal_factors(
    df: pd.DataFrame,
    store: KalmanFactorStore,
    pred_col: str = "pred_base",
    week_col: str = "week",
    out_col: str = "seasonal_factor",
) -> pd.DataFrame:
    """Apply seasonal adjustment factors to predictions.

    Adds a seasonal_factor column and multiplies pred_col by it.
    """
    df = df.copy()
    woy = pd.to_datetime(df[week_col]).dt.isocalendar().week.astype(int)
    df[out_col] = woy.apply(lambda w: store.get_factor(f"woy_{w}"))
    return df
