"""Promo Kalman factor — self-learning adjustment per promo type.

Maintains a KalmanFactorStore keyed by promo_type
(discount_bundle, direct_discount, buy_x_get_y, buy_gift, no_promo).
Observation = actual / pred_base for each promo type.
Factor is applied: pred_final *= promo_factor[promo_type]
"""

from __future__ import annotations

import pandas as pd

from ml.training.factors.kalman_filter import KalmanConfig, KalmanFactorStore


def create_promo_store(config: KalmanConfig | None = None) -> KalmanFactorStore:
    """Create a new promo factor store with optional custom config."""
    return KalmanFactorStore(config=config)


def update_promo_factors(
    store: KalmanFactorStore,
    actuals: pd.DataFrame,
    pred_col: str = "pred_base",
    actual_col: str = "units",
    promo_type_col: str = "promo_type",
) -> KalmanFactorStore:
    """Update promo factors with new actual vs predicted data.

    Args:
        store: KalmanFactorStore for promo factors
        actuals: DataFrame with promo_type, actual_col, pred_col
        pred_col: Column name for base predictions
        actual_col: Column name for actual values
        promo_type_col: Column name for promo type

    Returns:
        Updated store
    """
    # Default promo_type for rows without promo
    df = actuals.copy()
    if promo_type_col not in df.columns:
        df[promo_type_col] = "no_promo"
    df[promo_type_col] = df[promo_type_col].fillna("no_promo")

    # Aggregate by promo_type
    promo_agg = (
        df.groupby(promo_type_col)
        .agg(
            actual_sum=(actual_col, "sum"),
            pred_sum=(pred_col, "sum"),
        )
        .reset_index()
    )

    for _, row in promo_agg.iterrows():
        ptype = str(row[promo_type_col])
        key = f"promo_{ptype}"
        if row["pred_sum"] > 0:
            obs = row["actual_sum"] / row["pred_sum"]
            store.update(key, obs)

    return store


def apply_promo_factors(
    df: pd.DataFrame,
    store: KalmanFactorStore,
    promo_type_col: str = "promo_type",
    out_col: str = "promo_factor",
) -> pd.DataFrame:
    """Apply promo adjustment factors to predictions.

    Adds a promo_factor column.
    """
    df = df.copy()
    if promo_type_col not in df.columns:
        df[out_col] = 1.0
        return df

    df[out_col] = (
        df[promo_type_col]
        .fillna("no_promo")
        .apply(lambda pt: store.get_factor(f"promo_{pt}"))
    )
    return df
