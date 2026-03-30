"""Baseline models for FMCG demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor


def seasonal_naive(
    train: pd.DataFrame,
    val: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "units",
    lag_weeks: int = 4,
) -> np.ndarray:
    """Seasonal naive: predict using last known value per group.

    For data < 1 year, uses last observed value per SKU-branch as proxy.
    """
    if group_cols is None:
        group_cols = ["sku_id", "branch_id"]
    last_vals = train.sort_values("week").groupby(group_cols)[target_col].last()
    preds = (
        val.merge(
            last_vals.reset_index().rename(columns={target_col: "pred"}),
            on=group_cols,
            how="left",
        )["pred"]
        .fillna(0)
        .values
    )
    return np.maximum(preds, 0).astype(float)


def poisson_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
) -> np.ndarray:
    """Scikit-learn PoissonRegressor baseline."""
    model = PoissonRegressor(alpha=1.0, max_iter=300)
    model.fit(X_train, y_train)
    return model.predict(X_val)
