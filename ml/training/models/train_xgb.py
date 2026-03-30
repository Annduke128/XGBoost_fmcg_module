"""XGBoost Poisson regression trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_PARAMS: dict = {
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "max_delta_step": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}


def train_xgb_poisson(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict | None = None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 30,
) -> xgb.XGBRegressor:
    """Train XGBoost with Poisson objective and native categorical support.

    Args:
        X_train: Training features (categorical cols must be pd.Categorical dtype).
        y_train: Training target (non-negative units).
        X_val: Validation features.
        y_val: Validation target.
        params: XGBoost hyperparameters. Uses DEFAULT_PARAMS if None.
        n_estimators: Max boosting rounds.
        early_stopping_rounds: Early stopping patience.

    Returns:
        Fitted XGBRegressor model.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=n_estimators,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model
