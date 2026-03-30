"""Optuna hyperparameter tuning for XGBoost Poisson."""

from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit

from ml.shared.utils.metrics import wape
from ml.training.models.train_xgb import train_xgb_poisson

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 3,
) -> float:
    """Optuna objective: mean WAPE across time-series folds."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        model = train_xgb_poisson(
            X_tr,
            y_tr,
            X_va,
            y_va,
            params=params,
            n_estimators=200,
            early_stopping_rounds=20,
        )
        preds = model.predict(X_va)
        scores.append(wape(y_va, preds))

    return float(np.mean(scores))


def run_optuna_study(
    X: pd.DataFrame,
    y: np.ndarray,
    weeks: pd.Series | pd.DatetimeIndex,
    n_trials: int = 50,
    n_splits: int = 3,
) -> dict:
    """Run Optuna study and return best hyperparameters.

    Args:
        X: Feature DataFrame (sorted by time).
        y: Target array (non-negative units).
        weeks: Week dates (used for documentation, data assumed pre-sorted).
        n_trials: Number of Optuna trials.
        n_splits: Number of TimeSeriesSplit folds.

    Returns:
        Dictionary of best hyperparameters.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, X, y, n_splits),
        n_trials=n_trials,
    )
    return study.best_params
