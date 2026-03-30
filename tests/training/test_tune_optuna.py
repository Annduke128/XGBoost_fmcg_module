"""Tests for Optuna hyperparameter tuning."""

import numpy as np
import pandas as pd
from ml.training.models.tune_optuna import run_optuna_study


def test_optuna_returns_best_params():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        {
            "f1": np.random.rand(n),
            "f2": np.random.rand(n),
        }
    )
    y = np.random.poisson(5, n).astype(float)
    weeks = pd.date_range("2025-01-06", periods=n, freq="W")
    best_params = run_optuna_study(X, y, weeks, n_trials=3, n_splits=2)
    assert "max_depth" in best_params
    assert "eta" in best_params
    assert "subsample" in best_params
    assert "colsample_bytree" in best_params
    assert "min_child_weight" in best_params
    assert "max_delta_step" in best_params
    assert "reg_alpha" in best_params
    assert "reg_lambda" in best_params
