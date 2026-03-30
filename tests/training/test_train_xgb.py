"""Tests for XGBoost Poisson trainer."""

import numpy as np
import pandas as pd
from ml.training.models.train_xgb import train_xgb_poisson


def test_train_xgb_poisson_returns_model():
    np.random.seed(42)
    n = 100
    X_train = pd.DataFrame(
        {
            "f1": np.random.rand(n),
            "f2": np.random.rand(n),
            "cat1": pd.Categorical(np.random.choice(["a", "b", "c"], n)),
        }
    )
    y_train = np.random.poisson(5, n).astype(float)
    X_val = X_train.iloc[:10].copy()
    y_val = y_train[:10]
    model = train_xgb_poisson(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert len(preds) == 10
    assert all(p >= 0 for p in preds)


def test_train_xgb_poisson_with_custom_params():
    np.random.seed(42)
    n = 80
    X_train = pd.DataFrame(
        {
            "f1": np.random.rand(n),
            "f2": np.random.rand(n),
        }
    )
    y_train = np.random.poisson(3, n).astype(float)
    X_val = X_train.iloc[:10].copy()
    y_val = y_train[:10]
    custom_params = {
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,
        "max_delta_step": 0,
        "reg_alpha": 0.01,
        "reg_lambda": 0.5,
    }
    model = train_xgb_poisson(
        X_train, y_train, X_val, y_val, params=custom_params, n_estimators=50
    )
    preds = model.predict(X_val)
    assert len(preds) == 10
    assert all(p >= 0 for p in preds)
