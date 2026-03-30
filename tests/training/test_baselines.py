"""Tests for baseline models."""

import numpy as np
import pandas as pd
from ml.training.models.baselines import seasonal_naive, poisson_baseline


def test_seasonal_naive():
    train = pd.DataFrame(
        {
            "sku_id": ["A"] * 8,
            "branch_id": ["B"] * 8,
            "week": pd.date_range("2025-01-06", periods=8, freq="W"),
            "units": [10, 12, 14, 16, 18, 20, 22, 24],
        }
    )
    val = pd.DataFrame(
        {
            "sku_id": ["A"] * 2,
            "branch_id": ["B"] * 2,
            "week": pd.date_range("2025-03-03", periods=2, freq="W"),
        }
    )
    preds = seasonal_naive(train, val)
    assert len(preds) == 2
    assert all(p >= 0 for p in preds)
    # Last value in train is 24
    assert preds[0] == 24.0


def test_seasonal_naive_unknown_group():
    """Unknown SKU-branch should get 0."""
    train = pd.DataFrame(
        {
            "sku_id": ["A"] * 4,
            "branch_id": ["B"] * 4,
            "week": pd.date_range("2025-01-06", periods=4, freq="W"),
            "units": [10, 12, 14, 16],
        }
    )
    val = pd.DataFrame(
        {
            "sku_id": ["UNKNOWN"],
            "branch_id": ["UNKNOWN"],
            "week": pd.date_range("2025-02-03", periods=1, freq="W"),
        }
    )
    preds = seasonal_naive(train, val)
    assert preds[0] == 0.0


def test_poisson_baseline():
    np.random.seed(42)
    X_train = np.random.rand(50, 3)
    y_train = np.random.poisson(5, 50).astype(float)
    X_val = np.random.rand(10, 3)
    preds = poisson_baseline(X_train, y_train, X_val)
    assert len(preds) == 10
    assert all(p >= 0 for p in preds)
