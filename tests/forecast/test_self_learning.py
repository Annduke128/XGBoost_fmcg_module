import numpy as np

from ml.forecast.replenishment.self_learning import (
    update_branch_adjustment,
    update_safety_sigma,
    update_seasonal_index,
)


def test_seasonal_index_update():
    current = {1: 1.0, 2: 1.0}
    actuals = np.array([12, 14])
    preds = np.array([10, 10])
    woys = np.array([1, 2])
    updated = update_seasonal_index(current, actuals, preds, woys, alpha=0.2)
    assert abs(updated[1] - 1.04) < 1e-6


def test_branch_adjustment_update():
    current = {"typeA": 1.0}
    actuals = np.array([15])
    preds = np.array([10])
    branch_types = np.array(["typeA"])
    updated = update_branch_adjustment(current, actuals, preds, branch_types, alpha=0.2)
    assert updated["typeA"] > 1.0


def test_safety_sigma_update():
    residuals = np.array([1, -2, 3, -1, 2, -3, 1, 0])
    sigma = update_safety_sigma(residuals, window=4)
    assert sigma > 0
