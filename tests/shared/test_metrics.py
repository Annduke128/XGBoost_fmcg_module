import numpy as np
from ml.shared.utils.metrics import wape, mae, mape, mdape


def test_wape_basic():
    y = np.array([10, 10])
    yhat = np.array([8, 12])
    result = wape(y, yhat)
    assert abs(result - 0.2) < 1e-6


def test_mae_basic():
    y = np.array([10, 20])
    yhat = np.array([12, 18])
    assert abs(mae(y, yhat) - 2.0) < 1e-6


def test_mape_with_zero():
    y = np.array([10, 0, 20])
    yhat = np.array([9, 1, 22])
    result = mape(y, yhat)
    assert result >= 0


def test_mdape_with_zero():
    y = np.array([10, 0, 20])
    yhat = np.array([9, 1, 22])
    result = mdape(y, yhat)
    assert result >= 0


def test_wape_all_zero():
    y = np.array([0, 0])
    yhat = np.array([1, 1])
    result = wape(y, yhat)
    assert result > 0  # should not divide by zero
