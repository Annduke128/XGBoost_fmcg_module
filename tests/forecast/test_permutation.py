import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ml.forecast.explain.permutation import run_permutation


def test_run_permutation_returns_importances_mean():
    X = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [2, 1, 2, 1, 2]})
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    model = LinearRegression().fit(X, y)

    result = run_permutation(model, X, y, n_repeats=2)
    assert hasattr(result, "importances_mean")
