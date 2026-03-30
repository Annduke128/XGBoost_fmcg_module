"""Permutation importance utilities for forecast explainability."""

from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from ml.shared.utils.metrics import wape


def _neg_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -wape(y_true, y_pred)


def run_permutation(
    model: object,
    X_val,
    y_val,
    n_repeats: int = 5,
):
    """Run permutation importance with negative WAPE scorer."""
    scorer = make_scorer(_neg_wape, greater_is_better=True)
    return permutation_importance(
        model,
        X_val,
        y_val,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=42,
    )


def select_top_features(
    result,
    feature_names: list[str],
    threshold: float = 0.0,
) -> list[str]:
    """Select features with importance mean above threshold."""
    importances = result.importances_mean
    return [
        name
        for name, score in zip(feature_names, importances, strict=False)
        if score > threshold
    ]
