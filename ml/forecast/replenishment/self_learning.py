"""Self-learning adjustments for seasonal, branch-type, and safety stock."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

EPS = 1e-9


def update_seasonal_index(
    current_index: dict[int, float],
    actuals: np.ndarray,
    predictions: np.ndarray,
    week_of_year: np.ndarray,
    alpha: float = 0.2,
) -> dict[int, float]:
    """Update seasonal index using EMA on actual/predicted ratio.
    seasonal[woy] = (1 - alpha) * seasonal[woy] + alpha * (actual / pred)
    """
    updated = dict(current_index)
    ratios_by_woy: dict[int, list[float]] = defaultdict(list)
    for actual, pred, woy in zip(actuals, predictions, week_of_year):
        ratio = actual / (pred + EPS)
        ratios_by_woy[int(woy)].append(ratio)
    for woy, ratios in ratios_by_woy.items():
        avg_ratio = float(np.mean(ratios))
        old = updated.get(woy, 1.0)
        updated[woy] = (1 - alpha) * old + alpha * avg_ratio
    return updated


def update_branch_adjustment(
    current_adj: dict[str, float],
    actuals: np.ndarray,
    predictions: np.ndarray,
    branch_types: np.ndarray,
    alpha: float = 0.2,
) -> dict[str, float]:
    """Update branch-type adjustment factor using EMA on actual/predicted ratio."""
    updated = dict(current_adj)
    ratios_by_type: dict[str, list[float]] = defaultdict(list)
    for actual, pred, bt in zip(actuals, predictions, branch_types):
        ratio = actual / (pred + EPS)
        ratios_by_type[str(bt)].append(ratio)
    for bt, ratios in ratios_by_type.items():
        avg_ratio = float(np.mean(ratios))
        old = updated.get(bt, 1.0)
        updated[bt] = (1 - alpha) * old + alpha * avg_ratio
    return updated


def update_safety_sigma(
    residuals: np.ndarray,
    window: int = 8,
) -> float:
    """Compute rolling std of forecast residuals for safety stock."""
    recent = residuals[-window:] if len(residuals) > window else residuals
    return float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
