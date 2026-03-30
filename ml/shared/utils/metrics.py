"""Forecast evaluation metrics for FMCG pipeline."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

EPS: float = 1e-6


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true)) + EPS
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Percentage Error (epsilon-safe for y=0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def mdape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Median Absolute Percentage Error (epsilon-safe for y=0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.median(np.abs((y_true - y_pred) / denom)))
