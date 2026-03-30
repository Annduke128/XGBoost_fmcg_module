"""Time-based train/val/test split for weekly data."""

from __future__ import annotations
import pandas as pd


def time_split(
    df: pd.DataFrame,
    val_weeks: int = 4,
    test_weeks: int = 4,
    time_col: str = "week",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe by time. No shuffling — preserves temporal order."""
    df = df.sort_values(time_col).copy()
    max_week = df[time_col].max()
    test_cutoff = max_week - pd.Timedelta(weeks=test_weeks)
    val_cutoff = test_cutoff - pd.Timedelta(weeks=val_weeks)

    train = df[df[time_col] < val_cutoff].copy()
    val = df[(df[time_col] >= val_cutoff) & (df[time_col] < test_cutoff)].copy()
    test = df[df[time_col] >= test_cutoff].copy()
    return train, val, test
