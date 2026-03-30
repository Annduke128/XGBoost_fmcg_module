"""Censor & impute demand during stockout periods."""

from __future__ import annotations
import pandas as pd


def impute_stockout(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "units",
    stockout_col: str = "stockout_flag",
    window: int = 4,
) -> pd.DataFrame:
    """Replace units during stockout with rolling median of prior non-stockout weeks.

    Uses shift(1) to avoid leakage — only past values used.
    """
    df = df.sort_values("week").copy()
    # Ensure target is float to accept imputed values
    df[target_col] = df[target_col].astype(float)
    # Rolling median on shifted (past-only) values within each group
    roll_med = (
        df.groupby(group_cols)[target_col]
        .shift(1)
        .rolling(window, min_periods=1)
        .median()
    )
    mask = df[stockout_col] == 1
    df.loc[mask, target_col] = roll_med[mask]
    # Fallback: if still NaN (e.g. first row), use group median
    if df.loc[mask, target_col].isna().any():
        group_med = df.groupby(group_cols)[target_col].transform("median")
        still_na = mask & df[target_col].isna()
        df.loc[still_na, target_col] = group_med[still_na]
    return df
