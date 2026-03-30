"""Encoding utilities for high-cardinality categorical features."""

from __future__ import annotations
import pandas as pd


def make_categorical(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Cast columns to pandas Categorical dtype for XGBoost native support."""
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df
