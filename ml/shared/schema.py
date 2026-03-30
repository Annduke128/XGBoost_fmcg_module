"""Data schema validation for FMCG forecast pipeline."""

from __future__ import annotations
import pandas as pd

REQUIRED_COLS: list[str] = [
    "week",
    "sku_id",
    "branch_id",
    "units",
    "price",
    "promo_flag",
    "brand_type",
    "branch_type",
    "stockout_flag",
    "display_units",
]


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that all required columns are present.

    Raises ValueError if any required column is missing.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df
