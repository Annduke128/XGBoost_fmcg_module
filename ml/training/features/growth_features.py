"""Growth feature engineering for FMCG weekly forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def add_sku_growth(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "units",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Add qty_growth_1w, qty_growth_4w at SKU-branch level."""
    if group_cols is None:
        group_cols = ["sku_id", "branch_id"]

    df = df.sort_values(group_cols + ["week"]).copy()
    grp = df.groupby(group_cols)[target_col]

    if "lag_1" in df.columns:
        lag_1 = df["lag_1"]
    else:
        lag_1 = grp.shift(1)

    if "lag_4" in df.columns:
        lag_4 = df["lag_4"]
    else:
        lag_4 = grp.shift(4)

    df["qty_growth_1w"] = (df[target_col] - lag_1) / (lag_1 + eps)
    df["qty_growth_4w"] = (df[target_col] - lag_4) / (lag_4 + eps)

    return df


def add_category_growth(
    df: pd.DataFrame,
    target_col: str = "units",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Add category_growth_1w/4w and subcat_growth_1w/4w."""
    df = df.copy()

    # Category-level aggregation
    cat_agg = df.groupby(["category", "week"], as_index=False)[target_col].sum()
    cat_agg = cat_agg.sort_values(["category", "week"])
    cat_grp = cat_agg.groupby("category")[target_col]
    cat_agg["category_growth_1w"] = (cat_agg[target_col] - cat_grp.shift(1)) / (
        cat_grp.shift(1) + eps
    )
    cat_agg["category_growth_4w"] = (cat_agg[target_col] - cat_grp.shift(4)) / (
        cat_grp.shift(4) + eps
    )
    cat_agg = cat_agg[["category", "week", "category_growth_1w", "category_growth_4w"]]

    # Sub-category-level aggregation
    subcat_agg = df.groupby(["sub_category", "week"], as_index=False)[target_col].sum()
    subcat_agg = subcat_agg.sort_values(["sub_category", "week"])
    subcat_grp = subcat_agg.groupby("sub_category")[target_col]
    subcat_agg["subcat_growth_1w"] = (subcat_agg[target_col] - subcat_grp.shift(1)) / (
        subcat_grp.shift(1) + eps
    )
    subcat_agg["subcat_growth_4w"] = (subcat_agg[target_col] - subcat_grp.shift(4)) / (
        subcat_grp.shift(4) + eps
    )
    subcat_agg = subcat_agg[
        ["sub_category", "week", "subcat_growth_1w", "subcat_growth_4w"]
    ]

    df = df.merge(cat_agg, on=["category", "week"], how="left")
    df = df.merge(subcat_agg, on=["sub_category", "week"], how="left")

    return df


def discretize_growth(
    df: pd.DataFrame, columns: list[str] | None = None, n_bins: int = 5
) -> pd.DataFrame:
    """Discretize growth columns into ordinal bins using KBinsDiscretizer."""
    if columns is None:
        columns = ["qty_growth_1w", "qty_growth_4w"]

    df = df.copy()
    data = df[columns].to_numpy(dtype=float)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    binned = kbd.fit_transform(data)

    for idx, col in enumerate(columns):
        df[f"{col}_bin"] = binned[:, idx].astype(int)

    return df


def build_growth_features(
    df: pd.DataFrame, group_cols: list[str] | None = None, target_col: str = "units"
) -> pd.DataFrame:
    """Full growth feature pipeline."""
    df = add_sku_growth(df, group_cols=group_cols, target_col=target_col)
    df = add_category_growth(df, target_col=target_col)
    df = discretize_growth(df)
    return df
