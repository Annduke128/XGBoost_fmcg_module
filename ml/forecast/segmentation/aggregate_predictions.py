from __future__ import annotations

import pandas as pd


def aggregate_to_store_type(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["sku_id", "store_type"]

    required = {"forecast_units", "units", "branch_id"} | set(group_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            total_forecast=("forecast_units", "sum"),
            total_actual=("units", "sum"),
            n_branches=("branch_id", "nunique"),
        )
        .reset_index()
    )

    agg["avg_forecast_per_branch"] = agg["total_forecast"] / agg["n_branches"]

    return agg
