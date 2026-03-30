"""Reorder point and order quantity computation using Polars."""

from __future__ import annotations

import polars as pl


def compute_reorder(
    df: pl.DataFrame,
    z: float = 1.65,
    forecast_col: str = "forecast_units",
    lt_col: str = "lead_time_weeks",
    on_hand_col: str = "on_hand",
) -> pl.DataFrame:
    """Compute reorder point and order quantity per SKU-branch.

    ROP = demand_during_lead_time + z * sigma * sqrt(lead_time)
    order_qty = max(0, ROP - on_hand)
    """
    agg = (
        df.group_by(["sku_id", "branch_id", lt_col])
        .agg(
            [
                pl.col(forecast_col).sum().alias("demand_lt"),
                pl.col(forecast_col).std().alias("sigma"),
                pl.col(on_hand_col).first().alias("on_hand"),
            ]
        )
        .with_columns(
            [
                pl.col("sigma").fill_null(0.0),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("demand_lt")
                    + z * pl.col("sigma") * pl.col(lt_col).cast(pl.Float64).sqrt()
                ).alias("reorder_point"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("reorder_point") - pl.col("on_hand") > 0)
                .then(pl.col("reorder_point") - pl.col("on_hand"))
                .otherwise(0.0)
                .alias("order_qty"),
            ]
        )
    )
    return agg
