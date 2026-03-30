import polars as pl

from ml.forecast.replenishment.reorder_polars import compute_reorder


def test_reorder_basic():
    df = pl.DataFrame(
        {
            "sku_id": ["A", "A", "A", "A"],
            "branch_id": ["B", "B", "B", "B"],
            "lead_time_weeks": [2, 2, 2, 2],
            "forecast_units": [10.0, 12.0, 11.0, 13.0],
            "week": ["2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27"],
            "on_hand": [15, 15, 15, 15],
        }
    )
    result = compute_reorder(df)
    assert "reorder_point" in result.columns
    assert "order_qty" in result.columns
    assert result["order_qty"].to_list()[0] >= 0


def test_reorder_no_negative_order():
    df = pl.DataFrame(
        {
            "sku_id": ["A"],
            "branch_id": ["B"],
            "lead_time_weeks": [1],
            "forecast_units": [5.0],
            "week": ["2025-01-06"],
            "on_hand": [100],
        }
    )
    result = compute_reorder(df)
    assert result["order_qty"].to_list()[0] == 0


def test_reorder_respects_lead_time():
    df_short = pl.DataFrame(
        {
            "sku_id": ["A", "A"],
            "branch_id": ["B", "B"],
            "lead_time_weeks": [1, 1],
            "forecast_units": [10.0, 10.0],
            "week": ["2025-01-06", "2025-01-13"],
            "on_hand": [5, 5],
        }
    )
    df_long = df_short.with_columns(pl.lit(4).alias("lead_time_weeks"))
    r_short = compute_reorder(df_short)
    r_long = compute_reorder(df_long)
    assert r_long["reorder_point"].to_list()[0] >= r_short["reorder_point"].to_list()[0]
