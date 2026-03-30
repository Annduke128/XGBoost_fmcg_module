import pandas as pd
from ml.training.data.stockout import impute_stockout


def test_impute_stockout_replaces_zero():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=6, freq="W"),
            "sku_id": ["A"] * 6,
            "branch_id": ["B"] * 6,
            "units": [10, 12, 0, 11, 13, 0],
            "stockout_flag": [0, 0, 1, 0, 0, 1],
        }
    )
    result = impute_stockout(df, ["sku_id", "branch_id"])
    # Stockout rows should have imputed (non-zero) values
    assert result.loc[result["stockout_flag"] == 1, "units"].min() > 0


def test_impute_stockout_no_change_non_stockout():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=4, freq="W"),
            "sku_id": ["A"] * 4,
            "branch_id": ["B"] * 4,
            "units": [10, 12, 14, 16],
            "stockout_flag": [0, 0, 0, 0],
        }
    )
    result = impute_stockout(df, ["sku_id", "branch_id"])
    pd.testing.assert_series_equal(result["units"], df["units"].astype(float))
