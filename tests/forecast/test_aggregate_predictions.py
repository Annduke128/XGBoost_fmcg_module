import pandas as pd

from ml.forecast.segmentation.aggregate_predictions import aggregate_to_store_type


def test_aggregate_to_store_type_structure():
    df = pd.DataFrame(
        {
            "sku_id": ["s1", "s1", "s2"],
            "store_type": ["A", "A", "B"],
            "branch_id": ["b1", "b2", "b3"],
            "forecast_units": [10, 20, 5],
            "units": [12, 18, 6],
        }
    )

    result = aggregate_to_store_type(df)

    expected_cols = {
        "sku_id",
        "store_type",
        "total_forecast",
        "total_actual",
        "n_branches",
        "avg_forecast_per_branch",
    }
    assert expected_cols.issubset(set(result.columns))
    assert len(result) == 2
