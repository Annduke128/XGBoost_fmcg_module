import pandas as pd

from ml.training.segmentation.store_profiles import build_store_profiles


def test_build_store_profiles_structure():
    df = pd.DataFrame(
        {
            "branch_id": ["b1", "b1", "b2", "b2"],
            "units": [10, 20, 5, 7],
            "promo": [0, 1, 0, 0],
            "stockout_flag": [0, 0, 1, 0],
            "store_type": ["A", "A", "B", "B"],
            "display_capacity_type": ["S", "S", "M", "M"],
            "service_scale": ["L", "L", "S", "S"],
            "channel": ["offline", "offline", "online", "online"],
            "week": ["2024-01-01", "2024-01-08", "2024-01-01", "2024-01-08"],
        }
    )

    profiles = build_store_profiles(df)

    expected_cols = {
        "branch_id",
        "avg_weekly_sales",
        "volatility",
        "promo_lift",
        "stockout_rate",
        "seasonality_strength",
        "store_type",
        "display_capacity_type",
        "service_scale",
        "channel",
    }
    assert expected_cols.issubset(set(profiles.columns))
    assert len(profiles) == df["branch_id"].nunique()
