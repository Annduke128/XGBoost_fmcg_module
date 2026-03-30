import pandas as pd
from ml.training.data.lead_time import assign_lead_time


def test_lead_time_fast_mover():
    """High velocity (cover <= 1 week) -> lead_time = 1."""
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=1, freq="W"),
            "sku_id": ["A"],
            "branch_id": ["B"],
            "display_units": [10],
            "ema_sales_8w": [15.0],  # cover = 10/15 = 0.67
        }
    )
    result = assign_lead_time(df)
    assert result["lead_time_weeks"].iloc[0] == 1


def test_lead_time_medium_mover():
    """Medium velocity (1 < cover <= 2) -> lead_time = 2."""
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=1, freq="W"),
            "sku_id": ["A"],
            "branch_id": ["B"],
            "display_units": [10],
            "ema_sales_8w": [6.0],  # cover = 10/6 = 1.67
        }
    )
    result = assign_lead_time(df)
    assert result["lead_time_weeks"].iloc[0] == 2


def test_lead_time_slow_mover():
    """Low velocity (cover > 2) -> lead_time = 4."""
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=1, freq="W"),
            "sku_id": ["A"],
            "branch_id": ["B"],
            "display_units": [10],
            "ema_sales_8w": [3.0],  # cover = 10/3 = 3.33
        }
    )
    result = assign_lead_time(df)
    assert result["lead_time_weeks"].iloc[0] == 4


def test_lead_time_zero_sales():
    """Zero sales -> max lead_time = 4."""
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=1, freq="W"),
            "sku_id": ["A"],
            "branch_id": ["B"],
            "display_units": [10],
            "ema_sales_8w": [0.0],
        }
    )
    result = assign_lead_time(df)
    assert result["lead_time_weeks"].iloc[0] == 4
