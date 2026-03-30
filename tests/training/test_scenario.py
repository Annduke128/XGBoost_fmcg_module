import pandas as pd
from ml.training.data.scenario import build_scenarios


def test_build_scenarios_two_scenarios():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=2, freq="W"),
            "sku_id": ["A", "A"],
            "price": [100.0, 100.0],
            "promo_flag": [0, 0],
        }
    )
    scenarios = build_scenarios(df)
    assert "A" in scenarios  # scenario A: no promo
    assert "B" in scenarios  # scenario B: 50% discount


def test_scenario_b_price_halved():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=1, freq="W"),
            "sku_id": ["A"],
            "price": [100.0],
            "promo_flag": [0],
        }
    )
    scenarios = build_scenarios(df)
    assert scenarios["B"]["price"].iloc[0] == 50.0
    assert scenarios["B"]["promo_flag"].iloc[0] == 1
