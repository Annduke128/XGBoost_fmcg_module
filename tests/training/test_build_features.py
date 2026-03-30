import pandas as pd
import numpy as np
from ml.training.features.build_features import (
    add_time_features,
    add_lag_features,
    build_all_features,
)


def _make_df(n=10):
    return pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=n, freq="W"),
            "sku_id": ["A"] * n,
            "branch_id": ["B"] * n,
            "units": list(range(1, n + 1)),
            "price": [100.0] * n,
            "promo_flag": [0] * n,
            "brand_type": ["X"] * n,
            "branch_type": ["Y"] * n,
            "stockout_flag": [0] * n,
            "display_units": [20] * n,
            "channel": ["MT"] * n,
            "store_type": ["S1"] * n,
            "display_capacity_type": ["D1"] * n,
            "service_scale": ["L"] * n,
        }
    )


def test_time_features_created():
    df = _make_df()
    result = add_time_features(df)
    assert "weekofyear" in result.columns
    assert "sin_woy" in result.columns
    assert "cos_woy" in result.columns


def test_lag_features_no_leakage():
    """lag_1 at row i should equal units at row i-1."""
    df = _make_df()
    result = add_lag_features(df)
    assert pd.isna(result["lag_1"].iloc[0])  # first row has no history
    assert result["lag_1"].iloc[1] == df["units"].iloc[0]


def test_rolling_no_leakage():
    df = _make_df()
    result = add_lag_features(df)
    # roll_4_mean at row 1 should only use row 0
    assert result["roll_4_mean"].iloc[1] == df["units"].iloc[0]


def test_build_all_features_output_shape():
    df = _make_df(20)
    result = build_all_features(df)
    assert "lag_1" in result.columns
    assert "sin_woy" in result.columns
    assert "ema_4" in result.columns
