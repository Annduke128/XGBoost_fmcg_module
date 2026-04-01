import numpy as np
import pandas as pd

from ml.training.features.growth_features import (
    add_category_growth,
    add_sku_growth,
    build_growth_features,
    discretize_growth,
)


def _make_df() -> pd.DataFrame:
    weeks = pd.date_range("2025-01-06", periods=6, freq="W")
    return pd.DataFrame(
        {
            "week": weeks,
            "sku_id": ["A"] * 6,
            "branch_id": ["B"] * 6,
            "units": [10, 15, 20, 25, 30, 35],
            "category": ["C1"] * 6,
            "sub_category": ["SC1"] * 6,
        }
    )


def test_add_sku_growth_columns_and_values():
    df = _make_df()
    result = add_sku_growth(df)
    assert "qty_growth_1w" in result.columns
    assert "qty_growth_4w" in result.columns

    # lag_1=10, current=15 => growth=0.5
    assert np.isclose(result.loc[1, "qty_growth_1w"], 0.5)


def test_add_category_growth_columns():
    df = _make_df()
    result = add_category_growth(df)
    assert "category_growth_1w" in result.columns
    assert "category_growth_4w" in result.columns
    assert "subcat_growth_1w" in result.columns
    assert "subcat_growth_4w" in result.columns


def test_discretize_growth_bins_range():
    df = _make_df()
    df = add_sku_growth(df)
    result = discretize_growth(df, n_bins=5)

    for col in ["qty_growth_1w_bin", "qty_growth_4w_bin"]:
        assert col in result.columns
        vals = result[col].dropna().astype(int)
        assert vals.min() >= 0
        assert vals.max() <= 4


def test_build_growth_features_pipeline():
    df = _make_df()
    result = build_growth_features(df)
    assert "qty_growth_1w" in result.columns
    assert "category_growth_1w" in result.columns
    assert "qty_growth_1w_bin" in result.columns


def test_zero_lag_handled_with_eps():
    df = _make_df()
    df.loc[0, "units"] = 0
    df.loc[1, "units"] = 5
    result = add_sku_growth(df)

    assert np.isfinite(result.loc[1, "qty_growth_1w"])
