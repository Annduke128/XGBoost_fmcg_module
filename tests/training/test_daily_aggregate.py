"""Tests for daily_aggregate module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.training.data.daily_aggregate import (
    add_daily_lag_rolling_ema,
    aggregate_lag_features_to_weekly,
)


def _make_daily_df(n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2025-01-06", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "sku_id": ["SKU1"] * n_days,
            "branch_id": ["BR1"] * n_days,
            "units": np.arange(1, n_days + 1, dtype=float),
        }
    )


def test_add_daily_lag_rolling_ema_creates_expected_columns():
    df = _make_daily_df(40)
    out = add_daily_lag_rolling_ema(df)

    expected = {
        "lag_1d",
        "lag_3d",
        "lag_7d",
        "lag_14d",
        "lag_28d",
        "roll_3d_mean",
        "roll_7d_mean",
        "roll_14d_mean",
        "roll_28d_mean",
        "roll_7d_std",
        "roll_14d_std",
        "roll_28d_std",
        "ema_14d",
        "ema_30d",
    }
    assert expected.issubset(out.columns)


def test_lag_1d_is_previous_day_units():
    df = _make_daily_df(10)
    out = add_daily_lag_rolling_ema(df)

    # Day index 1 should equal day index 0
    assert out.loc[1, "lag_1d"] == df.loc[0, "units"]
    # Day index 3 should equal day index 2
    assert out.loc[3, "lag_1d"] == df.loc[2, "units"]


def test_rolling_features_have_no_nan_after_warmup():
    df = _make_daily_df(40)
    out = add_daily_lag_rolling_ema(df)

    warm_idx = 28
    assert not out.loc[warm_idx:, "roll_3d_mean"].isna().any()
    assert not out.loc[warm_idx:, "roll_7d_mean"].isna().any()
    assert not out.loc[warm_idx:, "roll_14d_mean"].isna().any()
    assert not out.loc[warm_idx:, "roll_28d_mean"].isna().any()
    assert not out.loc[warm_idx:, "roll_7d_std"].isna().any()
    assert not out.loc[warm_idx:, "roll_14d_std"].isna().any()
    assert not out.loc[warm_idx:, "roll_28d_std"].isna().any()


def test_aggregate_lag_features_to_weekly_reduces_rows_and_suffixes():
    df = _make_daily_df(21)
    df = add_daily_lag_rolling_ema(df)
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    weekly = aggregate_lag_features_to_weekly(df)

    assert len(weekly) == 3
    assert "lag_1d_last" in weekly.columns
    assert "roll_7d_mean_last" in weekly.columns
    assert "ema_14d_last" in weekly.columns


def test_weekly_last_value_matches_last_day_in_week():
    df = _make_daily_df(21)
    df = add_daily_lag_rolling_ema(df)
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    weekly = aggregate_lag_features_to_weekly(df)

    week_value = weekly.loc[0, "lag_1d_last"]
    last_day_value = df.loc[df["week"] == weekly.loc[0, "week"], "lag_1d"].iloc[-1]
    assert week_value == last_day_value
