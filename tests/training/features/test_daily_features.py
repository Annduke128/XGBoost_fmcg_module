"""Tests for daily_features module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ml.training.features.daily_features import (
    add_daily_time_features,
    aggregate_daily_to_weekly,
)


def _make_daily_df(n_days: int = 14) -> pd.DataFrame:
    """Create a simple daily DataFrame for testing."""
    dates = pd.date_range("2025-01-06", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "sku_id": ["SKU1"] * n_days,
            "branch_id": ["BR1"] * n_days,
            "units": np.random.default_rng(42).integers(5, 20, size=n_days),
        }
    )


class TestAddDailyTimeFeatures:
    def test_creates_weekend_flag(self):
        df = _make_daily_df(7)
        out = add_daily_time_features(df)
        assert "is_weekend" in out.columns
        # Jan 6 2025 is Monday, so Sat=Jan 11, Sun=Jan 12
        assert (
            out.loc[out["date"] == pd.Timestamp("2025-01-11"), "is_weekend"].iloc[0]
            == 1
        )
        assert (
            out.loc[out["date"] == pd.Timestamp("2025-01-06"), "is_weekend"].iloc[0]
            == 0
        )

    def test_creates_month_boundary_flags(self):
        df = _make_daily_df(7)
        out = add_daily_time_features(df)
        assert "month_start_flag" in out.columns
        assert "month_end_flag" in out.columns

    def test_creates_holiday_flag(self):
        # Use Jan 1 2025 as holiday
        dates = pd.date_range("2024-12-30", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "sku_id": ["A"] * 5,
                "branch_id": ["B"] * 5,
                "units": [10] * 5,
            }
        )
        out = add_daily_time_features(df, holiday_years=[2025])
        assert "is_holiday" in out.columns
        jan1_row = out.loc[out["date"] == pd.Timestamp("2025-01-01")]
        assert jan1_row["is_holiday"].iloc[0] == 1

    def test_dow_column(self):
        df = _make_daily_df(7)
        out = add_daily_time_features(df)
        assert "dow" in out.columns
        # Jan 6 2025 is Monday = 0
        assert out.iloc[0]["dow"] == 0


class TestAggregateDailyToWeekly:
    def test_aggregation_reduces_rows(self):
        df = _make_daily_df(14)
        df = add_daily_time_features(df)
        weekly = aggregate_daily_to_weekly(df)
        assert len(weekly) == 2  # 14 days = 2 weeks

    def test_units_summed(self):
        df = _make_daily_df(7)
        df = add_daily_time_features(df)
        weekly = aggregate_daily_to_weekly(df)
        assert weekly["units"].iloc[0] == df["units"].sum()

    def test_weekend_ratio_computed(self):
        df = _make_daily_df(7)
        df = add_daily_time_features(df)
        weekly = aggregate_daily_to_weekly(df)
        assert "weekend_ratio" in weekly.columns
        ratio = weekly["weekend_ratio"].iloc[0]
        assert 0 <= ratio <= 1

    def test_peak_features_present(self):
        df = _make_daily_df(7)
        df = add_daily_time_features(df)
        weekly = aggregate_daily_to_weekly(df)
        assert "peak_day" in weekly.columns
        assert "peak_level" in weekly.columns
        assert 0 <= weekly["peak_level"].iloc[0] <= 1

    def test_holiday_features_present(self):
        df = _make_daily_df(7)
        df = add_daily_time_features(df)
        weekly = aggregate_daily_to_weekly(df)
        assert "holiday_flag" in weekly.columns
        assert "holiday_ratio" in weekly.columns
