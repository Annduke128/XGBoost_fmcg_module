"""Tests for recursive multi-step forecast pipeline."""

import numpy as np
import pandas as pd
import pytest

from ml.forecast.pipelines.forecast_weekly import (
    ForecastConfig,
    _recompute_lags,
    _rollout_forecast,
)


def test_forecast_config_defaults():
    cfg = ForecastConfig()
    assert cfg.horizons == [1, 2, 4]
    assert cfg.scenarios == ["A", "B"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyModel:
    """Always predicts a fixed value (or sum of lag_1 + 1 for traceability)."""

    def __init__(self, fixed: float | None = None):
        self._fixed = fixed

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._fixed is not None:
            return np.full(len(X), self._fixed)
        # Use lag_1 + 1 so we can trace recursive feeding
        if "lag_1" in X.columns:
            return (X["lag_1"].fillna(0).values + 1).astype(float)
        return np.ones(len(X))


def _make_history(n_weeks: int = 10) -> pd.DataFrame:
    """Build a simple single-SKU history with features already present."""
    weeks = pd.date_range("2025-01-06", periods=n_weeks, freq="W")
    units = list(range(10, 10 + n_weeks))
    df = pd.DataFrame(
        {
            "week": weeks,
            "sku_id": ["A"] * n_weeks,
            "branch_id": ["B"] * n_weeks,
            "units": units,
            "price": [100.0] * n_weeks,
            "promo_flag": [0] * n_weeks,
            "brand_type": ["X"] * n_weeks,
            "branch_type": ["Y"] * n_weeks,
            "stockout_flag": [0] * n_weeks,
            "display_units": [20] * n_weeks,
        }
    )
    # Build lags manually (shift-based, same as build_features)
    df = df.sort_values(["sku_id", "branch_id", "week"])
    for lag in [1, 2, 4]:
        df[f"lag_{lag}"] = df["units"].shift(lag)
    shifted = df["units"].shift(1)
    df["roll_4_mean"] = shifted.rolling(4, min_periods=1).mean()
    df["ema_4"] = shifted.ewm(span=4, adjust=False).mean()
    return df.dropna(subset=["lag_1"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# _recompute_lags
# ---------------------------------------------------------------------------


class TestRecomputeLags:
    def test_lag_1_matches_prev_units(self):
        df = _make_history()
        result = _recompute_lags(df)
        # lag_1 at row i should be units at row i-1
        assert result["lag_1"].iloc[1] == result["units"].iloc[0]

    def test_appended_row_updates_lags(self):
        df = _make_history()
        last_week = df["week"].max()
        new_row = df[df["week"] == last_week].copy()
        new_row["week"] = last_week + pd.Timedelta(weeks=1)
        new_row["units"] = 999.0
        combined = pd.concat([df, new_row], ignore_index=True)
        result = _recompute_lags(combined)
        # lag_1 of the appended row should be units of the previous last row
        appended = result[result["week"] == new_row["week"].iloc[0]]
        prev_units = df[df["week"] == last_week]["units"].iloc[0]
        assert appended["lag_1"].iloc[0] == prev_units


# ---------------------------------------------------------------------------
# _rollout_forecast
# ---------------------------------------------------------------------------


class TestRolloutForecast:
    def test_output_only_requested_horizons(self):
        df = _make_history()
        model = _DummyModel(fixed=5.0)
        result = _rollout_forecast(model, df, [1, 2, 4], ["lag_1", "price"])
        assert set(result["horizon"].unique()) == {1, 2, 4}
        # h=3 should NOT appear even though it is computed internally
        assert 3 not in result["horizon"].values

    def test_lag_1_at_h2_equals_pred_h1(self):
        """At h=2, lag_1 should be the prediction from h=1 (recursive).

        Chain with model(predict = lag_1 + 1):
        - h=1 target T+1: lag_1 = units[T] (original last).  pred = units[T]+1
        - h=2 target T+2: lag_1 = units[T+1] = pred(h=1).    pred = pred(h=1)+1
        """
        df = _make_history()
        model = _DummyModel(fixed=None)  # predict = lag_1 + 1

        last_units = df[df["week"] == df["week"].max()]["units"].iloc[0]
        h1_pred = last_units + 1  # lag_1 at T+1 = units[T]
        h2_expected = h1_pred + 1  # lag_1 at T+2 = pred(h=1)

        result = _rollout_forecast(model, df, [1, 2], ["lag_1"])
        h2_actual = result[result["horizon"] == 2]["forecast_units"].iloc[0]
        assert abs(h2_actual - h2_expected) < 1e-6

    def test_lag_at_h4_uses_h3_internal(self):
        """h=4 lag_1 should be prediction from h=3 (internal step).

        Chain with model(predict = lag_1 + 1):
        - h=1 target T+1: lag_1 = units[T].           pred = units[T]+1
        - h=2 target T+2: lag_1 = pred(h=1).          pred = pred(h=1)+1
        - h=3 target T+3: lag_1 = pred(h=2).          pred = pred(h=2)+1
        - h=4 target T+4: lag_1 = pred(h=3).          pred = pred(h=3)+1
        """
        df = _make_history()
        model = _DummyModel(fixed=None)

        last_units = df[df["week"] == df["week"].max()]["units"].iloc[0]
        h1 = last_units + 1
        h2 = h1 + 1
        h3 = h2 + 1
        h4 = h3 + 1

        result = _rollout_forecast(model, df, [1, 2, 4], ["lag_1"])
        actual_h4 = result[result["horizon"] == 4]["forecast_units"].iloc[0]
        assert abs(actual_h4 - h4) < 1e-6

    def test_forecast_units_non_negative(self):
        df = _make_history()
        # Model that returns negative values — should be clipped to 0
        model = _DummyModel(fixed=-5.0)
        result = _rollout_forecast(model, df, [1, 2, 4], ["lag_1"])
        assert (result["forecast_units"] >= 0).all()

    def test_empty_df_returns_correct_schema(self):
        df = pd.DataFrame(
            columns=[
                "week",
                "sku_id",
                "branch_id",
                "units",
                "lag_1",
                "lag_2",
                "lag_4",
                "roll_4_mean",
                "ema_4",
                "price",
            ]
        )
        model = _DummyModel(fixed=1.0)
        result = _rollout_forecast(model, df, [1, 2, 4], ["lag_1"])
        expected_cols = {"sku_id", "branch_id", "week", "horizon", "forecast_units"}
        assert set(result.columns) == expected_cols
        assert len(result) == 0

    def test_multiple_skus(self):
        """Rollout should work independently per (sku_id, branch_id) group."""
        df1 = _make_history()
        df2 = _make_history()
        df2["sku_id"] = "C"
        df2["units"] = df2["units"] + 100  # different level
        combined = pd.concat([df1, df2], ignore_index=True)

        model = _DummyModel(fixed=None)
        result = _rollout_forecast(model, combined, [1, 2], ["lag_1"])
        # Should have 2 groups x 2 horizons = 4 rows
        assert len(result) == 4
        assert set(result["sku_id"].unique()) == {"A", "C"}
