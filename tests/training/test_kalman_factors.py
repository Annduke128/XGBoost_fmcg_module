"""Tests for Kalman filter, seasonal_kalman, and promo_kalman."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from ml.training.factors.kalman_filter import (
    KalmanConfig,
    KalmanFactorStore,
    KalmanState,
    kalman_update,
)
from ml.training.factors.seasonal_kalman import (
    apply_seasonal_factors,
    create_seasonal_store,
    update_seasonal_factors,
)
from ml.training.factors.promo_kalman import (
    apply_promo_factors,
    create_promo_store,
    update_promo_factors,
)


# ── Kalman filter core ──────────────────────────────────────────


class TestKalmanUpdate:
    def test_update_moves_toward_observation(self):
        config = KalmanConfig(Q=0.01, R=0.05)
        state = KalmanState(x=1.0, P=0.1)
        # Observation says factor should be 1.2
        new = kalman_update(state, 1.2, config)
        assert new.x > 1.0
        assert new.x < 1.2  # Should move toward but not reach

    def test_invalid_observation_skipped(self):
        config = KalmanConfig()
        state = KalmanState(x=1.0, P=0.1)
        new = kalman_update(state, float("nan"), config)
        assert new.x == 1.0  # State unchanged
        assert new.P > state.P  # Uncertainty grows

    def test_negative_observation_skipped(self):
        config = KalmanConfig()
        state = KalmanState(x=1.0, P=0.1)
        new = kalman_update(state, -0.5, config)
        assert new.x == 1.0

    def test_factor_clamped(self):
        config = KalmanConfig(Q=0.01, R=0.001)  # Very low R → high gain
        state = KalmanState(x=1.0, P=1.0)
        # Very extreme observation
        new = kalman_update(state, 10.0, config)
        assert new.x <= 2.0  # Clamped

    def test_convergence_after_repeated_updates(self):
        config = KalmanConfig(Q=0.01, R=0.05)
        state = KalmanState(x=1.0, P=0.1)
        # Repeated observations at 1.3
        for _ in range(20):
            state = kalman_update(state, 1.3, config)
        assert abs(state.x - 1.3) < 0.1


class TestKalmanFactorStore:
    def test_default_factor_is_one(self):
        store = KalmanFactorStore()
        assert store.get_factor("unknown_key") == 1.0

    def test_update_changes_factor(self):
        store = KalmanFactorStore()
        factor = store.update("key1", 1.5)
        assert factor > 1.0

    def test_save_and_load(self, tmp_path):
        store = KalmanFactorStore()
        store.update("woy_1", 1.2)
        store.update("woy_2", 0.8)

        save_path = tmp_path / "factors.json"
        store.save(save_path)

        loaded = KalmanFactorStore.load(save_path)
        assert abs(loaded.get_factor("woy_1") - store.get_factor("woy_1")) < 1e-10
        assert abs(loaded.get_factor("woy_2") - store.get_factor("woy_2")) < 1e-10

    def test_len(self):
        store = KalmanFactorStore()
        assert len(store) == 0
        store.update("k1", 1.0)
        assert len(store) == 1


# ── Seasonal Kalman ─────────────────────────────────────────────


class TestSeasonalKalman:
    def _make_df(self) -> pd.DataFrame:
        weeks = pd.date_range("2025-01-06", periods=4, freq="W")
        return pd.DataFrame(
            {
                "week": weeks,
                "units": [100, 120, 90, 110],
                "pred_base": [95, 100, 100, 100],
            }
        )

    def test_update_creates_factors(self):
        store = create_seasonal_store()
        df = self._make_df()
        update_seasonal_factors(store, df)
        assert len(store) > 0

    def test_apply_adds_column(self):
        store = create_seasonal_store()
        df = self._make_df()
        update_seasonal_factors(store, df)
        out = apply_seasonal_factors(df, store)
        assert "seasonal_factor" in out.columns
        assert all(out["seasonal_factor"] > 0)


# ── Promo Kalman ────────────────────────────────────────────────


class TestPromoKalman:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "promo_type": [
                    "discount_bundle",
                    "direct_discount",
                    "no_promo",
                    "buy_gift",
                ],
                "units": [150, 200, 80, 120],
                "pred_base": [100, 150, 90, 100],
            }
        )

    def test_update_creates_factors(self):
        store = create_promo_store()
        df = self._make_df()
        update_promo_factors(store, df)
        assert len(store) == 4

    def test_apply_adds_column(self):
        store = create_promo_store()
        df = self._make_df()
        update_promo_factors(store, df)
        out = apply_promo_factors(df, store)
        assert "promo_factor" in out.columns
        assert all(out["promo_factor"] > 0)

    def test_missing_promo_type_defaults(self):
        store = create_promo_store()
        df = pd.DataFrame({"units": [10], "pred_base": [10]})
        out = apply_promo_factors(df, store)
        assert out["promo_factor"].iloc[0] == 1.0
