"""Tests for compute_promo_depth — unified promo discount rate computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.training.data.promo_depth import compute_promo_depth


def _base_df(**kwargs) -> pd.DataFrame:
    """Minimal DataFrame with required columns; kwargs override defaults."""
    data = {
        "price": [100.0],
        "promo_flag": [1],
    }
    data.update(kwargs)
    return pd.DataFrame(data)


# ── Skip when promo_depth already exists ─────────────────────────


class TestSkipExisting:
    def test_returns_unchanged_if_promo_depth_exists(self):
        df = _base_df(promo_depth=[0.3])
        result = compute_promo_depth(df)
        assert result["promo_depth"].iloc[0] == 0.3

    def test_no_copy_when_skipped(self):
        df = _base_df(promo_depth=[0.5])
        result = compute_promo_depth(df)
        assert result is df  # same object


# ── Conversion by promo_type ────────────────────────────────────


class TestDirectDiscount:
    def test_direct_discount_uses_promo_discount(self):
        df = _base_df(promo_type=["direct_discount"], promo_discount=[0.25])
        result = compute_promo_depth(df)
        assert np.isclose(result["promo_depth"].iloc[0], 0.25)

    def test_discount_bundle_uses_promo_discount(self):
        df = _base_df(promo_type=["discount_bundle"], promo_discount=[0.40])
        result = compute_promo_depth(df)
        assert np.isclose(result["promo_depth"].iloc[0], 0.40)


class TestBuyXGetY:
    def test_buy_2_get_1(self):
        df = _base_df(
            promo_type=["buy_x_get_y"],
            promo_x_qty=[2],
            promo_y_qty=[1],
        )
        result = compute_promo_depth(df)
        # y / (x + y) = 1 / 3 ≈ 0.333
        assert np.isclose(result["promo_depth"].iloc[0], 1 / 3, atol=1e-4)

    def test_buy_1_get_1(self):
        df = _base_df(
            promo_type=["buy_x_get_y"],
            promo_x_qty=[1],
            promo_y_qty=[1],
        )
        result = compute_promo_depth(df)
        assert np.isclose(result["promo_depth"].iloc[0], 0.5, atol=1e-4)


class TestBuyGift:
    def test_gift_value_over_price(self):
        df = _base_df(
            promo_type=["buy_gift"],
            gift_value=[20.0],
            price=[100.0],
            promo_x_qty=[1],
        )
        result = compute_promo_depth(df)
        # 20 / (100 * 1) = 0.2
        assert np.isclose(result["promo_depth"].iloc[0], 0.2, atol=1e-4)

    def test_gift_clamped_to_one(self):
        df = _base_df(
            promo_type=["buy_gift"],
            gift_value=[500.0],
            price=[100.0],
            promo_x_qty=[1],
        )
        result = compute_promo_depth(df)
        assert result["promo_depth"].iloc[0] <= 1.0


class TestNoPromo:
    def test_no_promo_returns_zero(self):
        df = _base_df(promo_type=["no_promo"])
        result = compute_promo_depth(df)
        assert result["promo_depth"].iloc[0] == 0.0

    def test_nan_promo_type_returns_zero(self):
        df = _base_df(promo_type=[None])
        result = compute_promo_depth(df)
        assert result["promo_depth"].iloc[0] == 0.0


# ── Missing raw columns ────────────────────────────────────────


class TestMissingColumns:
    def test_no_promo_columns_at_all(self):
        """If no promo columns exist, promo_depth defaults to 0."""
        df = pd.DataFrame({"price": [100.0], "units": [50]})
        result = compute_promo_depth(df)
        assert "promo_depth" in result.columns
        assert result["promo_depth"].iloc[0] == 0.0

    def test_no_price_column(self):
        """buy_gift without price should not crash."""
        df = pd.DataFrame(
            {
                "promo_type": ["buy_gift"],
                "gift_value": [20.0],
                "promo_x_qty": [1],
            }
        )
        result = compute_promo_depth(df)
        assert "promo_depth" in result.columns


# ── Mixed promo types ───────────────────────────────────────────


class TestMixed:
    def test_multiple_promo_types(self):
        df = pd.DataFrame(
            {
                "price": [100.0, 100.0, 100.0, 100.0],
                "promo_type": [
                    "direct_discount",
                    "buy_x_get_y",
                    "buy_gift",
                    "no_promo",
                ],
                "promo_discount": [0.30, 0.0, 0.0, 0.0],
                "promo_x_qty": [0, 2, 1, 0],
                "promo_y_qty": [0, 1, 0, 0],
                "gift_value": [0.0, 0.0, 15.0, 0.0],
            }
        )
        result = compute_promo_depth(df)

        assert np.isclose(result["promo_depth"].iloc[0], 0.30)  # direct
        assert np.isclose(result["promo_depth"].iloc[1], 1 / 3, atol=1e-4)  # bxgy
        assert np.isclose(result["promo_depth"].iloc[2], 0.15, atol=1e-4)  # gift
        assert result["promo_depth"].iloc[3] == 0.0  # no promo

    def test_output_clipped_0_to_1(self):
        df = _base_df(promo_type=["direct_discount"], promo_discount=[1.5])
        result = compute_promo_depth(df)
        assert result["promo_depth"].iloc[0] <= 1.0

    def test_raw_columns_preserved(self):
        df = _base_df(
            promo_type=["direct_discount"],
            promo_discount=[0.25],
            promo_x_qty=[0],
        )
        result = compute_promo_depth(df)
        assert "promo_type" in result.columns
        assert "promo_discount" in result.columns
        assert "promo_x_qty" in result.columns
