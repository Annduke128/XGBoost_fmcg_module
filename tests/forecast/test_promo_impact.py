"""Tests for promo impact analysis: promo_type_impact, promo_depth_curve, promo_impact_summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.forecast.explain.promo_impact import (
    promo_depth_curve,
    promo_impact_summary,
    promo_type_impact,
)


def _sample_df() -> pd.DataFrame:
    """Sample DataFrame with mixed promo and non-promo rows."""
    return pd.DataFrame(
        {
            "sku_id": ["A"] * 8 + ["B"] * 8,
            "branch_id": ["BR1"] * 16,
            "promo_flag": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
            "promo_type": (
                ["no_promo"] * 4
                + ["direct_discount", "direct_discount", "buy_x_get_y", "buy_x_get_y"]
            )
            * 2,
            "brand_type": ["X"] * 16,
            "units": [
                # SKU A: base ~10, promo ~20
                10,
                10,
                10,
                10,
                20,
                20,
                15,
                15,
                # SKU B: base ~5, promo ~10
                5,
                5,
                5,
                5,
                10,
                10,
                8,
                8,
            ],
            "promo_depth": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.3,
                0.3,
                0.2,
                0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3,
                0.3,
                0.2,
                0.2,
            ],
        }
    )


# ── promo_type_impact ───────────────────────────────────────────


class TestPromoTypeImpact:
    def test_basic_uplift_computed(self):
        df = _sample_df()
        result = promo_type_impact(df)
        assert "uplift_pct" in result.columns
        assert "promo_mean" in result.columns
        assert "base_mean" in result.columns
        assert "sample_size" in result.columns
        assert len(result) > 0

    def test_uplift_positive_for_promo(self):
        df = _sample_df()
        result = promo_type_impact(df, sku_group_col=None)
        # Promo units > base units, so uplift should be positive
        for _, row in result.iterrows():
            assert row["uplift_pct"] > 0

    def test_group_by_sku(self):
        df = _sample_df()
        result = promo_type_impact(df, sku_group_col="sku_id")
        assert "sku_id" in result.columns

    def test_group_by_brand_type(self):
        df = _sample_df()
        result = promo_type_impact(df, sku_group_col="brand_type")
        assert "brand_type" in result.columns

    def test_no_promo_type_col_returns_empty(self):
        df = _sample_df().drop(columns=["promo_type"])
        result = promo_type_impact(df, promo_type_col="promo_type")
        assert len(result) == 0
        assert "uplift_pct" in result.columns

    def test_sample_size_correct(self):
        df = _sample_df()
        result = promo_type_impact(df, sku_group_col=None)
        total_promo = (df["promo_flag"] == 1).sum()
        assert result["sample_size"].sum() == total_promo


# ── promo_depth_curve ───────────────────────────────────────────


class TestPromoDepthCurve:
    def test_basic_bins(self):
        df = _sample_df()
        result = promo_depth_curve(df, n_bins=5, sku_group_col=None)
        assert "depth_range" in result.columns
        assert "uplift_pct" in result.columns
        # n_bins=5 → 5 rows (some may be empty)
        assert len(result) == 5

    def test_only_promo_rows_binned(self):
        df = _sample_df()
        result = promo_depth_curve(df, n_bins=5, sku_group_col=None)
        # Total sample_size should match promo rows only
        total_sample = result["sample_size"].sum()
        assert total_sample == (df["promo_flag"] == 1).sum()

    def test_no_depth_col_returns_empty(self):
        df = _sample_df().drop(columns=["promo_depth"])
        result = promo_depth_curve(df, depth_col="promo_depth")
        assert len(result) == 0
        assert "uplift_pct" in result.columns

    def test_no_promo_rows_returns_empty(self):
        df = _sample_df()
        df["promo_flag"] = 0  # No promos
        result = promo_depth_curve(df, n_bins=5)
        assert len(result) == 0

    def test_custom_bins(self):
        df = _sample_df()
        result = promo_depth_curve(df, n_bins=3, sku_group_col=None)
        assert len(result) == 3


# ── promo_impact_summary ────────────────────────────────────────


class TestPromoImpactSummary:
    def test_default_cross_tab(self):
        df = _sample_df()
        result = promo_impact_summary(df)
        assert "promo_type" in result.columns
        assert "brand_type" in result.columns
        assert "uplift_pct" in result.columns

    def test_custom_group_cols(self):
        df = _sample_df()
        result = promo_impact_summary(df, group_cols=["promo_type"])
        assert "promo_type" in result.columns
        assert "brand_type" not in result.columns

    def test_missing_group_cols_returns_empty(self):
        df = _sample_df()
        result = promo_impact_summary(df, group_cols=["nonexistent_col"])
        assert len(result) == 0

    def test_no_promo_rows(self):
        df = _sample_df()
        df["promo_flag"] = 0
        result = promo_impact_summary(df)
        assert len(result) == 0

    def test_sample_sizes_sum_to_promo_count(self):
        df = _sample_df()
        result = promo_impact_summary(df, group_cols=["promo_type"])
        total = result["sample_size"].sum()
        assert total == (df["promo_flag"] == 1).sum()
