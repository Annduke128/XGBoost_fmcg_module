"""Canonical feature definitions shared across training and forecast."""

from __future__ import annotations

CAT_COLS: list[str] = [
    "sku_id",
    "branch_id",
    "brand_type",
    "branch_type",
    "channel",
    "store_type",
    "display_capacity_type",
    "service_scale",
]

NUM_COLS: list[str] = [
    "price",
    "promo_flag",
    "sin_woy",
    "cos_woy",
    "lag_1",
    "lag_2",
    "lag_4",
    "roll_4_mean",
    "ema_4",
]

ALL_FEATURES: list[str] = CAT_COLS + NUM_COLS
