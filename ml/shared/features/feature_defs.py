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
    "promo_type_major",
]

# ── Original weekly features ────────────────────────────────────
NUM_COLS_CORE: list[str] = [
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

# ── Daily time features (aggregated to weekly) ──────────────────
NUM_COLS_DAILY_TIME: list[str] = [
    "weekend_ratio",
    "holiday_flag",
    "holiday_ratio",
    "month_start_flag",
    "month_end_flag",
    "peak_day",
    "peak_level",
]

# ── Daily lag/rolling/EMA (last value per week) ─────────────────
NUM_COLS_DAILY_LAG: list[str] = [
    "lag_1d_last",
    "lag_3d_last",
    "lag_7d_last",
    "lag_14d_last",
    "lag_28d_last",
    "roll_3d_mean_last",
    "roll_7d_mean_last",
    "roll_14d_mean_last",
    "roll_28d_mean_last",
    "roll_7d_std_last",
    "roll_14d_std_last",
    "roll_28d_std_last",
    "ema_14d_last",
    "ema_30d_last",
]

# ── Growth features ─────────────────────────────────────────────
NUM_COLS_GROWTH: list[str] = [
    "qty_growth_1w",
    "qty_growth_4w",
    "category_growth_1w",
    "category_growth_4w",
    "subcat_growth_1w",
    "subcat_growth_4w",
    "qty_growth_1w_bin",
    "qty_growth_4w_bin",
]

# ── Promo detail features ───────────────────────────────────────
NUM_COLS_PROMO: list[str] = [
    "promo_ratio",
    "promo_depth_avg",
]

# ── Weather features ────────────────────────────────────────────
NUM_COLS_WEATHER: list[str] = [
    "rainy_ratio",
    "temp_avg",
    "temp_max",
    "temp_min",
]

# ── Kalman self-learning factors ────────────────────────────────
NUM_COLS_KALMAN: list[str] = [
    "seasonal_factor",
    "promo_factor",
]

# ── Combined ────────────────────────────────────────────────────
NUM_COLS: list[str] = (
    NUM_COLS_CORE
    + NUM_COLS_DAILY_TIME
    + NUM_COLS_DAILY_LAG
    + NUM_COLS_GROWTH
    + NUM_COLS_PROMO
    + NUM_COLS_WEATHER
    + NUM_COLS_KALMAN
)

ALL_FEATURES: list[str] = CAT_COLS + NUM_COLS
