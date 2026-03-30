from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9

CAT_DEFAULTS = {
    "store_type": "unknown",
    "display_capacity_type": "unknown",
    "service_scale": "unknown",
    "channel": "unknown",
}


def _mode_or_default(series: pd.Series, default: str) -> str:
    if series.empty:
        return default
    mode_vals = series.dropna().mode()
    if mode_vals.empty:
        return default
    return str(mode_vals.iloc[0])


def build_store_profiles(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"branch_id", "units"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "promo" not in df.columns:
        df = df.assign(promo=0)
    if "stockout_flag" not in df.columns:
        df = df.assign(stockout_flag=0)

    for col, default in CAT_DEFAULTS.items():
        if col not in df.columns:
            df = df.assign(**{col: default})

    has_week_col = "week" in df.columns
    has_date_col = "date" in df.columns

    df_local = df.copy()
    if has_date_col and not has_week_col:
        df_local["week"] = (
            pd.to_datetime(df_local["date"]).dt.to_period("W").astype(str)
        )

    group = df_local.groupby("branch_id", dropna=False)

    avg_weekly_sales = group["units"].mean()
    std_weekly_sales = group["units"].std(ddof=0)
    volatility = std_weekly_sales / (avg_weekly_sales + EPS)

    def promo_lift_fn(g: pd.DataFrame) -> float:
        promo_units = g.loc[g["promo"] == 1, "units"]
        base_units = g.loc[g["promo"] == 0, "units"]
        if promo_units.empty or base_units.empty:
            return 1.0
        return float(promo_units.mean() / (base_units.mean() + EPS))

    promo_lift = group.apply(promo_lift_fn)
    stockout_rate = group["stockout_flag"].mean()

    def seasonality_fn(g: pd.DataFrame) -> float:
        if "week" not in g.columns:
            return 1.0
        week_series = g["week"]
        if pd.api.types.is_datetime64_any_dtype(week_series):
            woy = pd.to_datetime(week_series).dt.isocalendar().week
        else:
            woy = pd.to_datetime(week_series, errors="coerce").dt.isocalendar().week
        weekly_avg = g.assign(woy=woy).groupby("woy")["units"].mean()
        if weekly_avg.empty:
            return 1.0
        min_val = weekly_avg.min()
        max_val = weekly_avg.max()
        if min_val <= 0:
            return 1.0
        return float(max_val / (min_val + EPS))

    seasonality_strength = group.apply(seasonality_fn)

    cat_profile = {}
    for col, default in CAT_DEFAULTS.items():
        cat_profile[col] = group[col].apply(lambda s, d=default: _mode_or_default(s, d))

    profiles = pd.DataFrame(
        {
            "avg_weekly_sales": avg_weekly_sales,
            "volatility": volatility,
            "promo_lift": promo_lift,
            "stockout_rate": stockout_rate,
            "seasonality_strength": seasonality_strength,
            **cat_profile,
        }
    ).reset_index()

    return profiles
