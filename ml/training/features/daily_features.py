"""Daily-level feature extraction aggregated to weekly.

Takes daily sales data and computes time-based features
that are then aggregated to the weekly level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.training.data.holiday_calendar import build_holiday_set


def add_daily_time_features(
    df: pd.DataFrame,
    holiday_csv: str | None = None,
    holiday_years: list[int] | None = None,
) -> pd.DataFrame:
    """Add daily time features before weekly aggregation.

    Expects a DataFrame with a 'date' column (daily granularity).
    Returns the same DataFrame with additional columns.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Day-of-week (0=Mon, 6=Sun)
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Month boundary flags
    df["month_start_flag"] = (df["date"].dt.day <= 3).astype(int)
    df["month_end_flag"] = (
        (df["date"] + pd.Timedelta(days=3)).dt.month != df["date"].dt.month
    ).astype(int)

    # Holiday
    holidays = build_holiday_set(csv_path=holiday_csv, years=holiday_years)
    df["is_holiday"] = df["date"].dt.normalize().isin(holidays).astype(int)

    return df


def aggregate_daily_to_weekly(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "units",
) -> pd.DataFrame:
    """Aggregate daily features to weekly level.

    Expects daily DataFrame with features from add_daily_time_features().
    Returns weekly DataFrame with aggregated features.

    Group by: group_cols + week.
    """
    if group_cols is None:
        group_cols = ["sku_id", "branch_id"]

    df = df.copy()
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    agg_dict: dict = {
        target_col: "sum",
        "is_weekend": "mean",  # weekend_ratio
        "is_holiday": "max",  # holiday_flag (1 if any holiday in week)
        "month_start_flag": "max",  # 1 if week overlaps month start
        "month_end_flag": "max",  # 1 if week overlaps month end
    }

    # Holiday ratio: fraction of days that are holidays
    if "is_holiday" in df.columns:
        agg_dict["is_holiday"] = ["max", "mean"]

    # Peak day and peak level within the week
    peak_agg = (
        df.groupby(group_cols + ["week"])
        .apply(_compute_peak, target_col=target_col, include_groups=False)
        .reset_index()
    )

    # Main aggregation
    weekly = df.groupby(group_cols + ["week"]).agg(agg_dict).reset_index()

    # Flatten multi-level columns
    weekly.columns = _flatten_columns(weekly.columns)

    # Rename aggregated columns
    rename_map = {
        f"{target_col}_sum": target_col,
        "is_weekend_mean": "weekend_ratio",
        "is_holiday_max": "holiday_flag",
        "is_holiday_mean": "holiday_ratio",
        "month_start_flag_max": "month_start_flag",
        "month_end_flag_max": "month_end_flag",
    }
    weekly = weekly.rename(columns=rename_map)

    # Merge peak features
    weekly = weekly.merge(peak_agg, on=group_cols + ["week"], how="left")

    return weekly


def _compute_peak(group: pd.DataFrame, target_col: str = "units") -> pd.Series:
    """Compute peak day (day-of-week) and peak level within a week group."""
    if group.empty or group[target_col].sum() == 0:
        return pd.Series({"peak_day": 0, "peak_level": 0.0})

    idx_max = group[target_col].idxmax()
    peak_day = group.loc[idx_max, "dow"] if "dow" in group.columns else 0
    total = group[target_col].sum()
    peak_level = group[target_col].max() / total if total > 0 else 0.0

    return pd.Series({"peak_day": int(peak_day), "peak_level": float(peak_level)})


def _flatten_columns(columns: pd.Index) -> list[str]:
    """Flatten multi-level column index to single level."""
    flat = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(c) for c in col if c != ""]
            flat.append("_".join(parts) if len(parts) > 1 else parts[0])
        else:
            flat.append(str(col))
    return flat
