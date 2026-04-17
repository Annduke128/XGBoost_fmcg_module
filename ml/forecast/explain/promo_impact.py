"""Promo impact analysis: uplift by type, depth curve, and summary cross-tab."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


# ── Matched baseline helper ─────────────────────────────────────


def _matched_baseline(
    df: pd.DataFrame,
    units_col: str = "units",
    promo_col: str = "promo_flag",
    group_cols: list[str] | None = None,
) -> pd.Series:
    """Mean units when promo_flag == 0, per group.

    Returns a Series aligned to *df.index* so every row gets its
    matched (no-promo) baseline.  Groups without any non-promo data
    fall back to the overall non-promo mean.
    """
    if group_cols is None:
        group_cols = ["sku_id", "branch_id"]

    no_promo = df[df[promo_col] == 0]
    global_base = no_promo[units_col].mean() if len(no_promo) else 0.0

    if no_promo.empty:
        return pd.Series(global_base, index=df.index, dtype=float)

    group_base = no_promo.groupby(group_cols, observed=True)[units_col].mean()
    baseline = df.set_index(group_cols).index.map(
        lambda idx: group_base.get(idx, global_base)
    )
    return pd.Series(baseline.values, index=df.index, dtype=float)


# ── Public API ──────────────────────────────────────────────────


def promo_type_impact(
    df: pd.DataFrame,
    promo_type_col: str = "promo_type",
    units_col: str = "units",
    sku_group_col: str | None = "sku_id",
) -> pd.DataFrame:
    """Compute uplift percentage per promo type vs matched baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``promo_flag``, ``sku_id``, ``branch_id``, and
        the *promo_type_col* and *units_col* columns.
    promo_type_col : str
        Column identifying the promo type.
    units_col : str
        Column with sales units.
    sku_group_col : str | None
        Optional extra grouping (``"sku_id"``, ``"brand_type"``, or ``None``).
        When *None*, analysis is global (no SKU-level grouping).

    Returns
    -------
    pd.DataFrame
        Columns: ``promo_type``, ``promo_mean``, ``base_mean``,
        ``uplift_pct``, ``sample_size``.
    """
    if promo_type_col not in df.columns:
        return pd.DataFrame(
            columns=[
                "promo_type",
                "promo_mean",
                "base_mean",
                "uplift_pct",
                "sample_size",
            ]
        )

    # Baseline per (sku_id, branch_id)
    baseline_cols = ["sku_id", "branch_id"]
    baseline = _matched_baseline(df, units_col=units_col, group_cols=baseline_cols)

    promo_rows = df[df["promo_flag"] == 1].copy()
    promo_rows["_baseline"] = baseline[promo_rows.index]

    group_keys = [promo_type_col]
    if sku_group_col is not None and sku_group_col in promo_rows.columns:
        group_keys = [promo_type_col, sku_group_col]

    agg = (
        promo_rows.groupby(group_keys, observed=True)
        .agg(
            promo_mean=(units_col, "mean"),
            base_mean=("_baseline", "mean"),
            sample_size=(units_col, "size"),
        )
        .reset_index()
    )

    agg["uplift_pct"] = (
        (agg["promo_mean"] - agg["base_mean"]) / (agg["base_mean"] + EPS) * 100
    )

    return agg


def promo_depth_curve(
    df: pd.DataFrame,
    depth_col: str = "promo_depth",
    units_col: str = "units",
    sku_group_col: str | None = "sku_id",
    n_bins: int = 5,
) -> pd.DataFrame:
    """Bin promo_depth into *n_bins* buckets and compute uplift per bin.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``promo_flag``, ``sku_id``, ``branch_id``, and
        the *depth_col* and *units_col* columns.
    depth_col : str
        Column with promo depth values (0–1).
    units_col : str
        Column with sales units.
    sku_group_col : str | None
        Extra grouping (``"sku_id"``, ``"brand_type"``, or ``None``).
    n_bins : int
        Number of equal-width bins for depth.

    Returns
    -------
    pd.DataFrame
        Columns: ``depth_bin``, ``depth_range``, ``promo_mean``,
        ``base_mean``, ``uplift_pct``, ``sample_size``.
    """
    if depth_col not in df.columns:
        return pd.DataFrame(
            columns=[
                "depth_bin",
                "depth_range",
                "promo_mean",
                "base_mean",
                "uplift_pct",
                "sample_size",
            ]
        )

    baseline = _matched_baseline(df, units_col=units_col)

    promo_rows = df[df["promo_flag"] == 1].copy()
    if promo_rows.empty:
        return pd.DataFrame(
            columns=[
                "depth_bin",
                "depth_range",
                "promo_mean",
                "base_mean",
                "uplift_pct",
                "sample_size",
            ]
        )

    promo_rows["_baseline"] = baseline[promo_rows.index]

    # Bin promo_depth
    bins = np.linspace(0, 1, n_bins + 1)
    labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(n_bins)]
    promo_rows["depth_bin"] = pd.cut(
        promo_rows[depth_col], bins=bins, labels=labels, include_lowest=True
    )

    group_keys = ["depth_bin"]
    if sku_group_col is not None and sku_group_col in promo_rows.columns:
        group_keys = ["depth_bin", sku_group_col]

    agg = (
        promo_rows.groupby(group_keys, observed=False)
        .agg(
            promo_mean=(units_col, "mean"),
            base_mean=("_baseline", "mean"),
            sample_size=(units_col, "size"),
        )
        .reset_index()
    )

    agg["uplift_pct"] = (
        (agg["promo_mean"] - agg["base_mean"]) / (agg["base_mean"] + EPS) * 100
    )

    # Rename depth_bin to depth_range for clarity
    if "depth_bin" in agg.columns:
        agg = agg.rename(columns={"depth_bin": "depth_range"})

    return agg


def promo_impact_summary(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    units_col: str = "units",
) -> pd.DataFrame:
    """Cross-tab summary of promo uplift by arbitrary group columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``promo_flag``, ``sku_id``, ``branch_id``, and
        columns listed in *group_cols*.
    group_cols : list[str] | None
        Columns to cross-tabulate. Defaults to ``["promo_type", "brand_type"]``.
    units_col : str
        Column with sales units.

    Returns
    -------
    pd.DataFrame
        Columns: *group_cols* + ``promo_mean``, ``base_mean``,
        ``uplift_pct``, ``sample_size``.
    """
    if group_cols is None:
        group_cols = ["promo_type", "brand_type"]

    # Filter to columns that actually exist
    valid_cols = [c for c in group_cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame(
            columns=group_cols
            + ["promo_mean", "base_mean", "uplift_pct", "sample_size"]
        )

    baseline = _matched_baseline(df, units_col=units_col)

    promo_rows = df[df["promo_flag"] == 1].copy()
    if promo_rows.empty:
        return pd.DataFrame(
            columns=valid_cols
            + ["promo_mean", "base_mean", "uplift_pct", "sample_size"]
        )

    promo_rows["_baseline"] = baseline[promo_rows.index]

    agg = (
        promo_rows.groupby(valid_cols, observed=True)
        .agg(
            promo_mean=(units_col, "mean"),
            base_mean=("_baseline", "mean"),
            sample_size=(units_col, "size"),
        )
        .reset_index()
    )

    agg["uplift_pct"] = (
        (agg["promo_mean"] - agg["base_mean"]) / (agg["base_mean"] + EPS) * 100
    )

    return agg
