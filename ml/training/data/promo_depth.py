"""Compute unified promo_depth from heterogeneous promo columns."""

from __future__ import annotations

import pandas as pd

EPS = 1e-6


def compute_promo_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all promo types to effective discount rate (promo_depth).

    If ``promo_depth`` already exists, returns the DataFrame unchanged.

    Conversion rules per promo_type:
        - direct_discount  → promo_discount as-is
        - discount_bundle  → promo_discount as-is
        - buy_x_get_y      → y / (x + y)
        - buy_gift         → gift_value / (price * x_qty)
        - no_promo / NaN   → 0.0

    Raw promo columns are preserved; only ``promo_depth`` is added.
    Missing raw columns default to sensible zero values.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with optional promo columns:
        ``promo_type``, ``promo_discount``, ``promo_x_qty``,
        ``promo_y_qty``, ``gift_value``, ``price``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``promo_depth`` column (float, clipped 0.0–1.0).
    """
    if "promo_depth" in df.columns:
        return df

    df = df.copy()

    # Ensure optional raw columns exist with safe defaults
    _defaults: list[tuple[str, object]] = [
        ("promo_type", "no_promo"),
        ("promo_discount", 0.0),
        ("promo_x_qty", 0.0),
        ("promo_y_qty", 0.0),
        ("gift_value", 0.0),
    ]
    for col, default in _defaults:
        if col not in df.columns:
            df[col] = default

    # Coerce numerics
    for col in ("promo_discount", "promo_x_qty", "promo_y_qty", "gift_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "price" in df.columns:
        price = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    else:
        price = pd.Series(0.0, index=df.index)

    ptype = df["promo_type"].fillna("no_promo").astype(str).str.lower().str.strip()

    depth = pd.Series(0.0, index=df.index, dtype=float)

    # direct_discount / discount_bundle → promo_discount
    mask_discount = ptype.isin(["direct_discount", "discount_bundle"])
    if mask_discount.any():
        depth[mask_discount] = df.loc[mask_discount, "promo_discount"].clip(0, 1)

    # buy_x_get_y → y / (x + y)
    mask_bxgy = ptype == "buy_x_get_y"
    if mask_bxgy.any():
        x = df.loc[mask_bxgy, "promo_x_qty"]
        y = df.loc[mask_bxgy, "promo_y_qty"]
        depth[mask_bxgy] = y / (x + y + EPS)

    # buy_gift → gift_value / (price * x_qty)
    mask_gift = ptype == "buy_gift"
    if mask_gift.any():
        gv = df.loc[mask_gift, "gift_value"]
        x_qty = df.loc[mask_gift, "promo_x_qty"].clip(lower=1)
        px = price[mask_gift] * x_qty
        depth[mask_gift] = gv / (px + EPS)

    df["promo_depth"] = depth.clip(0, 1)
    return df
