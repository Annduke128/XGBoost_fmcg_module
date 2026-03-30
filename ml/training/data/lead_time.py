"""Assign lead-time (1/2/4 weeks) based on velocity and display capacity."""

from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-9


def assign_lead_time(
    df: pd.DataFrame,
    display_col: str = "display_units",
    velocity_col: str = "ema_sales_8w",
) -> pd.DataFrame:
    """Assign lead_time_weeks based on weeks_of_cover rule.

    Rule:
        weeks_of_cover = display_units / ema_sales_8w
        <= 1  -> 1 week
        <= 2  -> 2 weeks
        else  -> 4 weeks
    """
    df = df.copy()
    cover = df[display_col] / (df[velocity_col] + EPS)
    df["weeks_of_cover"] = cover
    df["lead_time_weeks"] = np.select(
        [cover <= 1.0, cover <= 2.0],
        [1, 2],
        default=4,
    )
    return df
