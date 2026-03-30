"""Generate promo scenarios for forecast evaluation."""

from __future__ import annotations
import pandas as pd


def build_scenarios(
    df: pd.DataFrame,
    discount_pct: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Build two forecast scenarios.

    Scenario A: no promo (promo_flag=0, price unchanged).
    Scenario B: promo active (promo_flag=1, price * (1 - discount_pct)).
    """
    scenario_a = df.copy()
    scenario_a["promo_flag"] = 0

    scenario_b = df.copy()
    scenario_b["promo_flag"] = 1
    scenario_b["price"] = scenario_b["price"] * (1 - discount_pct)

    return {"A": scenario_a, "B": scenario_b}
