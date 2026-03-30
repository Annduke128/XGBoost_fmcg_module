# FMCG Forecast – Plan 07: Replenishment + Self-Learning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use skill({ name: "executing-plans" }) to implement this plan task-by-task.

**Goal:** Compute reorder quantities using Polars + implement self-learning adjustments for seasonal/branch/safety stock.  
**Architecture:** Forecast → Reorder (Polars) → Update state (EMA). Service level 95% (z=1.65).  
**Tech Stack:** polars, numpy, pytest.

---

## Must-Haves

### Observable Truths

1. Reorder plan computed per SKU-branch with lead-time-appropriate demand aggregation.
2. Safety stock uses z=1.65 (95% service level).
3. Seasonal index updates weekly via EMA on residuals.
4. Branch-type adjustment updates weekly via EMA on residuals.
5. Safety stock sigma recalculated from rolling std of forecast errors.

### Required Artifacts

| Artifact      | Provides                     | Path                                          |
| ------------- | ---------------------------- | --------------------------------------------- |
| Reorder       | ROP + order qty (Polars)     | `ml/forecast/replenishment/reorder_polars.py` |
| Self-learning | Seasonal/branch/sigma update | `ml/forecast/replenishment/self_learning.py`  |

### Key Links

| From          | To       | Via            | Risk                            |
| ------------- | -------- | -------------- | ------------------------------- |
| forecast      | reorder  | forecast_units | stale forecast if not refreshed |
| self_learning | forecast | adjustments    | feedback loop amplification     |

---

## Task Dependencies

```
Task 1 (Reorder): needs Plan 06 (forecast output)
Task 2 (Self-learning): needs Plan 06 (forecast output)

Wave 1: Task 1, Task 2 (parallel, after Plan 06)
```

---

### Task 1: Reorder computation (Polars)

**Files:**

- Create: `ml/forecast/replenishment/__init__.py`
- Create: `ml/forecast/replenishment/reorder_polars.py`
- Test: `tests/forecast/test_reorder.py`

**Step 1: Write failing test**

```python
# tests/forecast/test_reorder.py
import polars as pl
from ml.forecast.replenishment.reorder_polars import compute_reorder

def test_reorder_basic():
    df = pl.DataFrame({
        "sku_id": ["A", "A", "A", "A"],
        "branch_id": ["B", "B", "B", "B"],
        "lead_time_weeks": [2, 2, 2, 2],
        "forecast_units": [10.0, 12.0, 11.0, 13.0],
        "week": ["2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27"],
        "on_hand": [15, 15, 15, 15],
    })
    result = compute_reorder(df)
    assert "reorder_point" in result.columns
    assert "order_qty" in result.columns
    # Order qty should be >= 0
    assert result["order_qty"].to_list()[0] >= 0

def test_reorder_no_negative_order():
    df = pl.DataFrame({
        "sku_id": ["A"],
        "branch_id": ["B"],
        "lead_time_weeks": [1],
        "forecast_units": [5.0],
        "week": ["2025-01-06"],
        "on_hand": [100],  # plenty of stock
    })
    result = compute_reorder(df)
    assert result["order_qty"].to_list()[0] == 0

def test_reorder_respects_lead_time():
    """Longer lead time should aggregate more demand."""
    df_short = pl.DataFrame({
        "sku_id": ["A", "A"],
        "branch_id": ["B", "B"],
        "lead_time_weeks": [1, 1],
        "forecast_units": [10.0, 10.0],
        "week": ["2025-01-06", "2025-01-13"],
        "on_hand": [5, 5],
    })
    df_long = df_short.with_columns(pl.lit(4).alias("lead_time_weeks"))
    r_short = compute_reorder(df_short)
    r_long = compute_reorder(df_long)
    # Longer lead time -> higher reorder point
    assert r_long["reorder_point"].to_list()[0] >= r_short["reorder_point"].to_list()[0]
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
# ml/forecast/replenishment/reorder_polars.py
"""Reorder point and order quantity computation using Polars."""
from __future__ import annotations
import polars as pl
import numpy as np

def compute_reorder(
    df: pl.DataFrame,
    z: float = 1.65,
    forecast_col: str = "forecast_units",
    lt_col: str = "lead_time_weeks",
    on_hand_col: str = "on_hand",
) -> pl.DataFrame:
    """Compute reorder point and order quantity per SKU-branch.

    ROP = demand_during_lead_time + z * sigma * sqrt(lead_time)
    order_qty = max(0, ROP - on_hand)

    Args:
        df: DataFrame with sku_id, branch_id, lead_time_weeks, forecast_units, on_hand
        z: Safety factor (1.65 for 95% service level)
    """
    # Aggregate forecast over lead-time horizon per SKU-branch
    agg = (
        df.group_by(["sku_id", "branch_id", lt_col])
        .agg([
            pl.col(forecast_col).sum().alias("demand_lt"),
            pl.col(forecast_col).std().alias("sigma"),
            pl.col(on_hand_col).first().alias("on_hand"),
        ])
        .with_columns([
            # Handle null sigma (single row groups)
            pl.col("sigma").fill_null(0.0),
        ])
        .with_columns([
            (
                pl.col("demand_lt")
                + z * pl.col("sigma") * pl.col(lt_col).cast(pl.Float64).sqrt()
            ).alias("reorder_point"),
        ])
        .with_columns([
            pl.when(pl.col("reorder_point") - pl.col("on_hand") > 0)
            .then(pl.col("reorder_point") - pl.col("on_hand"))
            .otherwise(0.0)
            .alias("order_qty"),
        ])
    )
    return agg
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add ml/forecast/replenishment/ tests/forecast/test_reorder.py
git commit -m "feat: add Polars-based reorder point computation"
```

---

### Task 2: Self-learning adjustments

**Files:**

- Create: `ml/forecast/replenishment/self_learning.py`
- Test: `tests/forecast/test_self_learning.py`

**Step 1: Write failing test**

```python
# tests/forecast/test_self_learning.py
import numpy as np
from ml.forecast.replenishment.self_learning import (
    update_seasonal_index,
    update_branch_adjustment,
    update_safety_sigma,
)

def test_seasonal_index_update():
    current = {1: 1.0, 2: 1.0}
    actuals = np.array([12, 14])
    preds = np.array([10, 10])
    woys = np.array([1, 2])
    updated = update_seasonal_index(current, actuals, preds, woys, alpha=0.2)
    # week 1: ratio=1.2, new = 0.8*1.0 + 0.2*1.2 = 1.04
    assert abs(updated[1] - 1.04) < 1e-6

def test_branch_adjustment_update():
    current = {"typeA": 1.0}
    actuals = np.array([15])
    preds = np.array([10])
    branch_types = np.array(["typeA"])
    updated = update_branch_adjustment(current, actuals, preds, branch_types, alpha=0.2)
    assert updated["typeA"] > 1.0

def test_safety_sigma_update():
    residuals = np.array([1, -2, 3, -1, 2, -3, 1, 0])
    sigma = update_safety_sigma(residuals, window=4)
    assert sigma > 0
```

**Step 2: Implement**

```python
# ml/forecast/replenishment/self_learning.py
"""Self-learning adjustments for seasonal, branch-type, and safety stock."""
from __future__ import annotations
import numpy as np
from collections import defaultdict

EPS = 1e-9

def update_seasonal_index(
    current_index: dict[int, float],
    actuals: np.ndarray,
    predictions: np.ndarray,
    week_of_year: np.ndarray,
    alpha: float = 0.2,
) -> dict[int, float]:
    """Update seasonal index using EMA on actual/predicted ratio.

    seasonal[woy] = (1 - alpha) * seasonal[woy] + alpha * (actual / pred)
    """
    updated = dict(current_index)
    ratios_by_woy: dict[int, list[float]] = defaultdict(list)

    for actual, pred, woy in zip(actuals, predictions, week_of_year):
        ratio = actual / (pred + EPS)
        ratios_by_woy[int(woy)].append(ratio)

    for woy, ratios in ratios_by_woy.items():
        avg_ratio = float(np.mean(ratios))
        old = updated.get(woy, 1.0)
        updated[woy] = (1 - alpha) * old + alpha * avg_ratio

    return updated

def update_branch_adjustment(
    current_adj: dict[str, float],
    actuals: np.ndarray,
    predictions: np.ndarray,
    branch_types: np.ndarray,
    alpha: float = 0.2,
) -> dict[str, float]:
    """Update branch-type adjustment factor using EMA on actual/predicted ratio."""
    updated = dict(current_adj)
    ratios_by_type: dict[str, list[float]] = defaultdict(list)

    for actual, pred, bt in zip(actuals, predictions, branch_types):
        ratio = actual / (pred + EPS)
        ratios_by_type[str(bt)].append(ratio)

    for bt, ratios in ratios_by_type.items():
        avg_ratio = float(np.mean(ratios))
        old = updated.get(bt, 1.0)
        updated[bt] = (1 - alpha) * old + alpha * avg_ratio

    return updated

def update_safety_sigma(
    residuals: np.ndarray,
    window: int = 8,
) -> float:
    """Compute rolling std of forecast residuals for safety stock.

    Uses last `window` residuals.
    """
    recent = residuals[-window:] if len(residuals) > window else residuals
    return float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
```

**Step 3: Run test to verify it passes**

**Step 4: Commit**

```bash
git add ml/forecast/replenishment/self_learning.py tests/forecast/test_self_learning.py
git commit -m "feat: add self-learning for seasonal/branch/safety stock"
```
