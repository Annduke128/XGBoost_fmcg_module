# FMCG Forecast – Plan 02: Training Data Prep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use skill({ name: "executing-plans" }) to implement this plan task-by-task.

**Goal:** Time split + stockout censor & impute + lead-time rule + promo scenario generation.  
**Architecture:** Batch preprocessing, strict time-ordering to prevent leakage.  
**Tech Stack:** Python, pandas, numpy, pytest.

---

## Must-Haves

**Goal:** Clean, split, augmented training data ready for feature engineering.

### Observable Truths

1. Train/val/test split by time — no future data leaks into training.
2. Stockout weeks have demand imputed from rolling history.
3. Lead-time assigned by velocity rule (1/2/4 weeks).
4. Two promo scenarios generated (0% and 50% discount).

### Required Artifacts

| Artifact  | Provides              | Path                            |
| --------- | --------------------- | ------------------------------- |
| Split     | Time-based split      | `ml/training/data/split.py`     |
| Stockout  | Censor & impute       | `ml/training/data/stockout.py`  |
| Lead time | Rule-based assignment | `ml/training/data/lead_time.py` |
| Scenario  | Promo scenario gen    | `ml/training/data/scenario.py`  |

### Key Links

| From      | To       | Via                 | Risk                                  |
| --------- | -------- | ------------------- | ------------------------------------- |
| stockout  | features | imputed units       | leakage if not shift before rolling   |
| lead_time | reorder  | lead_time_weeks col | wrong assignment if display_units bad |

---

## Task Dependencies

```
Task 1 (Split): needs ml/shared/schema.py (Plan 01)
Task 2 (Stockout): needs nothing
Task 3 (Lead-time): needs nothing
Task 4 (Scenario): needs nothing

Wave 1: Task 1, Task 2, Task 3, Task 4 (parallel after Plan 01)
```

---

### Task 1: Time split

**Files:**

- Create: `ml/training/__init__.py`
- Create: `ml/training/data/__init__.py`
- Create: `ml/training/data/split.py`
- Test: `tests/training/test_split.py`

**Step 1: Write failing test**

```python
# tests/training/test_split.py
import pandas as pd
import pytest
from ml.training.data.split import time_split

def test_time_split_order():
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=20, freq="W"),
        "units": list(range(20)),
    })
    train, val, test = time_split(df, val_weeks=4, test_weeks=4)
    assert train["week"].max() < val["week"].min()
    assert val["week"].max() < test["week"].min()

def test_time_split_no_overlap():
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=20, freq="W"),
        "units": list(range(20)),
    })
    train, val, test = time_split(df, val_weeks=4, test_weeks=4)
    all_weeks = pd.concat([train["week"], val["week"], test["week"]])
    assert all_weeks.duplicated().sum() == 0

def test_time_split_covers_all():
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=20, freq="W"),
        "units": list(range(20)),
    })
    train, val, test = time_split(df, val_weeks=4, test_weeks=4)
    assert len(train) + len(val) + len(test) == len(df)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_split.py -q
```

**Step 3: Write minimal implementation**

```python
# ml/training/data/split.py
"""Time-based train/val/test split for weekly data."""
from __future__ import annotations
import pandas as pd

def time_split(
    df: pd.DataFrame,
    val_weeks: int = 4,
    test_weeks: int = 4,
    time_col: str = "week",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe by time. No shuffling — preserves temporal order."""
    df = df.sort_values(time_col).copy()
    max_week = df[time_col].max()
    test_cutoff = max_week - pd.Timedelta(weeks=test_weeks)
    val_cutoff = test_cutoff - pd.Timedelta(weeks=val_weeks)

    train = df[df[time_col] < val_cutoff].copy()
    val = df[(df[time_col] >= val_cutoff) & (df[time_col] < test_cutoff)].copy()
    test = df[df[time_col] >= test_cutoff].copy()
    return train, val, test
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/training/test_split.py -q
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add ml/training/ tests/training/test_split.py
git commit -m "feat: add time-based train/val/test split"
```

---

### Task 2: Stockout censor & impute

**Files:**

- Create: `ml/training/data/stockout.py`
- Test: `tests/training/test_stockout.py`

**Step 1: Write failing test**

```python
# tests/training/test_stockout.py
import pandas as pd
import numpy as np
from ml.training.data.stockout import impute_stockout

def test_impute_stockout_replaces_zero():
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=6, freq="W"),
        "sku_id": ["A"] * 6,
        "branch_id": ["B"] * 6,
        "units": [10, 12, 0, 11, 13, 0],
        "stockout_flag": [0, 0, 1, 0, 0, 1],
    })
    result = impute_stockout(df, ["sku_id", "branch_id"])
    # Stockout rows should have imputed (non-zero) values
    assert result.loc[result["stockout_flag"] == 1, "units"].min() > 0

def test_impute_stockout_no_change_non_stockout():
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=4, freq="W"),
        "sku_id": ["A"] * 4,
        "branch_id": ["B"] * 4,
        "units": [10, 12, 14, 16],
        "stockout_flag": [0, 0, 0, 0],
    })
    result = impute_stockout(df, ["sku_id", "branch_id"])
    pd.testing.assert_series_equal(result["units"], df["units"])
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_stockout.py -q
```

**Step 3: Write minimal implementation**

```python
# ml/training/data/stockout.py
"""Censor & impute demand during stockout periods."""
from __future__ import annotations
import pandas as pd

def impute_stockout(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "units",
    stockout_col: str = "stockout_flag",
    window: int = 4,
) -> pd.DataFrame:
    """Replace units during stockout with rolling median of prior non-stockout weeks.

    Uses shift(1) to avoid leakage — only past values used.
    """
    df = df.sort_values("week").copy()
    # Rolling median on shifted (past-only) values within each group
    roll_med = (
        df.groupby(group_cols)[target_col]
        .shift(1)
        .rolling(window, min_periods=1)
        .median()
    )
    mask = df[stockout_col] == 1
    df.loc[mask, target_col] = roll_med[mask]
    # Fallback: if still NaN (e.g. first row), use group median
    if df.loc[mask, target_col].isna().any():
        group_med = df.groupby(group_cols)[target_col].transform("median")
        still_na = mask & df[target_col].isna()
        df.loc[still_na, target_col] = group_med[still_na]
    return df
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/training/test_stockout.py -q
```

**Step 5: Commit**

```bash
git add ml/training/data/stockout.py tests/training/test_stockout.py
git commit -m "feat: add stockout censor and impute"
```

---

### Task 3: Lead-time rule

**Files:**

- Create: `ml/training/data/lead_time.py`
- Test: `tests/training/test_lead_time.py`

**Step 1: Write failing test**

```python
# tests/training/test_lead_time.py
import pandas as pd
from ml.training.data.lead_time import assign_lead_time

def test_lead_time_fast_mover():
    """High velocity (cover <= 1 week) -> lead_time = 1."""
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=1, freq="W"),
        "sku_id": ["A"], "branch_id": ["B"],
        "display_units": [10],
        "ema_sales_8w": [15.0],  # cover = 10/15 = 0.67
    })
    result = assign_lead_time(df)
    assert result["lead_time_weeks"].iloc[0] == 1

def test_lead_time_medium_mover():
    """Medium velocity (1 < cover <= 2) -> lead_time = 2."""
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=1, freq="W"),
        "sku_id": ["A"], "branch_id": ["B"],
        "display_units": [10],
        "ema_sales_8w": [6.0],  # cover = 10/6 = 1.67
    })
    result = assign_lead_time(df)
    assert result["lead_time_weeks"].iloc[0] == 2

def test_lead_time_slow_mover():
    """Low velocity (cover > 2) -> lead_time = 4."""
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=1, freq="W"),
        "sku_id": ["A"], "branch_id": ["B"],
        "display_units": [10],
        "ema_sales_8w": [3.0],  # cover = 10/3 = 3.33
    })
    result = assign_lead_time(df)
    assert result["lead_time_weeks"].iloc[0] == 4

def test_lead_time_zero_sales():
    """Zero sales -> max lead_time = 4."""
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=1, freq="W"),
        "sku_id": ["A"], "branch_id": ["B"],
        "display_units": [10],
        "ema_sales_8w": [0.0],
    })
    result = assign_lead_time(df)
    assert result["lead_time_weeks"].iloc[0] == 4
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
# ml/training/data/lead_time.py
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
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add ml/training/data/lead_time.py tests/training/test_lead_time.py
git commit -m "feat: add lead-time rule assignment (1/2/4 weeks)"
```

---

### Task 4: Promo scenario generation

**Files:**

- Create: `ml/training/data/scenario.py`
- Test: `tests/training/test_scenario.py`

**Step 1: Write failing test**

```python
# tests/training/test_scenario.py
import pandas as pd
from ml.training.data.scenario import build_scenarios

def test_build_scenarios_two_scenarios():
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=2, freq="W"),
        "sku_id": ["A", "A"],
        "price": [100.0, 100.0],
        "promo_flag": [0, 0],
    })
    scenarios = build_scenarios(df)
    assert "A" in scenarios  # scenario A: no promo
    assert "B" in scenarios  # scenario B: 50% discount

def test_scenario_b_price_halved():
    df = pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=1, freq="W"),
        "sku_id": ["A"],
        "price": [100.0],
        "promo_flag": [0],
    })
    scenarios = build_scenarios(df)
    assert scenarios["B"]["price"].iloc[0] == 50.0
    assert scenarios["B"]["promo_flag"].iloc[0] == 1
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
# ml/training/data/scenario.py
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
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add ml/training/data/scenario.py tests/training/test_scenario.py
git commit -m "feat: add promo scenario generation (0% and 50%)"
```
