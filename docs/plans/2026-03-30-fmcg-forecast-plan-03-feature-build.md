# FMCG Forecast – Plan 03: Feature Build Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use skill({ name: "executing-plans" }) to implement this plan task-by-task.

**Goal:** Build time/seasonal + lag/rolling/EMA features without leakage.  
**Architecture:** Uses shared feature defs from Plan 01; all temporal features use shift(1)+ to prevent future leakage.  
**Tech Stack:** Python, pandas, numpy, pytest.

---

## Must-Haves

**Goal:** Complete feature matrix ready for model training.

### Observable Truths

1. Lag features use shift(N) — never include current or future values.
2. Rolling/EMA computed on shifted series only.
3. Seasonal features (Fourier) are deterministic from week number.
4. Feature names match `ml/shared/features/feature_defs.py`.

### Required Artifacts

| Artifact        | Provides                     | Path                                     |
| --------------- | ---------------------------- | ---------------------------------------- |
| Feature builder | All features in one pipeline | `ml/training/features/build_features.py` |

### Key Links

| From            | To             | Via       | Risk                                  |
| --------------- | -------------- | --------- | ------------------------------------- |
| shared defs     | build_features | import    | name mismatch                         |
| stockout impute | build_features | units col | leakage if lag computed before impute |

---

## Task Dependencies

```
Task 1 (Feature builder): needs Plan 01 (feature_defs), Plan 02 (stockout)
Wave 1: Task 1 (after Plan 01 + 02)
```

---

### Task 1: Time/seasonal features

**Files:**

- Create: `ml/training/features/__init__.py`
- Create: `ml/training/features/build_features.py`
- Test: `tests/training/test_build_features.py`

**Step 1: Write failing test**

```python
# tests/training/test_build_features.py
import pandas as pd
import numpy as np
from ml.training.features.build_features import (
    add_time_features,
    add_lag_features,
    build_all_features,
)

def _make_df(n=10):
    return pd.DataFrame({
        "week": pd.date_range("2025-01-06", periods=n, freq="W"),
        "sku_id": ["A"] * n,
        "branch_id": ["B"] * n,
        "units": list(range(1, n + 1)),
        "price": [100.0] * n,
        "promo_flag": [0] * n,
        "brand_type": ["X"] * n,
        "branch_type": ["Y"] * n,
        "stockout_flag": [0] * n,
        "display_units": [20] * n,
        "channel": ["MT"] * n,
        "store_type": ["S1"] * n,
        "display_capacity_type": ["D1"] * n,
        "service_scale": ["L"] * n,
    })

def test_time_features_created():
    df = _make_df()
    result = add_time_features(df)
    assert "weekofyear" in result.columns
    assert "sin_woy" in result.columns
    assert "cos_woy" in result.columns

def test_lag_features_no_leakage():
    """lag_1 at row i should equal units at row i-1."""
    df = _make_df()
    result = add_lag_features(df)
    assert pd.isna(result["lag_1"].iloc[0])  # first row has no history
    assert result["lag_1"].iloc[1] == df["units"].iloc[0]

def test_rolling_no_leakage():
    df = _make_df()
    result = add_lag_features(df)
    # roll_4_mean at row 1 should only use row 0
    assert result["roll_4_mean"].iloc[1] == df["units"].iloc[0]

def test_build_all_features_output_shape():
    df = _make_df(20)
    result = build_all_features(df)
    assert "lag_1" in result.columns
    assert "sin_woy" in result.columns
    assert "ema_4" in result.columns
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_build_features.py -q
```

**Step 3: Write minimal implementation**

```python
# ml/training/features/build_features.py
"""Feature engineering pipeline for FMCG weekly forecast."""
from __future__ import annotations
import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic time/seasonal features from week column."""
    df = df.copy()
    df["weekofyear"] = df["week"].dt.isocalendar().week.astype(int)
    df["month"] = df["week"].dt.month
    df["sin_woy"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["cos_woy"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    return df

def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "units",
) -> pd.DataFrame:
    """Add lag, rolling mean, and EMA features.

    All computed on shift(1)+ to prevent leakage.
    """
    if group_cols is None:
        group_cols = ["sku_id", "branch_id"]

    df = df.sort_values(["sku_id", "branch_id", "week"]).copy()
    grp = df.groupby(group_cols)[target_col]

    # Lags
    for lag in [1, 2, 4]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling mean on shifted series (past only)
    shifted = grp.shift(1)
    df["roll_4_mean"] = shifted.rolling(4, min_periods=1).mean()

    # EMA on shifted series
    df["ema_4"] = shifted.ewm(span=4, adjust=False).mean()

    return df

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    df = add_time_features(df)
    df = add_lag_features(df)
    return df
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/training/test_build_features.py -q
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add ml/training/features/ tests/training/test_build_features.py
git commit -m "feat: add feature builder (time/seasonal/lag/rolling/EMA)"
```
