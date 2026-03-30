# FMCG Forecast – Plan 01: Shared Foundations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use skill({ name: "executing-plans" }) to implement this plan task-by-task.

**Goal:** Chuẩn hóa schema, metrics và feature/encoding dùng chung cho training & forecast.  
**Architecture:** Shared module đảm bảo consistency và tránh mismatch giữa train/forecast.  
**Tech Stack:** Python, pandas/numpy, scikit-learn, pytest.

---

## Must-Haves

**Goal:** Shared foundations sẵn sàng cho toàn pipeline.

### Observable Truths

1. Dữ liệu không đúng schema sẽ bị chặn.
2. Metrics (WAPE/MAE/MAPE/MDAPE) chuẩn, xử lý y=0.
3. Feature defs + encoding thống nhất giữa training và forecast.

### Required Artifacts

| Artifact     | Provides               | Path                                 |
| ------------ | ---------------------- | ------------------------------------ |
| Schema       | Validate required cols | `ml/shared/schema.py`                |
| Metrics      | WAPE/MAE/MAPE/MDAPE    | `ml/shared/utils/metrics.py`         |
| Feature defs | canonical feature list | `ml/shared/features/feature_defs.py` |
| Encoder      | categorical casting    | `ml/shared/features/encode.py`       |

### Key Links

| From           | To          | Via     | Risk             |
| -------------- | ----------- | ------- | ---------------- |
| train/forecast | shared defs | imports | mismatch columns |

---

## Task Dependencies

```
Task 1 (Schema): needs nothing, creates ml/shared/schema.py
Task 2 (Metrics): needs nothing, creates ml/shared/utils/metrics.py
Task 3 (Feature defs + Encode): needs nothing, creates ml/shared/features/

Wave 1: Task 1, Task 2, Task 3 (parallel — no dependencies)
```

---

### Task 1: Schema validation

**Files:**

- Create: `ml/shared/__init__.py`
- Create: `ml/shared/schema.py`
- Test: `tests/shared/test_schema.py`

**Step 1: Write failing test**

```python
# tests/shared/test_schema.py
import pandas as pd
import pytest
from ml.shared.schema import validate_columns, REQUIRED_COLS

def test_validate_columns_pass():
    data = {c: ["x"] for c in REQUIRED_COLS}
    df = pd.DataFrame(data)
    result = validate_columns(df)
    assert len(result) == 1

def test_validate_columns_missing():
    df = pd.DataFrame({"week": ["2025-01-06"]})
    with pytest.raises(ValueError, match="Missing columns"):
        validate_columns(df)

def test_validate_columns_empty_df():
    data = {c: [] for c in REQUIRED_COLS}
    df = pd.DataFrame(data)
    result = validate_columns(df)
    assert len(result) == 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/shared/test_schema.py -q
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# ml/shared/schema.py
"""Data schema validation for FMCG forecast pipeline."""
from __future__ import annotations
import pandas as pd

REQUIRED_COLS: list[str] = [
    "week",
    "sku_id",
    "branch_id",
    "units",
    "price",
    "promo_flag",
    "brand_type",
    "branch_type",
    "stockout_flag",
    "display_units",
]

def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that all required columns are present.

    Raises ValueError if any required column is missing.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/shared/test_schema.py -q
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add ml/shared/ tests/shared/test_schema.py
git commit -m "feat: add shared schema validation"
```

---

### Task 2: Metrics (WAPE/MAE/MAPE/MDAPE)

**Files:**

- Create: `ml/shared/utils/__init__.py`
- Create: `ml/shared/utils/metrics.py`
- Test: `tests/shared/test_metrics.py`

**Step 1: Write failing test**

```python
# tests/shared/test_metrics.py
import numpy as np
import pytest
from ml.shared.utils.metrics import wape, mae, mape, mdape

def test_wape_basic():
    y = np.array([10, 10])
    yhat = np.array([8, 12])
    result = wape(y, yhat)
    assert abs(result - 0.2) < 1e-6

def test_mae_basic():
    y = np.array([10, 20])
    yhat = np.array([12, 18])
    assert abs(mae(y, yhat) - 2.0) < 1e-6

def test_mape_with_zero():
    y = np.array([10, 0, 20])
    yhat = np.array([9, 1, 22])
    result = mape(y, yhat)
    assert result >= 0

def test_mdape_with_zero():
    y = np.array([10, 0, 20])
    yhat = np.array([9, 1, 22])
    result = mdape(y, yhat)
    assert result >= 0

def test_wape_all_zero():
    y = np.array([0, 0])
    yhat = np.array([1, 1])
    result = wape(y, yhat)
    assert result > 0  # should not divide by zero
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/shared/test_metrics.py -q
```

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# ml/shared/utils/metrics.py
"""Forecast evaluation metrics for FMCG pipeline."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

EPS: float = 1e-6

def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true)) + EPS
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Percentage Error (epsilon-safe for y=0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def mdape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Median Absolute Percentage Error (epsilon-safe for y=0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.median(np.abs((y_true - y_pred) / denom)))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/shared/test_metrics.py -q
```

Expected: 5 passed

**Step 5: Commit**

```bash
git add ml/shared/utils/ tests/shared/test_metrics.py
git commit -m "feat: add WAPE/MAE/MAPE/MDAPE metrics"
```

---

### Task 3: Feature defs + Encoding

**Files:**

- Create: `ml/shared/features/__init__.py`
- Create: `ml/shared/features/feature_defs.py`
- Create: `ml/shared/features/encode.py`
- Test: `tests/shared/test_feature_defs.py`
- Test: `tests/shared/test_encode.py`

**Step 1: Write failing tests**

```python
# tests/shared/test_feature_defs.py
from ml.shared.features.feature_defs import CAT_COLS, NUM_COLS, ALL_FEATURES

def test_all_features_is_union():
    assert set(ALL_FEATURES) == set(CAT_COLS) | set(NUM_COLS)

def test_no_duplicates():
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES))
```

```python
# tests/shared/test_encode.py
import pandas as pd
from ml.shared.features.encode import make_categorical
from ml.shared.features.feature_defs import CAT_COLS

def test_make_categorical():
    df = pd.DataFrame({c: ["a", "b"] for c in CAT_COLS})
    result = make_categorical(df, CAT_COLS)
    for c in CAT_COLS:
        assert result[c].dtype.name == "category"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/shared/test_feature_defs.py tests/shared/test_encode.py -q
```

**Step 3: Write minimal implementation**

```python
# ml/shared/features/feature_defs.py
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
]

NUM_COLS: list[str] = [
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

ALL_FEATURES: list[str] = CAT_COLS + NUM_COLS
```

```python
# ml/shared/features/encode.py
"""Encoding utilities for high-cardinality categorical features."""
from __future__ import annotations
import pandas as pd

def make_categorical(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Cast columns to pandas Categorical dtype for XGBoost native support."""
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/shared/test_feature_defs.py tests/shared/test_encode.py -q
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add ml/shared/features/ tests/shared/test_feature_defs.py tests/shared/test_encode.py
git commit -m "feat: add shared feature defs and categorical encoder"
```
