# FMCG Forecast – Plan 04: Modeling + Optuna Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use skill({ name: "executing-plans" }) to implement this plan task-by-task.

**Goal:** Train baselines + XGBoost Poisson + Optuna hyperparameter tuning.  
**Architecture:** Time-series CV, XGBoost native categorical for high cardinality (~10k).  
**Tech Stack:** scikit-learn, xgboost, optuna, pytest.

---

## Must-Haves

**Goal:** XGBoost Poisson model that beats baselines on WAPE/MAE.

### Observable Truths

1. Baselines (seasonal naive, PoissonRegressor) provide benchmark.
2. XGBoost Poisson uses enable_categorical=True (no one-hot explosion).
3. Optuna finds stable hyperparams across time-series folds.
4. Model beats baseline on validation WAPE by measurable margin.

### Required Artifacts

| Artifact  | Provides             | Path                                |
| --------- | -------------------- | ----------------------------------- |
| Baselines | Benchmark models     | `ml/training/models/baselines.py`   |
| XGB train | Poisson trainer      | `ml/training/models/train_xgb.py`   |
| Optuna    | Hyperparameter tuner | `ml/training/models/tune_optuna.py` |

### Key Links

| From     | To       | Via                | Risk                               |
| -------- | -------- | ------------------ | ---------------------------------- |
| encoder  | XGB      | enable_categorical | ordinal bias if dtype not category |
| features | training | feature matrix     | shape mismatch train vs val        |

---

## Task Dependencies

```
Task 1 (Baselines): needs Plan 01 (metrics)
Task 2 (XGB Poisson): needs Plan 01 (encode), Plan 03 (features)
Task 3 (Optuna): needs Task 2

Wave 1: Task 1 (after Plan 01)
Wave 2: Task 2 (after Plan 01 + 03)
Wave 3: Task 3 (after Task 2)
```

---

### Task 1: Baselines

**Files:**

- Create: `ml/training/models/__init__.py`
- Create: `ml/training/models/baselines.py`
- Test: `tests/training/test_baselines.py`

**Step 1: Write failing test**

```python
# tests/training/test_baselines.py
import pandas as pd
import numpy as np
from ml.training.models.baselines import seasonal_naive, poisson_baseline

def test_seasonal_naive():
    # Last year same week = lag_4 for monthly proxy
    train = pd.DataFrame({
        "sku_id": ["A"] * 8,
        "branch_id": ["B"] * 8,
        "week": pd.date_range("2025-01-06", periods=8, freq="W"),
        "units": [10, 12, 14, 16, 18, 20, 22, 24],
    })
    val = pd.DataFrame({
        "sku_id": ["A"] * 2,
        "branch_id": ["B"] * 2,
        "week": pd.date_range("2025-03-03", periods=2, freq="W"),
    })
    preds = seasonal_naive(train, val)
    assert len(preds) == 2
    assert all(p >= 0 for p in preds)

def test_poisson_baseline():
    np.random.seed(42)
    X_train = np.random.rand(50, 3)
    y_train = np.random.poisson(5, 50).astype(float)
    X_val = np.random.rand(10, 3)
    preds = poisson_baseline(X_train, y_train, X_val)
    assert len(preds) == 10
    assert all(p >= 0 for p in preds)
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
# ml/training/models/baselines.py
"""Baseline models for FMCG demand forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

def seasonal_naive(
    train: pd.DataFrame,
    val: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "units",
    lag_weeks: int = 4,
) -> np.ndarray:
    """Seasonal naive: predict using value from `lag_weeks` ago."""
    if group_cols is None:
        group_cols = ["sku_id", "branch_id"]
    # Get last known value per group
    last_vals = (
        train.sort_values("week")
        .groupby(group_cols)[target_col]
        .last()
    )
    preds = val.merge(
        last_vals.reset_index().rename(columns={target_col: "pred"}),
        on=group_cols,
        how="left",
    )["pred"].fillna(0).values
    return np.maximum(preds, 0)

def poisson_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
) -> np.ndarray:
    """Scikit-learn PoissonRegressor baseline."""
    model = PoissonRegressor(alpha=1.0, max_iter=300)
    model.fit(X_train, y_train)
    return model.predict(X_val)
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add ml/training/models/ tests/training/test_baselines.py
git commit -m "feat: add seasonal naive and Poisson baselines"
```

---

### Task 2: XGBoost Poisson trainer

**Files:**

- Create: `ml/training/models/train_xgb.py`
- Test: `tests/training/test_train_xgb.py`

**Step 1: Write failing test**

```python
# tests/training/test_train_xgb.py
import numpy as np
import pandas as pd
from ml.training.models.train_xgb import train_xgb_poisson

def test_train_xgb_poisson_returns_model():
    np.random.seed(42)
    n = 100
    X_train = pd.DataFrame({
        "f1": np.random.rand(n),
        "f2": np.random.rand(n),
        "cat1": pd.Categorical(np.random.choice(["a", "b", "c"], n)),
    })
    y_train = np.random.poisson(5, n).astype(float)
    X_val = X_train.iloc[:10].copy()
    y_val = y_train[:10]
    model = train_xgb_poisson(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert len(preds) == 10
    assert all(p >= 0 for p in preds)
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
# ml/training/models/train_xgb.py
"""XGBoost Poisson regression trainer."""
from __future__ import annotations
import xgboost as xgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS: dict = {
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "max_delta_step": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

def train_xgb_poisson(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict | None = None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 30,
) -> xgb.XGBRegressor:
    """Train XGBoost with Poisson objective and native categorical support."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    model = xgb.XGBRegressor(
        objective="reg:poisson",
        n_estimators=n_estimators,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add ml/training/models/train_xgb.py tests/training/test_train_xgb.py
git commit -m "feat: add XGBoost Poisson trainer with native categorical"
```

---

### Task 3: Optuna hyperparameter tuning

**Files:**

- Create: `ml/training/models/tune_optuna.py`
- Test: `tests/training/test_tune_optuna.py`

**Step 1: Write failing test**

```python
# tests/training/test_tune_optuna.py
import numpy as np
import pandas as pd
from ml.training.models.tune_optuna import run_optuna_study

def test_optuna_returns_best_params():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "f1": np.random.rand(n),
        "f2": np.random.rand(n),
    })
    y = np.random.poisson(5, n).astype(float)
    weeks = pd.date_range("2025-01-06", periods=n, freq="W")
    best_params = run_optuna_study(X, y, weeks, n_trials=3)
    assert "max_depth" in best_params
    assert "eta" in best_params
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
# ml/training/models/tune_optuna.py
"""Optuna hyperparameter tuning for XGBoost Poisson."""
from __future__ import annotations
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from ml.shared.utils.metrics import wape
from ml.training.models.train_xgb import train_xgb_poisson

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    weeks: pd.Series,
    n_splits: int = 3,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        model = train_xgb_poisson(X_tr, y_tr, X_va, y_va, params=params, n_estimators=200)
        preds = model.predict(X_va)
        scores.append(wape(y_va, preds))

    return float(np.mean(scores))

def run_optuna_study(
    X: pd.DataFrame,
    y: np.ndarray,
    weeks: pd.Series | pd.DatetimeIndex,
    n_trials: int = 50,
    n_splits: int = 3,
) -> dict:
    """Run Optuna study and return best hyperparameters."""
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, X, y, weeks, n_splits),
        n_trials=n_trials,
    )
    return study.best_params
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add ml/training/models/tune_optuna.py tests/training/test_tune_optuna.py
git commit -m "feat: add Optuna tuning for XGBoost Poisson"
```
