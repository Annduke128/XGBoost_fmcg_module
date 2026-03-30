# FMCG Forecast – Plan 06: Forecast + Explainability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use skill({ name: "executing-plans" }) to implement this plan task-by-task.

**Goal:** Load trained model, forecast 1/2/4 weeks by scenario, explain with Permutation/SHAP/PDP.  
**Architecture:** Forecast pipeline loads artifact, builds features from shared defs; explain runs on validation.  
**Tech Stack:** xgboost, scikit-learn, shap, matplotlib, pytest.

---

## Must-Haves

### Observable Truths

1. Forecast output has columns: sku_id, branch_id, week, horizon, scenario, forecast_units.
2. Permutation importance computed on validation set (no leakage).
3. SHAP summary shows top drivers.
4. PDP shows price/promo/weather partial effects.

### Required Artifacts

| Artifact      | Provides                 | Path                                       |
| ------------- | ------------------------ | ------------------------------------------ |
| Forecast pipe | Multi-horizon prediction | `ml/forecast/pipelines/forecast_weekly.py` |
| Permutation   | Feature ranking          | `ml/forecast/explain/permutation.py`       |
| SHAP report   | Global/local explain     | `ml/forecast/explain/shap_report.py`       |
| PDP report    | Partial dependence       | `ml/forecast/explain/pdp_report.py`        |

---

## Task Dependencies

```
Task 1 (Forecast pipeline): needs Plan 05 (registry)
Task 2 (Permutation): needs Task 1
Task 3 (SHAP + PDP): needs Task 1

Wave 1: Task 1 (after Plan 05)
Wave 2: Task 2, Task 3 (parallel, after Task 1)
```

---

### Task 1: Forecast pipeline

**Files:**

- Create: `ml/forecast/__init__.py`
- Create: `ml/forecast/pipelines/__init__.py`
- Create: `ml/forecast/pipelines/forecast_weekly.py`
- Test: `tests/forecast/test_forecast_weekly.py`

**Step 1: Write failing test**

```python
# tests/forecast/test_forecast_weekly.py
from ml.forecast.pipelines.forecast_weekly import ForecastConfig

def test_forecast_config_defaults():
    cfg = ForecastConfig()
    assert cfg.horizons == [1, 2, 4]
    assert cfg.scenarios == ["A", "B"]
```

**Step 2: Implement**

```python
# ml/forecast/pipelines/forecast_weekly.py
"""Weekly forecast pipeline: load model, predict multi-horizon by scenario."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np

from ml.shared.schema import validate_columns
from ml.shared.features.encode import make_categorical
from ml.shared.features.feature_defs import CAT_COLS, ALL_FEATURES
from ml.training.features.build_features import build_all_features
from ml.training.data.scenario import build_scenarios
from ml.training.data.lead_time import assign_lead_time
from ml.training.models.model_registry import load_latest

@dataclass
class ForecastConfig:
    data_path: str = "data/weekly_sales.parquet"
    model_dir: str = "artifacts/models"
    output_path: str = "output/forecast.parquet"
    horizons: list[int] = field(default_factory=lambda: [1, 2, 4])
    scenarios: list[str] = field(default_factory=lambda: ["A", "B"])

def run_forecast(cfg: ForecastConfig | None = None) -> pd.DataFrame:
    """Run forecast for each horizon and scenario."""
    if cfg is None:
        cfg = ForecastConfig()

    model = load_latest(cfg.model_dir)
    df = pd.read_parquet(cfg.data_path)
    df = validate_columns(df)
    df["week"] = pd.to_datetime(df["week"])

    # Build scenarios
    scenarios = build_scenarios(df)
    results: list[pd.DataFrame] = []

    for scenario_name in cfg.scenarios:
        scenario_df = scenarios[scenario_name]

        # Add ema_sales_8w if not present
        if "ema_sales_8w" not in scenario_df.columns:
            scenario_df = scenario_df.sort_values(["sku_id", "branch_id", "week"])
            scenario_df["ema_sales_8w"] = (
                scenario_df.groupby(["sku_id", "branch_id"])["units"]
                .shift(1).ewm(span=8, adjust=False).mean()
            )

        scenario_df = assign_lead_time(scenario_df)
        scenario_df = build_all_features(scenario_df)
        scenario_df = scenario_df.dropna(subset=["lag_1"])

        feature_cols = [c for c in ALL_FEATURES if c in scenario_df.columns]
        scenario_df = make_categorical(scenario_df, [c for c in CAT_COLS if c in scenario_df.columns])

        for horizon in cfg.horizons:
            # Use the latest N weeks for horizon prediction
            max_week = scenario_df["week"].max()
            forecast_weeks = scenario_df[
                scenario_df["week"] > max_week - pd.Timedelta(weeks=horizon)
            ].copy()

            if len(forecast_weeks) == 0:
                continue

            X = forecast_weeks[feature_cols]
            preds = model.predict(X)

            out = forecast_weeks[["sku_id", "branch_id", "week"]].copy()
            out["horizon"] = horizon
            out["scenario"] = scenario_name
            out["forecast_units"] = np.maximum(preds, 0)
            results.append(out)

    result_df = pd.concat(results, ignore_index=True)

    # Save output
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)

    return result_df

if __name__ == "__main__":
    result = run_forecast()
    print(f"Forecast complete: {len(result)} rows, {result['scenario'].nunique()} scenarios")
```

**Step 3: Run test**

**Step 4: Commit**

```bash
git add ml/forecast/ tests/forecast/
git commit -m "feat: add multi-horizon multi-scenario forecast pipeline"
```

---

### Task 2: Permutation importance

**Files:**

- Create: `ml/forecast/explain/__init__.py`
- Create: `ml/forecast/explain/permutation.py`
- Test: `tests/forecast/test_permutation.py`

**Step 1: Write failing test**

```python
# tests/forecast/test_permutation.py
import numpy as np
from unittest.mock import MagicMock
from ml.forecast.explain.permutation import run_permutation

def test_run_permutation_returns_importances():
    # Mock model that returns random predictions
    model = MagicMock()
    model.predict = lambda X: np.random.rand(len(X))
    X_val = np.random.rand(50, 5)
    y_val = np.random.poisson(5, 50).astype(float)
    result = run_permutation(model, X_val, y_val, n_repeats=2)
    assert hasattr(result, "importances_mean")
```

**Step 2: Implement**

```python
# ml/forecast/explain/permutation.py
"""Feature importance via permutation on validation set."""
from __future__ import annotations
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from ml.shared.utils.metrics import wape

def _neg_wape(y_true, y_pred):
    return -wape(y_true, y_pred)

def run_permutation(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
):
    """Run permutation importance on validation data.

    Returns sklearn PermutationImportance result with:
    - importances_mean
    - importances_std
    - importances
    """
    scorer = make_scorer(_neg_wape)
    return permutation_importance(
        model, X_val, y_val,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
    )

def select_top_features(
    importance_result,
    feature_names: list[str],
    threshold: float = 0.0,
) -> list[str]:
    """Select features with importance > threshold."""
    mask = importance_result.importances_mean > threshold
    return [f for f, m in zip(feature_names, mask) if m]
```

**Step 3: Commit**

```bash
git add ml/forecast/explain/ tests/forecast/test_permutation.py
git commit -m "feat: add permutation importance feature selection"
```

---

### Task 3: SHAP + PDP reports

**Files:**

- Create: `ml/forecast/explain/shap_report.py`
- Create: `ml/forecast/explain/pdp_report.py`
- Test: `tests/forecast/test_shap_report.py`

**Step 1: Implement SHAP**

```python
# ml/forecast/explain/shap_report.py
"""SHAP explanations for XGBoost model."""
from __future__ import annotations
from pathlib import Path
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def shap_summary(
    model,
    X_sample,
    output_dir: str | Path = "output/explain",
    max_display: int = 20,
) -> None:
    """Generate SHAP summary plot and save to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150)
    plt.close()

    return shap_values

def shap_local(
    model,
    X_row,
    output_dir: str | Path = "output/explain",
) -> None:
    """Generate SHAP force plot for a single prediction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    shap.force_plot(
        explainer.expected_value, shap_values, X_row,
        matplotlib=True, show=False,
    )
    plt.savefig(output_dir / "shap_local.png", dpi=150, bbox_inches="tight")
    plt.close()
```

**Step 2: Implement PDP**

```python
# ml/forecast/explain/pdp_report.py
"""Partial Dependence Plots for key features."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def pdp_plot(
    model,
    X,
    features: list[str | int],
    output_dir: str | Path = "output/explain",
) -> None:
    """Generate PDP/ICE plots for specified features."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5 * ((len(features) + 1) // 2)))
    PartialDependenceDisplay.from_estimator(
        model, X, features,
        kind="both",  # PDP + ICE
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "pdp_plots.png", dpi=150)
    plt.close()
```

**Step 3: Commit**

```bash
git add ml/forecast/explain/ tests/forecast/test_shap_report.py
git commit -m "feat: add SHAP summary/local and PDP reports"
```
