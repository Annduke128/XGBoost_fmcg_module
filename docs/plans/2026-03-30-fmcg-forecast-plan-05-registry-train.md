# FMCG Forecast – Plan 05: Model Registry + Training Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use skill({ name: "executing-plans" }) to implement this plan task-by-task.

**Goal:** Save model artifacts (.pkl), rotate 8 versions, orchestrate weekly training pipeline.  
**Architecture:** joblib for serialization, JSON metadata, latest/ symlink pattern.  
**Tech Stack:** Python, joblib, json, pytest.

---

## Must-Haves

### Observable Truths

1. Model saved as `.pkl` with metadata JSON alongside.
2. Registry keeps max 8 versions, oldest deleted automatically.
3. `train_weekly.py` orchestrates: load → schema → split → stockout → lead_time → features → train → save.

### Required Artifacts

| Artifact       | Provides             | Path                                    |
| -------------- | -------------------- | --------------------------------------- |
| Model registry | Save/load/rotate     | `ml/training/models/model_registry.py`  |
| Train pipeline | Weekly orchestration | `ml/training/pipelines/train_weekly.py` |

---

## Task Dependencies

```
Task 1 (Registry): needs nothing
Task 2 (Train pipeline): needs Plans 01-04, Task 1

Wave 1: Task 1
Wave 2: Task 2 (after all prior plans + Task 1)
```

---

### Task 1: Model registry

**Files:**

- Create: `ml/training/models/model_registry.py`
- Test: `tests/training/test_model_registry.py`

**Step 1: Write failing test**

```python
# tests/training/test_model_registry.py
import tempfile
import json
from pathlib import Path
from ml.training.models.model_registry import save_model, load_latest, rotate_versions

def test_save_model_creates_files(tmp_path):
    model = {"dummy": True}
    metadata = {"wape": 0.15, "params": {"max_depth": 6}}
    save_model(model, metadata, base_dir=tmp_path, version="v001")
    assert (tmp_path / "v001" / "model.pkl").exists()
    assert (tmp_path / "v001" / "metadata.json").exists()
    assert (tmp_path / "latest").is_symlink() or (tmp_path / "latest").is_dir()

def test_rotate_keeps_max_versions(tmp_path):
    for i in range(10):
        v = f"v{i:03d}"
        (tmp_path / v).mkdir()
        (tmp_path / v / "model.pkl").touch()
    rotate_versions(tmp_path, max_versions=8)
    versions = sorted(p for p in tmp_path.iterdir() if p.name.startswith("v"))
    assert len(versions) == 8

def test_load_latest(tmp_path):
    model = {"test": 42}
    save_model(model, {}, base_dir=tmp_path, version="v001")
    loaded = load_latest(tmp_path)
    assert loaded["test"] == 42
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
# ml/training/models/model_registry.py
"""Model artifact registry: save, load, rotate .pkl models."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import joblib

def save_model(
    model: object,
    metadata: dict,
    base_dir: Path | str,
    version: str,
) -> Path:
    """Save model as .pkl with metadata JSON."""
    base_dir = Path(base_dir)
    version_dir = base_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, version_dir / "model.pkl")
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Update latest symlink
    latest = base_dir / "latest"
    if latest.is_symlink() or latest.exists():
        if latest.is_symlink():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    latest.symlink_to(version_dir.resolve())
    return version_dir

def load_latest(base_dir: Path | str) -> object:
    """Load model from latest version."""
    base_dir = Path(base_dir)
    latest = base_dir / "latest"
    if not latest.exists():
        raise FileNotFoundError(f"No latest model in {base_dir}")
    return joblib.load(latest / "model.pkl")

def rotate_versions(base_dir: Path | str, max_versions: int = 8) -> None:
    """Keep only the most recent `max_versions` versions."""
    base_dir = Path(base_dir)
    versions = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("v")],
        key=lambda p: p.name,
    )
    while len(versions) > max_versions:
        oldest = versions.pop(0)
        shutil.rmtree(oldest)
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add ml/training/models/model_registry.py tests/training/test_model_registry.py
git commit -m "feat: add model registry with save/load/rotate"
```

---

### Task 2: Weekly training pipeline

**Files:**

- Create: `ml/training/pipelines/__init__.py`
- Create: `ml/training/pipelines/train_weekly.py`
- Test: `tests/training/test_train_weekly.py`

**Step 1: Write integration test (smoke test)**

```python
# tests/training/test_train_weekly.py
from ml.training.pipelines.train_weekly import TrainWeeklyConfig

def test_config_defaults():
    cfg = TrainWeeklyConfig()
    assert cfg.val_weeks == 4
    assert cfg.test_weeks == 4
    assert cfg.max_model_versions == 8
    assert cfg.n_optuna_trials == 50
```

**Step 2: Implement**

```python
# ml/training/pipelines/train_weekly.py
"""Weekly training pipeline orchestrator."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from ml.shared.schema import validate_columns
from ml.shared.features.encode import make_categorical
from ml.shared.features.feature_defs import CAT_COLS, ALL_FEATURES
from ml.shared.utils.metrics import wape, mae
from ml.training.data.split import time_split
from ml.training.data.stockout import impute_stockout
from ml.training.data.lead_time import assign_lead_time
from ml.training.features.build_features import build_all_features
from ml.training.models.baselines import seasonal_naive
from ml.training.models.train_xgb import train_xgb_poisson
from ml.training.models.tune_optuna import run_optuna_study
from ml.training.models.model_registry import save_model, rotate_versions

@dataclass
class TrainWeeklyConfig:
    data_path: str = "data/weekly_sales.parquet"
    model_dir: str = "artifacts/models"
    val_weeks: int = 4
    test_weeks: int = 4
    max_model_versions: int = 8
    n_optuna_trials: int = 50
    run_tuning: bool = True

def run_train_weekly(cfg: TrainWeeklyConfig | None = None) -> dict:
    """Execute full weekly training pipeline.

    Returns dict with metrics and model version.
    """
    if cfg is None:
        cfg = TrainWeeklyConfig()

    # 1. Load data
    df = pd.read_parquet(cfg.data_path)
    df = validate_columns(df)
    df["week"] = pd.to_datetime(df["week"])

    # 2. Stockout impute
    df = impute_stockout(df, ["sku_id", "branch_id"])

    # 3. Lead time
    # Need ema_sales_8w — compute if not present
    if "ema_sales_8w" not in df.columns:
        df = df.sort_values(["sku_id", "branch_id", "week"])
        df["ema_sales_8w"] = (
            df.groupby(["sku_id", "branch_id"])["units"]
            .shift(1)
            .ewm(span=8, adjust=False)
            .mean()
        )
    df = assign_lead_time(df)

    # 4. Features
    df = build_all_features(df)
    df = df.dropna(subset=["lag_1"])  # drop rows without lag history

    # 5. Split
    train, val, test = time_split(df, cfg.val_weeks, cfg.test_weeks)

    # 6. Encode
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    train = make_categorical(train, [c for c in CAT_COLS if c in train.columns])
    val = make_categorical(val, [c for c in CAT_COLS if c in val.columns])

    X_train, y_train = train[feature_cols], train["units"].values
    X_val, y_val = val[feature_cols], val["units"].values

    # 7. Baseline
    baseline_preds = seasonal_naive(train, val)
    baseline_wape = wape(y_val, baseline_preds)

    # 8. Tuning (optional)
    if cfg.run_tuning:
        best_params = run_optuna_study(X_train, y_train, train["week"], n_trials=cfg.n_optuna_trials)
    else:
        best_params = None

    # 9. Train final model
    model = train_xgb_poisson(X_train, y_train, X_val, y_val, params=best_params)
    preds = model.predict(X_val)
    model_wape = wape(y_val, preds)
    model_mae = mae(y_val, preds)

    # 10. Save
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    metadata = {
        "version": version,
        "baseline_wape": float(baseline_wape),
        "model_wape": float(model_wape),
        "model_mae": float(model_mae),
        "best_params": best_params,
        "feature_cols": feature_cols,
        "n_train": len(train),
        "n_val": len(val),
    }
    model_dir = Path(cfg.model_dir)
    save_model(model, metadata, model_dir, version)
    rotate_versions(model_dir, cfg.max_model_versions)

    return metadata

if __name__ == "__main__":
    result = run_train_weekly()
    print(f"Training complete: WAPE={result['model_wape']:.4f} (baseline={result['baseline_wape']:.4f})")
```

**Step 3: Run test to verify it passes**

**Step 4: Commit**

```bash
git add ml/training/pipelines/ tests/training/test_train_weekly.py
git commit -m "feat: add weekly training pipeline orchestrator"
```
