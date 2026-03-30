# FMCG Forecast – Plan 09: Docker Packaging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use skill({ name: "executing-plans" }) to implement this plan task-by-task.

**Goal:** Package the training and forecast pipelines in Docker for reproducible deployment.  
**Architecture:** Single image, multiple entrypoints (train_weekly, forecast_weekly).  
**Tech Stack:** Docker, Python 3.11.

---

## Must-Haves

### Observable Truths

1. `docker build` succeeds without errors.
2. `docker run <image> train` executes training pipeline.
3. `docker run <image> forecast` executes forecast pipeline.
4. All dependencies installed correctly in container.

### Required Artifacts

| Artifact     | Provides            | Path               |
| ------------ | ------------------- | ------------------ |
| Requirements | Python dependencies | `requirements.txt` |
| Dockerfile   | Container build     | `Dockerfile`       |
| Entrypoint   | CLI dispatcher      | `ml/cli.py`        |

---

## Task Dependencies

```
Task 1: needs all prior plans complete
Wave 1: Task 1
```

---

### Task 1: Docker packaging

**Files:**

- Create: `requirements.txt`
- Create: `Dockerfile`
- Create: `ml/cli.py`
- Create: `.dockerignore`

**Step 1: Create requirements.txt**

```txt
pandas>=2.0,<3.0
numpy>=1.24,<2.0
scikit-learn>=1.3,<2.0
xgboost>=2.0,<3.0
optuna>=3.4,<4.0
shap>=0.43,<1.0
polars>=0.20,<2.0
matplotlib>=3.7,<4.0
joblib>=1.3,<2.0
pyarrow>=14.0,<18.0
```

**Step 2: Create CLI entrypoint**

```python
# ml/cli.py
"""CLI entrypoint for Docker container."""
from __future__ import annotations
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ml.cli [train|forecast]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        from ml.training.pipelines.train_weekly import run_train_weekly, TrainWeeklyConfig
        cfg = TrainWeeklyConfig()
        result = run_train_weekly(cfg)
        print(f"Training complete: WAPE={result['model_wape']:.4f}")

    elif command == "forecast":
        from ml.forecast.pipelines.forecast_weekly import run_forecast, ForecastConfig
        cfg = ForecastConfig()
        result = run_forecast(cfg)
        print(f"Forecast complete: {len(result)} rows")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Step 3: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY ml/ ml/
COPY data/ data/

# Create output directories
RUN mkdir -p artifacts/models output/explain output/forecast

# Default entrypoint
ENTRYPOINT ["python", "-m", "ml.cli"]
CMD ["train"]
```

**Step 4: Create .dockerignore**

```
__pycache__
*.pyc
.git
.beads
tests/
docs/
*.md
.env
```

**Step 5: Verify**

```bash
docker build -t fmcg-forecast .
docker run --rm fmcg-forecast train
docker run --rm fmcg-forecast forecast
```

**Step 6: Commit**

```bash
git add requirements.txt Dockerfile .dockerignore ml/cli.py
git commit -m "feat: add Docker packaging for train and forecast pipelines"
```
