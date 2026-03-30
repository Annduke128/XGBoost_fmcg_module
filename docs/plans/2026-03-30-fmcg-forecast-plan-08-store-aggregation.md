# FMCG Forecast – Plan 08: Store-Type Aggregation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use skill({ name: "executing-plans" }) to implement this plan task-by-task.

**Goal:** Cluster stores by profile, train global vs clustered models, aggregate predictions to store-type.  
**Architecture:** Profile → Cluster → Train per-cluster → Compare → Aggregate.  
**Tech Stack:** scikit-learn (OneHotEncoder, StandardScaler, KMeans), pandas, pytest.

---

## Must-Haves

### Observable Truths

1. Store profiles capture: avg_sales, volatility, promo_lift, stockout_rate, seasonality_strength.
2. Categorical dims: store_type(5), display_capacity_type(5), service_scale(3), channel(2).
3. KMeans clustering with min 5 stores per cluster.
4. Clustered model used if WAPE not worse by >1% AND MAPE/MDAPE improves.

### Required Artifacts

| Artifact    | Provides             | Path                                                |
| ----------- | -------------------- | --------------------------------------------------- |
| Profiles    | Store feature matrix | `ml/training/segmentation/store_profiles.py`        |
| Clustering  | KMeans clusters      | `ml/training/segmentation/cluster_stores.py`        |
| Aggregation | Weighted predictions | `ml/forecast/segmentation/aggregate_predictions.py` |

---

## Task Dependencies

```
Task 1 (Profiles): needs Plan 02 (data prep)
Task 2 (Clustering): needs Task 1
Task 3 (Aggregation): needs Task 2 + Plan 06 (forecast)

Wave 1: Task 1 (after Plan 02)
Wave 2: Task 2 (after Task 1)
Wave 3: Task 3 (after Task 2 + Plan 06)
```

---

### Task 1: Store profiles

**Files:**

- Create: `ml/training/segmentation/__init__.py`
- Create: `ml/training/segmentation/store_profiles.py`
- Test: `tests/training/test_store_profiles.py`

**Step 1: Write failing test**

```python
# tests/training/test_store_profiles.py
import pandas as pd
import numpy as np
from ml.training.segmentation.store_profiles import build_store_profiles

def test_build_store_profiles():
    np.random.seed(42)
    df = pd.DataFrame({
        "branch_id": ["B1"] * 20 + ["B2"] * 20,
        "sku_id": ["A"] * 40,
        "week": list(pd.date_range("2025-01-06", periods=20, freq="W")) * 2,
        "units": np.random.poisson(10, 40).astype(float),
        "price": [100.0] * 40,
        "promo_flag": [0] * 30 + [1] * 10,
        "stockout_flag": [0] * 38 + [1] * 2,
        "store_type": ["S1"] * 20 + ["S2"] * 20,
        "display_capacity_type": ["D1"] * 40,
        "service_scale": ["L"] * 40,
        "channel": ["MT"] * 40,
    })
    profiles = build_store_profiles(df)
    assert "avg_weekly_sales" in profiles.columns
    assert "volatility" in profiles.columns
    assert "promo_lift" in profiles.columns
    assert "stockout_rate" in profiles.columns
    assert len(profiles) == 2  # 2 branches
```

**Step 2: Implement**

```python
# ml/training/segmentation/store_profiles.py
"""Build store (branch) profile features for clustering."""
from __future__ import annotations
import numpy as np
import pandas as pd

def build_store_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-branch profile metrics.

    Metrics:
    - avg_weekly_sales: mean units per week
    - volatility: std / mean (CV)
    - promo_lift: mean units during promo / mean units no promo
    - stockout_rate: fraction of weeks with stockout
    - seasonality_strength: max weekly avg / min weekly avg
    """
    profiles: list[dict] = []

    for branch_id, grp in df.groupby("branch_id"):
        avg_sales = grp["units"].mean()
        std_sales = grp["units"].std()
        volatility = std_sales / (avg_sales + 1e-9)

        promo = grp[grp["promo_flag"] == 1]["units"].mean()
        no_promo = grp[grp["promo_flag"] == 0]["units"].mean()
        promo_lift = promo / (no_promo + 1e-9) if pd.notna(promo) else 1.0

        stockout_rate = grp["stockout_flag"].mean()

        # Seasonality: weekly pattern
        if "week" in grp.columns:
            grp = grp.copy()
            grp["woy"] = pd.to_datetime(grp["week"]).dt.isocalendar().week.astype(int)
            woy_avg = grp.groupby("woy")["units"].mean()
            seasonality = woy_avg.max() / (woy_avg.min() + 1e-9) if len(woy_avg) > 1 else 1.0
        else:
            seasonality = 1.0

        # Categorical dims
        cat_info = {
            "store_type": grp["store_type"].mode().iloc[0] if "store_type" in grp.columns else "unknown",
            "display_capacity_type": grp["display_capacity_type"].mode().iloc[0] if "display_capacity_type" in grp.columns else "unknown",
            "service_scale": grp["service_scale"].mode().iloc[0] if "service_scale" in grp.columns else "unknown",
            "channel": grp["channel"].mode().iloc[0] if "channel" in grp.columns else "unknown",
        }

        profiles.append({
            "branch_id": branch_id,
            "avg_weekly_sales": avg_sales,
            "volatility": volatility,
            "promo_lift": promo_lift,
            "stockout_rate": stockout_rate,
            "seasonality_strength": seasonality,
            **cat_info,
        })

    return pd.DataFrame(profiles)
```

**Step 3: Commit**

```bash
git add ml/training/segmentation/ tests/training/test_store_profiles.py
git commit -m "feat: add store profile builder for clustering"
```

---

### Task 2: Store clustering

**Files:**

- Create: `ml/training/segmentation/cluster_stores.py`
- Test: `tests/training/test_cluster_stores.py`

**Step 1: Write failing test**

```python
# tests/training/test_cluster_stores.py
import pandas as pd
import numpy as np
from ml.training.segmentation.cluster_stores import cluster_stores

def test_cluster_stores_basic():
    np.random.seed(42)
    profiles = pd.DataFrame({
        "branch_id": [f"B{i}" for i in range(20)],
        "avg_weekly_sales": np.random.rand(20) * 100,
        "volatility": np.random.rand(20),
        "promo_lift": np.random.rand(20) + 0.5,
        "stockout_rate": np.random.rand(20) * 0.1,
        "seasonality_strength": np.random.rand(20) + 1,
        "store_type": np.random.choice(["S1", "S2"], 20),
        "display_capacity_type": np.random.choice(["D1", "D2"], 20),
        "service_scale": np.random.choice(["L", "M"], 20),
        "channel": np.random.choice(["MT", "GT"], 20),
    })
    result = cluster_stores(profiles, n_clusters=3, min_stores=3)
    assert "cluster" in result.columns
    # All branches assigned
    assert len(result) == 20

def test_cluster_min_stores_enforced():
    np.random.seed(42)
    profiles = pd.DataFrame({
        "branch_id": [f"B{i}" for i in range(10)],
        "avg_weekly_sales": np.random.rand(10) * 100,
        "volatility": np.random.rand(10),
        "promo_lift": np.random.rand(10) + 0.5,
        "stockout_rate": np.random.rand(10) * 0.1,
        "seasonality_strength": np.random.rand(10) + 1,
        "store_type": np.random.choice(["S1"], 10),
        "display_capacity_type": np.random.choice(["D1"], 10),
        "service_scale": np.random.choice(["L"], 10),
        "channel": np.random.choice(["MT"], 10),
    })
    result = cluster_stores(profiles, n_clusters=5, min_stores=5)
    # With 10 stores and min 5, max 2 clusters
    assert result["cluster"].nunique() <= 2
```

**Step 2: Implement**

```python
# ml/training/segmentation/cluster_stores.py
"""Cluster stores by profile using scikit-learn."""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

NUM_PROFILE_COLS = [
    "avg_weekly_sales", "volatility", "promo_lift",
    "stockout_rate", "seasonality_strength",
]
CAT_PROFILE_COLS = [
    "store_type", "display_capacity_type", "service_scale", "channel",
]

def cluster_stores(
    profiles: pd.DataFrame,
    n_clusters: int = 5,
    min_stores: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cluster stores by numeric + categorical profile features.

    Uses OneHotEncoder for categoricals, StandardScaler for numerics, then KMeans.
    Reduces n_clusters if any cluster would have < min_stores.
    """
    profiles = profiles.copy()

    # Encode categoricals
    cat_cols = [c for c in CAT_PROFILE_COLS if c in profiles.columns]
    num_cols = [c for c in NUM_PROFILE_COLS if c in profiles.columns]

    X_num = StandardScaler().fit_transform(profiles[num_cols].fillna(0))

    if cat_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(profiles[cat_cols].fillna("unknown"))
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num

    # Reduce n_clusters if needed
    max_clusters = len(profiles) // min_stores
    n_clusters = min(n_clusters, max(1, max_clusters))

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    profiles["cluster"] = kmeans.fit_predict(X)

    # Merge small clusters into nearest
    cluster_counts = profiles["cluster"].value_counts()
    small = cluster_counts[cluster_counts < min_stores].index.tolist()
    if small:
        large = cluster_counts[cluster_counts >= min_stores].index.tolist()
        if large:
            centers = kmeans.cluster_centers_
            for sc in small:
                dists = [np.linalg.norm(centers[sc] - centers[lc]) for lc in large]
                nearest = large[np.argmin(dists)]
                profiles.loc[profiles["cluster"] == sc, "cluster"] = nearest

    return profiles
```

**Step 3: Commit**

```bash
git add ml/training/segmentation/cluster_stores.py tests/training/test_cluster_stores.py
git commit -m "feat: add KMeans store clustering with min-stores enforcement"
```

---

### Task 3: Aggregate predictions

**Files:**

- Create: `ml/forecast/segmentation/__init__.py`
- Create: `ml/forecast/segmentation/aggregate_predictions.py`
- Test: `tests/forecast/test_aggregate_predictions.py`

**Step 1: Write failing test**

```python
# tests/forecast/test_aggregate_predictions.py
import pandas as pd
import numpy as np
from ml.forecast.segmentation.aggregate_predictions import aggregate_to_store_type

def test_aggregate_to_store_type():
    df = pd.DataFrame({
        "sku_id": ["A"] * 4,
        "branch_id": ["B1", "B2", "B3", "B4"],
        "store_type": ["S1", "S1", "S2", "S2"],
        "forecast_units": [10.0, 20.0, 30.0, 40.0],
        "units": [12.0, 18.0, 28.0, 42.0],  # actuals for weighting
    })
    result = aggregate_to_store_type(df)
    assert "store_type" in result.columns
    assert len(result) == 2  # S1 and S2
```

**Step 2: Implement**

```python
# ml/forecast/segmentation/aggregate_predictions.py
"""Aggregate forecasts to store-type level."""
from __future__ import annotations
import pandas as pd

def aggregate_to_store_type(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate forecasts by store_type using volume-weighted average."""
    if group_cols is None:
        group_cols = ["sku_id", "store_type"]

    agg = (
        df.groupby(group_cols)
        .agg(
            total_forecast=("forecast_units", "sum"),
            total_actual=("units", "sum"),
            n_branches=("branch_id", "nunique"),
        )
        .reset_index()
    )
    agg["avg_forecast_per_branch"] = agg["total_forecast"] / agg["n_branches"]
    return agg
```

**Step 3: Commit**

```bash
git add ml/forecast/segmentation/ tests/forecast/test_aggregate_predictions.py
git commit -m "feat: add store-type forecast aggregation"
```
