# AGENTS.md — XGBoost FMCG Demand Forecast Module

> This file is the single source of truth for AI agents working on this project.
> Read this FIRST before touching any code.

## Project Overview

Weekly SKU-branch demand forecasting for FMCG distributors using XGBoost Poisson regression. Predicts unit sales across ~10,000 SKU-branch combinations with 1/2/4 week horizons, generates replenishment orders with safety stock, and self-learns via Kalman filters.

**Target:** `units` (non-negative integer, weekly sales per SKU-branch)
**Granularity:** Weekly, per `(sku_id, branch_id)`
**Horizons:** 1, 2, 4 weeks
**Scenarios:** A (no promo) / B (50% discount)
**Service Level:** 95% (z=1.65)

## Tech Stack

| Component         | Technology                              |
| ----------------- | --------------------------------------- |
| Language          | Python 3.11+                            |
| ML Model          | XGBoost (`count:poisson`)               |
| Tuning            | Optuna (TimeSeriesSplit CV)             |
| Feature Filtering | Permutation Importance (WAPE scorer)    |
| Explainability    | SHAP (TreeExplainer) + PDP/ICE          |
| Replenishment     | Polars (reorder point + safety stock)   |
| Self-Learning     | 1D Kalman Filter (seasonal + promo)     |
| Encoding          | XGBoost native categorical (no one-hot) |
| Packaging         | Docker (python:3.11-slim)               |
| Testing           | pytest                                  |

## Quick Reference

```bash
# Activate venv
source /home/annduke/.venv/bin/activate

# Run tests
cd /home/annduke/project/XGBoost_fmcg_module
python -m pytest -v

# Run specific test
python -m pytest tests/training/test_train_xgb.py -v

# CLI
python -m ml.cli train --data data.parquet
python -m ml.cli forecast --data data.parquet

# Docker
docker build -t fmcg-forecast .
docker run -v $(pwd)/data:/app/data fmcg-forecast train --data /app/data/input.parquet
```

## Architecture

```
ml/
├── shared/                    # Shared across training & forecast (DO NOT DUPLICATE)
│   ├── schema.py              # REQUIRED_COLS validation
│   ├── utils/
│   │   └── metrics.py         # wape(), mae(), mape(), mdape()
│   └── features/
│       ├── feature_defs.py    # CAT_COLS (9) + NUM_COLS (47) = ALL_FEATURES (56)
│       └── encode.py          # make_categorical() for XGB native
│
├── training/                  # Training-only code
│   ├── data/
│   │   ├── split.py           # time_split() — train/val/test by time
│   │   ├── stockout.py        # impute_stockout() — censor & rolling median
│   │   ├── lead_time.py       # assign_lead_time() — rule: <=1→1wk, <=2→2wk, else 4wk
│   │   ├── scenario.py        # build_scenarios() — A: no promo, B: 50% discount
│   │   ├── holiday_calendar.py# VN holidays (CSV + built-in Tet 2023-2026)
│   │   ├── daily_aggregate.py # Daily lag/rolling/EMA → weekly _last values
│   │   └── promo_depth.py     # compute_promo_depth() — unified discount rate
│   ├── features/
│   │   ├── build_features.py  # add_time_features(), add_lag_features(), build_all_features()
│   │   ├── daily_features.py  # Daily time features → weekly aggregation
│   │   └── growth_features.py # qty/category/subcat growth + KBinsDiscretizer
│   ├── factors/
│   │   ├── kalman_filter.py   # KalmanState, KalmanConfig, KalmanFactorStore
│   │   ├── seasonal_kalman.py # Seasonal factor per week-of-year
│   │   └── promo_kalman.py    # Promo factor per promo_type
│   ├── models/
│   │   ├── baselines.py       # seasonal_naive(), poisson_baseline()
│   │   ├── train_xgb.py       # train_xgb_poisson() — count:poisson + enable_categorical
│   │   ├── tune_optuna.py     # run_optuna_study() — 8 hyperparams, WAPE scoring
│   │   └── model_registry.py  # save_model(), load_latest(), rotate_versions(max=8)
│   ├── pipelines/
│   │   └── train_weekly.py    # TrainWeeklyConfig + run_train_weekly()
│   └── segmentation/
│       ├── store_profiles.py  # build_store_profiles()
│       └── cluster_stores.py  # cluster_stores() — KMeans + small cluster merge
│
├── forecast/                  # Forecast & post-processing
│   ├── pipelines/
│   │   └── forecast_weekly.py # ForecastConfig + run_forecast()
│   ├── explain/
│   │   ├── permutation.py     # run_permutation(), select_top_features()
│   │   ├── shap_report.py     # shap_summary(), shap_local()
│   │   ├── pdp_report.py      # pdp_plot() — PDP + ICE
│   │   └── promo_impact.py    # promo_type_impact(), promo_depth_curve(), promo_impact_summary()
│   ├── replenishment/
│   │   ├── reorder_polars.py  # compute_reorder() — ROP + order_qty via Polars
│   │   └── self_learning.py   # EMA seasonal/branch/sigma (legacy, see Kalman)
│   └── segmentation/
│       └── aggregate_predictions.py  # aggregate_to_store_type()
│
└── cli.py                     # CLI entrypoint: train | forecast

tests/                         # Mirrors ml/ structure
├── shared/                    # 4 test files
├── training/                  # 14 test files (+1 promo_depth)
├── forecast/                  # 6 test files (+2 forecast_weekly, promo_impact)
└── test_cli.py                # CLI smoke tests

Dockerfile                     # python:3.11-slim
requirements.txt               # All dependencies with version ranges
docs/plans/                    # 9 implementation plan files (reference only)
```

## Data Contract

### Required Input Columns (`ml/shared/schema.py`)

| Column          | Type      | Description                                                              |
| --------------- | --------- | ------------------------------------------------------------------------ |
| `week`          | datetime  | Week start date                                                          |
| `sku_id`        | str/cat   | Product identifier (~10k)                                                |
| `branch_id`     | str/cat   | Branch/store identifier (~10k)                                           |
| `units`         | int/float | Target: weekly sales units                                               |
| `price`         | float     | Unit price                                                               |
| `promo_flag`    | int       | 0=no promo, 1=promo active                                               |
| `brand_type`    | str/cat   | Brand classification                                                     |
| `branch_type`   | str/cat   | Branch classification                                                    |
| `stockout_flag` | int       | 0=in-stock, 1=stockout                                                   |
| `display_units` | int/float | Shelf/display capacity                                                   |
| `promo_depth`   | float     | Discount depth (0-1), auto-computed by `compute_promo_depth()` if absent |

### Optional Columns (for extended features)

| Column                  | Required By                     | Description                                                     |
| ----------------------- | ------------------------------- | --------------------------------------------------------------- |
| `date`                  | daily_features, daily_aggregate | Daily-level date                                                |
| `category`              | growth_features                 | Product category                                                |
| `sub_category`          | growth_features                 | Product sub-category                                            |
| `promo_type`            | promo_kalman, promo_depth       | One of: discount_bundle, direct_discount, buy_x_get_y, buy_gift |
| `promo_discount`        | compute_promo_depth             | Raw discount rate for direct_discount/discount_bundle           |
| `promo_x_qty`           | compute_promo_depth             | Buy X quantity (buy_x_get_y, buy_gift)                          |
| `promo_y_qty`           | compute_promo_depth             | Get Y quantity (buy_x_get_y)                                    |
| `gift_value`            | compute_promo_depth             | Gift monetary value (buy_gift)                                  |
| `channel`               | segmentation                    | Sales channel                                                   |
| `store_type`            | segmentation                    | Store type classification                                       |
| `display_capacity_type` | segmentation                    | Display capacity tier                                           |
| `service_scale`         | segmentation                    | Service level tier                                              |
| `temperature`           | (future)                        | Daily temperature                                               |
| `rainfall`              | (future)                        | Daily rainfall                                                  |

## Feature Groups (56 total)

| Group       | Count | Examples                                                           | Source Module                           |
| ----------- | ----- | ------------------------------------------------------------------ | --------------------------------------- |
| Categorical | 9     | sku_id, branch_id, brand_type, promo_type_major                    | `feature_defs.py`                       |
| Core        | 9     | price, promo_flag, sin_woy, cos_woy, lag_1/2/4, roll_4_mean, ema_4 | `build_features.py`                     |
| Daily Time  | 7     | weekend_ratio, holiday_flag, month_start/end_flag                  | `daily_features.py`                     |
| Daily Lag   | 14    | lag_1d_last...lag_28d_last, roll_3d_mean_last...                   | `daily_aggregate.py`                    |
| Growth      | 8     | qty_growth_1w/4w, category/subcat_growth_1w/4w                     | `growth_features.py`                    |
| Promo       | 2     | promo_ratio, promo_depth_avg                                       | `daily_features.py`                     |
| Weather     | 4     | rainy_ratio, temp_avg/max/min                                      | (future POS integration)                |
| Kalman      | 2     | seasonal_factor, promo_factor                                      | `seasonal_kalman.py`, `promo_kalman.py` |

## Key Design Decisions

### 1. No Leakage

All lag/rolling/EMA features use `shift(1)+` — never access current or future data. Enforced in `build_features.py` and `daily_aggregate.py`.

### 2. XGBoost `count:poisson` (not `reg:poisson`)

XGBoost 3.2.0+ renamed the Poisson objective. The code uses `count:poisson`. If you see `reg:poisson` errors, this is the cause.

### 3. Native Categorical Encoding

With ~10k cardinality for sku_id/branch_id, one-hot encoding would explode dimensionality. Instead, columns are cast to `pandas.Categorical` and XGBoost handles them natively via `enable_categorical=True, tree_method="hist"`.

### 4. Stockout Handling: Censor & Impute

When `stockout_flag=1`, the recorded `units` is not true demand. `impute_stockout()` replaces it with rolling median of prior non-stockout weeks (group-level fallback).

### 5. Lead-Time Rule

```
weeks_of_cover = display_units / ema_sales_8w
if weeks_of_cover <= 1.0  → lead_time = 1 week
if weeks_of_cover <= 2.0  → lead_time = 2 weeks
else                      → lead_time = 4 weeks
```

### 6. Kalman Self-Learning (replaces EMA)

Seasonal and promo adjustment factors use 1D Kalman filters:

- `seasonal_factor[week_of_year]`: adjusts for seasonal bias
- `promo_factor[promo_type]`: adjusts for promo lift bias
- Config: Q=0.01, R=0.05, initial_x=1.0, clamp [0.5, 2.0]
- State persisted as JSON in `factors_dir/`
- Applied: `pred_final = pred_base * seasonal_factor * promo_factor`

### 7. Reorder Point Formula

```
ROP = demand_during_LT + z * sigma * sqrt(lead_time_weeks)
order_qty = max(0, ROP - on_hand)
z = 1.65 (95% service level)
```

### 8. Model Registry

- Artifacts saved as `.pkl` + `metadata.json` in `artifacts/models/<version>/`
- Symlink `latest/` → most recent version
- Auto-rotates, keeps max 8 versions

### 9. Recursive Multi-step Forecast (Rollout)

Multi-horizon forecasts (h=1,2,4) use **recursive rollout** — each step's prediction feeds as `lag_1` into the next:

```
h=1: lag_1 = actual[T]        → predict(h=1)
h=2: lag_1 = predict(h=1)     → predict(h=2)
h=3: lag_1 = predict(h=2)     → predict(h=3)  ← internal only
h=4: lag_1 = predict(h=3)     → predict(h=4)
```

- `_recompute_lags()` rebuilds lag_1/2/4, roll_4_mean, ema_4 per group after each step
- Intermediate steps (h=3) computed for correct lag propagation but excluded from output
- Predictions clipped >= 0 (Poisson target)
- See `ml/forecast/pipelines/forecast_weekly.py:_rollout_forecast()`

### 10. Promo Depth Auto-Computation

`compute_promo_depth()` converts heterogeneous promo types to a unified `promo_depth` (0–1):

| promo_type      | Formula                        |
| --------------- | ------------------------------ |
| direct_discount | `promo_discount` as-is         |
| discount_bundle | `promo_discount` as-is         |
| buy_x_get_y     | `y / (x + y)`                  |
| buy_gift        | `gift_value / (price * x_qty)` |
| no_promo / NaN  | 0.0                            |

- Called **before** `validate_columns()` in both pipelines
- Skips if `promo_depth` already present
- Raw promo columns are optional (default to safe values)
- Output clipped [0, 1]
- See `ml/training/data/promo_depth.py`

### 11. Promo Impact Analysis

Post-forecast analysis module with three functions:

- **`promo_type_impact()`**: uplift % per promo_type vs matched no-promo baseline per (sku_id, branch_id)
- **`promo_depth_curve()`**: bins promo_depth 0–1 into equal-width buckets, computes uplift per bin
- **`promo_impact_summary()`**: cross-tab of uplift by arbitrary group columns (default: promo_type × brand_type)

All functions return empty DataFrame with correct schema on edge cases (missing columns, no promo rows).
See `ml/forecast/explain/promo_impact.py`

## Pipeline Flow

### Training (`run_train_weekly`)

```
Load parquet → compute_promo_depth()
→ validate_columns()
→ impute_stockout()
→ assign_lead_time()
→ build_all_features()
→ [opt-in] add_daily_features() + aggregate
→ [opt-in] build_growth_features()
→ [opt-in] apply_kalman_factors()
→ time_split(val_weeks=4, test_weeks=4)
→ make_categorical()
→ baseline (seasonal_naive + PoissonRegressor)
→ [optional] run_optuna_study()
→ train_xgb_poisson()
→ save_model() + rotate_versions()
→ [opt-in] update_kalman_factors()
→ return metrics dict
```

### Forecast (`run_forecast`)

```
Load parquet + load_latest model
→ compute_promo_depth()
→ validate_columns()
→ build features (same pipeline as training)
→ for each scenario (A, B):
    → _rollout_forecast(horizons=[1,2,4]):
        h=1: predict from actual lags
        h=2: lag_1 = pred(h=1) → predict
        h=3: lag_1 = pred(h=2) → predict (internal only)
        h=4: lag_1 = pred(h=3) → predict
    → collect requested horizons only
→ concat results + save to output_path
```

### Replenishment

```
forecast output → compute_reorder(z=1.65)
→ ROP, demand_lt, sigma, order_qty per (sku_id, branch_id)
```

## Rules for Agents

### DO

- Run `python -m pytest -v` after any change
- Use `ml/shared/` for anything shared between training and forecast
- Add new features to `ml/shared/features/feature_defs.py` (canonical list)
- Keep all lag/rolling/EMA shifted by at least 1 period
- Use `enable_categorical=True` for XGBoost — never one-hot encode high-cardinality features
- Test with small synthetic DataFrames in tests
- Use `count:poisson` objective (not `reg:poisson`)

### DON'T

- Don't modify `ml/shared/` without updating both training and forecast pipelines
- Don't add features to `build_features.py` without adding to `feature_defs.py`
- Don't use future data in features (leakage)
- Don't one-hot encode sku_id or branch_id (10k+ categories)
- Don't use `reg:poisson` (deprecated in XGBoost 3.2.0+)
- Don't commit secrets, API keys, or `.env` files
- Don't modify test data files to make tests pass

### CONVENTIONS

- Tests mirror `ml/` structure under `tests/`
- Private helpers prefixed with `_` (e.g., `_neg_wape`, `_add_daily_features`)
- Config via dataclasses (`TrainWeeklyConfig`, `ForecastConfig`)
- Extended features are **opt-in** via config flags (`use_daily_features`, `use_growth_features`, `use_kalman_factors`)
- Metrics always use `EPS = 1e-6` to avoid division by zero

## Dependency Graph

```
ml/shared/schema          ← ml/training/pipelines/train_weekly
ml/shared/features/*      ← ml/training/pipelines/train_weekly
                          ← ml/forecast/pipelines/forecast_weekly

ml/training/data/*        ← ml/training/pipelines/train_weekly
                          ← ml/forecast/pipelines/forecast_weekly
ml/training/features/*    ← ml/training/pipelines/train_weekly
ml/training/factors/*     ← ml/training/pipelines/train_weekly
                          ← ml/forecast/pipelines/forecast_weekly
ml/training/models/*      ← ml/training/pipelines/train_weekly

ml/forecast/explain/*     ← (standalone, called after forecast)
ml/forecast/replenishment/*← (standalone, called after forecast)
ml/forecast/segmentation/* ← (standalone, called after forecast)
```

## Current Status

- **131/131 tests passing**
- All 9 implementation beads closed
- Feature extension (daily/holiday/growth/Kalman) complete
- Recursive multi-step forecast (rollout h=1→4) complete
- Promo depth auto-computation + promo impact analysis complete
- Docker packaging complete
- **Future:** POS integration for real-time stockout data, weather API integration

## Dependencies (requirements.txt)

```
pandas>=2.0,<3.0
numpy>=1.24,<2.0
scikit-learn>=1.3,<2.0
xgboost>=2.0,<4.0
optuna>=3.0,<5.0
shap>=0.42,<1.0
polars>=0.20,<2.0
matplotlib>=3.7,<4.0
joblib>=1.3,<2.0
pyarrow>=14.0,<18.0
```
