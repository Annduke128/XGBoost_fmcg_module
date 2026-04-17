# FMCG Demand Forecast & Replenishment

XGBoost Poisson regression model for weekly SKU-branch sales forecasting with automated replenishment planning.

## Features

- **56 engineered features**: time-based, seasonal (Fourier), lag, rolling stats, EMA, price, promo depth, daily, growth, Kalman factors
- **XGBoost Poisson** (`count:poisson`) with Optuna hyperparameter tuning
- **Recursive multi-step forecast**: h=1→2→3→4 rollout where each prediction feeds as lag into the next step
- **Promo depth auto-computation**: converts all promo types (direct discount, buy-X-get-Y, bundle, gift) to unified 0–1 discount rate
- **Promo impact analysis**: uplift by promo type, depth response curve, cross-tab summary
- **Feature selection**: Permutation Importance (WAPE-based)
- **Explainability**: SHAP (global + local) + PDP/ICE plots
- **Replenishment**: Polars-based reorder point calculation (95% service level)
- **Self-learning**: Kalman filter seasonal + promo adjustment factors (replaces legacy EMA)
- **Store segmentation**: KMeans clustering with volume-weighted aggregation
- **Multi-horizon**: 1/2/4 week forecasts with promo scenarios (A = no promo / B = 50% discount)

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python -m ml.cli train --data data.parquet --output artifacts/

# Forecast
python -m ml.cli forecast --data data.parquet --model artifacts/models/latest/ --output output/
```

## Docker

```bash
docker build -t fmcg-forecast .
docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts fmcg-forecast train --data data/input.parquet
```

## Architecture

```
ml/
├── shared/          # Schema, metrics, feature definitions, encoding
├── training/        # Data prep (stockout, lead-time, promo depth), feature build,
│                    #   models, Kalman factors, pipelines, segmentation
└── forecast/        # Recursive rollout prediction, explainability (SHAP, PDP,
                     #   promo impact), replenishment, self-learning
```

## Key Decisions

- **Objective**: `count:poisson` (XGBoost 3.2+) for non-negative unit predictions
- **Encoding**: XGBoost native categorical (handles ~10k SKU/branch cardinality)
- **Recursive rollout**: multi-step forecast (h=1→4) where each prediction feeds as lag_1 to the next step; h=3 computed internally for correct lag propagation but excluded from output
- **Promo depth**: all promo types (direct, bundle, buy-X-get-Y, gift) auto-converted to 0–1 discount rate before validation
- **Lead-time rule**: weeks_of_cover = display / EMA_sales → 1/2/4 weeks
- **Safety stock**: z=1.65 (95% service level), sigma from rolling residual std
- **No leakage**: all lag/rolling features use shift(1)+ before aggregation
- **Self-learning**: 1D Kalman filters for seasonal + promo bias correction (replaces legacy EMA)
