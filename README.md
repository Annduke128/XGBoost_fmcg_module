# FMCG Demand Forecast & Replenishment

XGBoost Poisson regression model for weekly SKU-branch sales forecasting with automated replenishment planning.

## Features

- **50+ engineered features**: time-based, seasonal (Fourier), lag, rolling stats, EMA, price, promo, weather
- **XGBoost Poisson** with Optuna hyperparameter tuning
- **Feature selection**: Permutation Importance (WAPE-based)
- **Explainability**: SHAP (global + local) + PDP/ICE plots
- **Replenishment**: Polars-based reorder point calculation (95% service level)
- **Self-learning**: EMA-updated seasonal index, branch adjustment, safety stock sigma
- **Store segmentation**: KMeans clustering with volume-weighted aggregation
- **Multi-horizon**: 1/2/4 week forecasts with promo scenarios (0% / 50% discount)

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
├── training/        # Data prep, feature build, models, pipelines, segmentation
└── forecast/        # Prediction, explainability, replenishment, self-learning
```

## Key Decisions

- **Objective**: `count:poisson` (XGBoost 3.2+) for non-negative unit predictions
- **Encoding**: XGBoost native categorical (handles ~10k SKU/branch cardinality)
- **Lead-time rule**: weeks_of_cover = display / EMA_sales → 1/2/4 weeks
- **Safety stock**: z=1.65 (95% service level), sigma from rolling residual std
- **No leakage**: all lag/rolling features use shift(1)+ before aggregation
