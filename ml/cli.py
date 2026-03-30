"""CLI entrypoint for Docker container."""

from __future__ import annotations
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ml.cli [train|forecast]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        from ml.training.pipelines.train_weekly import (
            run_train_weekly,
            TrainWeeklyConfig,
        )

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
