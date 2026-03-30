"""Tests for weekly training pipeline."""

from ml.training.pipelines.train_weekly import TrainWeeklyConfig


def test_config_defaults():
    cfg = TrainWeeklyConfig()
    assert cfg.val_weeks == 4
    assert cfg.test_weeks == 4
    assert cfg.max_model_versions == 8
    assert cfg.n_optuna_trials == 50
    assert cfg.run_tuning is True


def test_config_custom():
    cfg = TrainWeeklyConfig(
        data_path="custom/path.parquet",
        model_dir="custom/models",
        val_weeks=2,
        test_weeks=2,
        max_model_versions=5,
        n_optuna_trials=10,
        run_tuning=False,
    )
    assert cfg.data_path == "custom/path.parquet"
    assert cfg.val_weeks == 2
    assert cfg.run_tuning is False
