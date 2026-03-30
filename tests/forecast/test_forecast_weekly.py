from ml.forecast.pipelines.forecast_weekly import ForecastConfig


def test_forecast_config_defaults():
    cfg = ForecastConfig()
    assert cfg.horizons == [1, 2, 4]
    assert cfg.scenarios == ["A", "B"]
