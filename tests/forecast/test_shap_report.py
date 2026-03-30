def test_shap_report_imports():
    from ml.forecast.explain import shap_report

    assert shap_report is not None
