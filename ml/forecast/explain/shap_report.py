"""SHAP reporting utilities."""

from __future__ import annotations

from pathlib import Path


def shap_summary(
    model: object,
    X_sample,
    output_dir: str | Path,
    max_display: int = 20,
):
    """Generate SHAP summary plot and save as PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150)
    plt.close()
    return shap_values


def shap_local(model: object, X_row, output_dir: str | Path):
    """Generate SHAP force plot for a single row and save as PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    shap.force_plot(
        explainer.expected_value,
        shap_values,
        X_row,
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_local.png", dpi=150)
    plt.close()
    return shap_values
