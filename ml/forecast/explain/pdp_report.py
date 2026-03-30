"""Partial dependence plot utilities."""

from __future__ import annotations

from pathlib import Path

from sklearn.inspection import PartialDependenceDisplay


def pdp_plot(model: object, X, features, output_dir: str | Path) -> None:
    """Generate PDP + ICE plots and save as PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features,
        kind="both",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "pdp_ice.png", dpi=150)
    plt.close()
