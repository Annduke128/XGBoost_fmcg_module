"""Model artifact registry: save, load, rotate .pkl models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib


def save_model(
    model: object,
    metadata: dict,
    base_dir: Path | str,
    version: str,
) -> Path:
    """Save model as .pkl with metadata JSON.

    Args:
        model: Trained model object.
        metadata: Dict with metrics, params, feature_cols etc.
        base_dir: Root directory for model artifacts.
        version: Version string (e.g. v20260330_120000).

    Returns:
        Path to the version directory.
    """
    base_dir = Path(base_dir)
    version_dir = base_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, version_dir / "model.pkl")
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Update latest symlink
    latest = base_dir / "latest"
    if latest.is_symlink():
        latest.unlink()
    elif latest.exists():
        shutil.rmtree(latest)
    latest.symlink_to(version_dir.resolve())
    return version_dir


def load_latest(base_dir: Path | str) -> object:
    """Load model from latest version."""
    base_dir = Path(base_dir)
    latest = base_dir / "latest"
    if not latest.exists():
        raise FileNotFoundError(f"No latest model in {base_dir}")
    return joblib.load(latest / "model.pkl")


def load_metadata(base_dir: Path | str) -> dict:
    """Load metadata from latest version."""
    base_dir = Path(base_dir)
    latest = base_dir / "latest"
    if not latest.exists():
        raise FileNotFoundError(f"No latest model in {base_dir}")
    return json.loads((latest / "metadata.json").read_text())


def rotate_versions(base_dir: Path | str, max_versions: int = 8) -> None:
    """Keep only the most recent `max_versions` versions."""
    base_dir = Path(base_dir)
    versions = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("v")],
        key=lambda p: p.name,
    )
    while len(versions) > max_versions:
        oldest = versions.pop(0)
        shutil.rmtree(oldest)
