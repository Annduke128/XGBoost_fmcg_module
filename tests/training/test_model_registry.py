"""Tests for model registry."""

import json
from pathlib import Path
from ml.training.models.model_registry import (
    save_model,
    load_latest,
    load_metadata,
    rotate_versions,
)


def test_save_model_creates_files(tmp_path):
    model = {"dummy": True}
    metadata = {"wape": 0.15, "params": {"max_depth": 6}}
    save_model(model, metadata, base_dir=tmp_path, version="v001")
    assert (tmp_path / "v001" / "model.pkl").exists()
    assert (tmp_path / "v001" / "metadata.json").exists()
    assert (tmp_path / "latest").is_symlink()


def test_save_model_updates_latest(tmp_path):
    save_model({"v": 1}, {}, base_dir=tmp_path, version="v001")
    save_model({"v": 2}, {}, base_dir=tmp_path, version="v002")
    loaded = load_latest(tmp_path)
    assert loaded["v"] == 2


def test_rotate_keeps_max_versions(tmp_path):
    for i in range(10):
        v = f"v{i:03d}"
        (tmp_path / v).mkdir()
        (tmp_path / v / "model.pkl").touch()
    rotate_versions(tmp_path, max_versions=8)
    versions = sorted(p for p in tmp_path.iterdir() if p.name.startswith("v"))
    assert len(versions) == 8
    # Oldest (v000, v001) should be gone
    assert not (tmp_path / "v000").exists()
    assert not (tmp_path / "v001").exists()


def test_load_latest(tmp_path):
    model = {"test": 42}
    save_model(model, {"metric": 0.1}, base_dir=tmp_path, version="v001")
    loaded = load_latest(tmp_path)
    assert loaded["test"] == 42


def test_load_metadata(tmp_path):
    save_model({"m": 1}, {"wape": 0.12}, base_dir=tmp_path, version="v001")
    meta = load_metadata(tmp_path)
    assert meta["wape"] == 0.12
