"""Tests de trazabilidad MLflow (#22) y utilidades de P1 (sin MLflow ni datos reales)."""

import json

import numpy as np
import pytest

from mibici.config import DATA_END, DATA_START
from mibici.modeling import tracking
from mibici.modeling.regression import build_model, regression_metrics


def test_split_params_come_from_config():
    params = tracking.split_params()
    assert params["train_start"] == DATA_START.isoformat() == "2023-05-01"
    assert params["val_start"] == "2025-01-01"
    assert params["test_start"] == "2025-07-01"
    assert params["test_end"] == DATA_END.isoformat() == "2026-09-30"


def test_data_params_read_parquet_hash_from_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"unified_parquet": {"sha256": "a" * 64}}))
    monkeypatch.setattr(tracking, "MANIFEST_PATH", manifest)
    params = tracking.data_params()
    assert params["data_parquet_sha256"] == "a" * 64
    assert len(params["data_manifest_sha256"]) == 64


def test_run_tags_include_dynamic_commit():
    tags = tracking.run_tags("P1")
    assert tags["equipo"] == "01"
    assert tags["pregunta"] == "P1"
    assert tags["commit"] != "10dfa34"


def test_regression_metrics_perfect_prediction():
    y = np.array([0.0, 1.0, 3.0])
    assert regression_metrics(y, y) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_build_model_rejects_unknown_kind():
    with pytest.raises(ValueError, match="no soportado"):
        build_model("xgboost")
