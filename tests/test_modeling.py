import json

import numpy as np
import pytest

from mibici.modeling import tracking
from mibici.modeling.regression import build_model, regression_metrics


def test_fechas_de_config():
    params = tracking.split_params()
    assert params["train_start"] == "2023-05-01"
    assert params["val_start"] == "2025-01-01"
    assert params["test_start"] == "2025-07-01"
    assert params["test_end"] == "2026-09-30"


def test_hash_del_parquet(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"unified_parquet": {"sha256": "abc123"}}))
    monkeypatch.setattr(tracking, "MANIFEST_PATH", manifest)

    params = tracking.data_params()
    assert params["data_parquet_sha256"] == "abc123"


def test_tags():
    tags = tracking.run_tags("P1")
    assert tags["equipo"] == "01"
    assert tags["pregunta"] == "P1"
    assert tags["commit"] != "desconocido"


def test_metricas_prediccion_perfecta():
    y = np.array([0.0, 1.0, 3.0])
    metricas = regression_metrics(y, y)
    assert metricas["mae"] == 0
    assert metricas["rmse"] == 0
    assert metricas["r2"] == 1


def test_modelo_no_soportado():
    with pytest.raises(ValueError):
        build_model("xgboost")
