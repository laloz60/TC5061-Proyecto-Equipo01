import json

import numpy as np
import pandas as pd
import pytest

from mibici.modeling import tracking
from mibici.modeling.regression import build_model, regression_metrics, rule_metrics
from mibici.modeling.train import KMeansClusterTrainer


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


def test_regla_baseline_usa_promedio_de_4_semanas():
    val = pd.DataFrame({"salidas": [1.0, 3.0], "salidas_prom_4sem": [1.0, 3.0]})
    test = pd.DataFrame({"salidas": [2.0, 4.0], "salidas_prom_4sem": [3.0, 5.0]})

    metricas = rule_metrics(val, test)

    assert metricas["val_mae"] == 0
    assert metricas["test_mae"] == 1
    assert set(metricas) == {"val_mae", "val_rmse", "val_r2", "test_mae", "test_rmse", "test_r2"}


def test_seleccion_de_modelos_usa_validacion():
    """La seleccion en el Registry no puede usar metricas de test (comentario del profe)."""
    from mibici.modeling.register import _MODELOS

    assert _MODELOS, "faltan preguntas en el registro"
    for pregunta, (_, metrica, _) in _MODELOS.items():
        assert metrica.startswith("val_"), (
            f"{pregunta} se selecciona con {metrica!r}; debe ser una metrica de validacion"
        )


def test_p3_silhouette_se_calcula_en_ventanas_separadas():
    """P3: perfiles por ventana y silhouette protegido si la ventana no alcanza."""
    trainer_metrics = KMeansClusterTrainer._silhouette_o_nan

    # ventana con una sola etiqueta: no hay silhouette posible -> NaN, no truena
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    assert np.isnan(trainer_metrics(X, np.zeros(3)))

    # caso normal: 2 etiquetas -> silhouette computable
    valor = trainer_metrics(X, np.array([0, 0, 1]))
    assert -1.0 <= valor <= 1.0
