import json

import numpy as np
import pandas as pd
import pytest

from mibici.modeling import tracking
from mibici.modeling.regression import build_model, regression_metrics
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


def test_p3_silhouette_con_una_sola_fila_es_nan():
    X = np.array([[0.0, 1.0]])
    assert np.isnan(KMeansClusterTrainer._silhouette_o_nan(X, np.array([0])))


def test_perfil_ventana_vacio_da_ceros():
    """Una ventana sin viajes (p. ej. una estacion nueva) no debe tronar."""
    df_vacio = pd.DataFrame(columns=["origen_id", "hora"])
    X = KMeansClusterTrainer._perfil_ventana(df_vacio, None, 5)

    assert X.shape == (5, 24)
    assert (X == 0).all()


def test_perfil_ventana_normaliza_y_alinea_con_nomenclatura():
    # la estacion 1 viaja a las 8 (2 veces) y a las 9 (1); la 2 solo a las 8
    df = pd.DataFrame({"origen_id": [1, 1, 1, 2], "hora": [8, 8, 9, 8]})
    estaciones = pd.Series([1, 2, 3])  # la 3 esta en la nomenclatura sin viajes

    X = KMeansClusterTrainer._perfil_ventana(df, estaciones, 3)

    assert X.shape == (3, 24)
    np.testing.assert_allclose(X[0, 8], 2 / 3)  # proporcion de viajes, no conteos
    np.testing.assert_allclose(X[0, 9], 1 / 3)
    np.testing.assert_allclose(X[1, 8], 1.0)
    assert (X[2] == 0).all()  # estacion sin viajes: fila de ceros, no NaN
    # cada fila normalizada suma 1 (o 0 si no hay viajes)
    np.testing.assert_allclose(X.sum(axis=1), [1.0, 1.0, 0.0])


def test_perfil_ventana_sin_nomenclatura_solo_incluye_las_que_aparecen():
    df = pd.DataFrame({"origen_id": [7, 7, 9], "hora": [8, 8, 10]})
    X = KMeansClusterTrainer._perfil_ventana(df, None, 2)

    assert X.shape == (2, 24)  # solo las estaciones presentes, en su orden
    np.testing.assert_allclose(X[0, 8], 1.0)  # la 7: 2 de 2 viajes a las 8
    np.testing.assert_allclose(X[1, 10], 1.0)  # la 9: 1 de 1 viaje a las 10


def test_train_p3_misma_semilla_mismos_clusters(tmp_path, monkeypatch):
    """KMeans con SEED fija debe dar el mismo agrupamiento en dos corridas."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow_test.db")
    trainer = KMeansClusterTrainer(n_clusters=3)

    rng = np.random.default_rng(0)
    perfiles = np.vstack(
        [rng.normal(0, 0.05, (20, 24)), rng.normal(1, 0.05, (20, 24))]
    )

    _, etiquetas_1 = trainer.train_model(perfiles)
    _, etiquetas_2 = trainer.train_model(perfiles)

    np.testing.assert_array_equal(etiquetas_1, etiquetas_2)
    assert len(np.unique(etiquetas_1)) == 3
