"""Reproducibilidad del entrenamiento (issue #96).

Dos corridas con la misma semilla deben dar las mismas predicciones y
métricas; con semilla distinta deben cambiar (si no, la prueba no serviría).

Usa una muestra de los datos reales y modelos chicos para que el suite
complete en segundos.
"""

import numpy as np
import pyarrow.parquet as pq
import pytest

from mibici.config import PROCESSED_DATA_DIR, SEED
from mibici.modeling import regression
from mibici.modeling.regression import FEATURES, TARGET, build_model, regression_metrics

FEATURES_PATH = PROCESSED_DATA_DIR / "features_estacion_hora.parquet"

# columnas mínimas: features + target + split
COLUMNAS = [TARGET, "split"] + FEATURES
N_FILAS = 40_000  # muestra fija y chica: el fit completo son 4.7M de filas
N_ARBOLES = 5


@pytest.fixture(scope="module")
def muestra_train():
    """Muestra fija de train; salta si no se corrió `python -m mibici.features`."""
    if not FEATURES_PATH.exists():
        pytest.skip("features no generadas todavía")

    # solo el primer row group: se lee en milisegundos
    tabla = pq.ParquetFile(FEATURES_PATH).read_row_group(0, columns=COLUMNAS)
    df = tabla.to_pandas()
    df = df[df["split"] == "train"].head(N_FILAS)
    df = df.dropna(subset=FEATURES)
    if len(df) < 1_000:
        pytest.skip("muestra de train demasiado chica")
    return df.reset_index(drop=True)


def _entrenar(df):
    """Entrena RF (usando SEED de config) y devuelve predicciones + métricas."""
    modelo = build_model("rf", n_estimators=N_ARBOLES)
    modelo.fit(df[FEATURES], df[TARGET])
    predicho = modelo.predict(df[FEATURES])
    metricas = regression_metrics(df[TARGET].to_numpy(), predicho)
    return predicho, metricas


def test_misma_semilla_repite_predicciones_y_metricas(muestra_train):
    predicho_1, metricas_1 = _entrenar(muestra_train)
    predicho_2, metricas_2 = _entrenar(muestra_train)

    # predicciones idénticas: RF corre con n_jobs=-1 y el orden de la suma
    # en paralelo puede variar en el último bit (1e-15), así que se exige
    # igualdad hasta 1e-9; una semilla distinta cambia valores enteros
    np.testing.assert_allclose(predicho_1, predicho_2, rtol=0, atol=1e-9)

    # métricas idénticas (mismo orden de magnitud de tolerancia)
    assert metricas_1.keys() == metricas_2.keys()
    for nombre in metricas_1:
        assert abs(metricas_1[nombre] - metricas_2[nombre]) < 1e-9, (
            f"{nombre} no se repitió: {metricas_1[nombre]} vs {metricas_2[nombre]}"
        )


def test_semilla_distinta_cambia_el_entrenamiento(muestra_train, monkeypatch):
    """Si alguien cambia SEED, las predicciones deben cambiar: si no, la
    prueba anterior no estaría verificando nada."""
    predicho_con_42, _ = _entrenar(muestra_train)

    monkeypatch.setattr(regression, "SEED", SEED + 1)
    predicho_otra, _ = _entrenar(muestra_train)

    # allclose y no array_equal: con n_jobs=-1 hasta la MISMA semilla difiere
    # en el ultimo bit (1e-15), asi que array_equal daria False aunque la
    # semilla no tuviera efecto (revisado por @smuciton en el PR #114)
    assert not np.allclose(predicho_con_42, predicho_otra, rtol=0, atol=1e-9), (
        "cambiar la semilla no alteró las predicciones: la prueba de "
        "reproducibilidad no estaría midiendo nada"
    )


def test_ridge_tambien_es_reproducible(muestra_train):
    """El candidato Ridge de P1 no depende de azar, pero debe ser estable."""
    modelo_1 = build_model("ridge")
    modelo_1.fit(muestra_train[FEATURES], muestra_train[TARGET])
    modelo_2 = build_model("ridge")
    modelo_2.fit(muestra_train[FEATURES], muestra_train[TARGET])

    np.testing.assert_array_equal(
        modelo_1.predict(muestra_train[FEATURES]),
        modelo_2.predict(muestra_train[FEATURES]),
    )
    assert regression_metrics(
        muestra_train[TARGET].to_numpy(), modelo_1.predict(muestra_train[FEATURES])
    ) == regression_metrics(
        muestra_train[TARGET].to_numpy(), modelo_2.predict(muestra_train[FEATURES])
    )
