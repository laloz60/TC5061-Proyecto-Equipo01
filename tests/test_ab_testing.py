"""Pruebas del A/B testing de P2 (issue #138).

Cubre la logica pura (metricas, umbral, McNemar, tabla y reporte) y un
escenario integral en miniatura: entrena la V2, la registra y la compara
contra una V1 ya puesta en el alias `champion`, sobre un parquet sintetico
y con un backend MLflow propio en sqlite (no toca el mlflow.db del proyecto).
"""

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import SEED
from mibici.features import add_risk_label, assign_split
from mibici.modeling import ab_testing
from mibici.modeling.ab_testing import (
    _conclusion_sensibilidad,
    _texto_pvalue,
    lectura_de_la_diferencia,
    mcnemar_exacto,
    metricas_clasificacion,
    renderizar_reporte,
    seleccionar_umbral,
    tabla_comparativa,
    tabla_sensibilidad,
    veredicto,
)
from mibici.modeling.regression import FEATURES
from mibici.modeling.tracking import EXPERIMENT_NAME

# --------------------------------------------------------------------------
# metricas y umbral
# --------------------------------------------------------------------------


def test_metricas_con_prediccion_perfecta():
    y = np.array([0, 0, 1, 1])
    metricas = metricas_clasificacion(y, np.array([0.1, 0.2, 0.8, 0.9]), umbral=0.5)
    assert metricas["roc_auc"] == 1.0
    assert metricas["precision"] == 1.0
    assert metricas["recall"] == 1.0
    assert metricas["f1_macro"] == 1.0


def test_seleccionar_umbral_elige_el_que_maximiza_f1():
    # negativos en 0.4 y positivos en 0.6: 0.3 falla por los falsos positivos,
    # 0.7 falla por perder todos los positivos, 0.5 es el unico perfecto
    y = np.array([0] * 10 + [1] * 10)
    prob = np.array([0.4] * 10 + [0.6] * 10)
    umbral, f1 = seleccionar_umbral(y, prob, grid=[0.3, 0.5, 0.7])
    assert umbral == 0.5
    assert f1 == pytest.approx(1.0)


def test_seleccionar_umbral_es_reproducible_con_la_misma_semilla_de_datos():
    y = np.array([0, 1, 0, 1])
    prob = np.array([0.2, 0.8, 0.4, 0.6])
    assert seleccionar_umbral(y, prob) == seleccionar_umbral(y, prob)


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------


def test_mcnemar_cuenta_los_pares_discordantes():
    y = np.array([1] * 10 + [0] * 2 + [1] * 5)
    pred_v1 = np.array([0] * 10 + [0] * 2 + [1] * 5)  # falla los 10 primeros
    pred_v2 = np.array([1] * 10 + [1] * 2 + [1] * 5)  # falla los 2 del medio

    resultado = mcnemar_exacto(y, pred_v1, pred_v2)
    assert resultado["solo_v1"] == 2
    assert resultado["solo_v2"] == 10
    assert resultado["discordantes"] == 12
    assert resultado["pvalue"] < 0.05  # la V2 gana con significancia


def test_mcnemar_sin_discordantes_da_p_uno():
    y = np.array([0, 1, 0, 1])
    resultado = mcnemar_exacto(y, y.copy(), y.copy())
    assert resultado["discordantes"] == 0
    assert resultado["pvalue"] == 1.0


# --------------------------------------------------------------------------
# tabla y veredicto
# --------------------------------------------------------------------------


def _metricas(f1=0.60, auc=0.70):
    return {
        "roc_auc": auc,
        "average_precision": 0.50,
        "precision": 0.60,
        "recall": 0.60,
        "f1_macro": f1,
    }


def _sensibilidad(delta_igual_umbral=0.0):
    """Tabla de sensibilidad sintetica: V2 = V1 a menos por el desfase indicado."""
    base = {
        "roc_auc": 0.70,
        "average_precision": 0.50,
        "precision": 0.60,
        "recall": 0.60,
    }
    filas = [
        ("V1 @ su umbral", 0.5, 0.5129),
        ("V2 @ su umbral", 0.7, 0.5638),
        ("V1 @ umbral de la V2", 0.7, 0.5640),
        ("V2 @ umbral de la V1", 0.5, 0.5129 + delta_igual_umbral),
    ]
    return pd.DataFrame(
        [
            {"configuracion": config, "umbral": umbral, **base, "f1_macro": f1}
            for config, umbral, f1 in filas
        ]
    )


def test_tabla_comparativa_calcula_el_delta_v2_menos_v1():
    tabla = tabla_comparativa(_metricas(), _metricas(f1=0.65, auc=0.72))
    assert list(tabla.columns) == ["metrica", "v1", "v2", "delta"]
    fila_f1 = tabla.loc[tabla["metrica"] == "f1_macro"].iloc[0]
    assert fila_f1["delta"] == pytest.approx(0.05)
    fila_auc = tabla.loc[tabla["metrica"] == "roc_auc"].iloc[0]
    assert fila_auc["delta"] == pytest.approx(0.02)


def test_veredicto_sin_significancia_no_recomienda_mover_champion():
    tabla = tabla_comparativa(_metricas(), _metricas(f1=0.90, auc=0.95))
    texto = veredicto(
        tabla,
        {"pvalue": 0.4, "solo_v1": 5, "solo_v2": 7, "discordantes": 12},
        _sensibilidad(delta_igual_umbral=0.05),
    )
    assert "Sin diferencia significativa" in texto
    assert "champion" in texto


def test_veredicto_recomienda_la_v2_si_tambien_gana_a_igual_umbral():
    tabla = tabla_comparativa(_metricas(), _metricas(f1=0.65, auc=0.72))
    texto = veredicto(
        tabla,
        {"pvalue": 0.001, "solo_v1": 3, "solo_v2": 40, "discordantes": 43},
        _sensibilidad(delta_igual_umbral=0.02),
    )
    assert "V2 supera a V1" in texto
    assert "promover la V2" in texto


def test_veredicto_se_queda_con_v1_si_la_v2_pierde():
    tabla = tabla_comparativa(_metricas(), _metricas(f1=0.55, auc=0.66))
    texto = veredicto(
        tabla,
        {"pvalue": 0.001, "solo_v1": 40, "solo_v2": 3, "discordantes": 43},
        _sensibilidad(delta_igual_umbral=-0.02),
    )
    assert "V1 se mantiene" in texto


def test_veredicto_atribuye_la_mejora_al_umbral_si_los_modelos_empatan():
    # escenario real del A/B: la V2 gana desplegada pero empata a igual umbral
    tabla = tabla_comparativa(
        _metricas(f1=0.5129, auc=0.8239),
        {
            "roc_auc": 0.8248,
            "average_precision": 0.1287,
            "precision": 0.1144,
            "recall": 0.3773,
            "f1_macro": 0.5638,
        },
    )
    texto = veredicto(
        tabla,
        {"pvalue": 0.0, "solo_v1": 32983, "solo_v2": 527409, "discordantes": 560392},
        _sensibilidad(),
    )
    assert "La V2 gana en el despliegue" in texto
    assert "los modelos empatan" in texto
    assert "la explica el umbral" in texto
    assert "no el reentrenamiento" in texto
    assert "inocuo" in texto


# --------------------------------------------------------------------------
# reporte
# --------------------------------------------------------------------------


def _contexto():
    return {
        "fecha": "2026-11-09",
        "commit": "abc1234",
        "seed": SEED,
        "eval_inicio": "2025-07-01",
        "eval_fin": "2026-09-30",
        "eval_filas": 4305495,
        "eval_pct_riesgo": 0.29,
        "v1_alias": "models:/mibici-equipo01-p2-riesgo@champion",
        "v1_version": 2,
        "v1_run": "runv1runv1",
        "v1_C": 1.0,
        "v1_umbral": 0.45,
        "v2_alias": "models:/mibici-equipo01-p2-riesgo@ab-candidate",
        "v2_version": 3,
        "v2_run": "runv2runv2",
        "v2_C": 0.1,
        "v2_umbral": 0.42,
    }


def test_reporte_incluye_metodologia_tabla_y_veredicto():
    tabla = tabla_comparativa(_metricas(), _metricas(f1=0.65, auc=0.72))
    sensibilidad = tabla_sensibilidad(
        np.array([0, 1] * 20),
        np.array([0.1, 0.9] * 20),
        np.array([0.2, 0.8] * 20),
        umbral_v1=0.5,
        umbral_v2=0.7,
    )
    mcnemar = {"pvalue": 0.001, "solo_v1": 3, "solo_v2": 40, "discordantes": 43}
    texto = renderizar_reporte(tabla, mcnemar, _contexto(), sensibilidad)

    # el set de evaluacion tiene que estar definido POR FECHAS
    assert "2025-07-01" in texto and "2026-09-30" in texto
    assert "Set único de evaluación" in texto
    assert "| f1_macro |" in texto
    assert "Lectura de la diferencia" in texto
    assert "Sensibilidad al umbral" in texto
    assert "V2 @ umbral de la V1" in texto  # la fila que separa modelo de umbral
    assert "McNemar" in texto
    assert "## Veredicto" in texto
    # las dos versiones, con sus aliases
    assert "@champion" in texto and "@ab-candidate" in texto


def test_sensibilidad_cruza_modelo_y_umbral():
    y = np.array([0, 1] * 50)
    prob_v1 = np.array([0.4, 0.6] * 50)
    prob_v2 = np.array([0.4, 0.6] * 50)
    sens = tabla_sensibilidad(y, prob_v1, prob_v2, umbral_v1=0.5, umbral_v2=0.7)

    assert list(sens["configuracion"]) == [
        "V1 @ su umbral",
        "V2 @ su umbral",
        "V1 @ umbral de la V2",
        "V2 @ umbral de la V1",
    ]
    assert set(sens.columns) == {
        "configuracion",
        "umbral",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1_macro",
    }
    # con umbral 0.7 los positivos (0.6) se pierden: el recall se va a 0
    assert sens.loc[sens["configuracion"] == "V2 @ su umbral", "recall"].iloc[0] == 0.0
    # el AUC no depende del umbral: las 4 filas miden lo mismo
    assert sens["roc_auc"].nunique() == 1


def test_conclusion_de_sensibilidad_detecta_si_gana_a_igual_umbral():
    y = np.array([0, 1] * 50)
    prob = np.array([0.4, 0.6] * 50)
    sens = tabla_sensibilidad(y, prob, prob, umbral_v1=0.5, umbral_v2=0.5)
    texto = _conclusion_sensibilidad(sens)
    assert "A igual umbral (0.50)" in texto


def test_pvalue_extremo_se_escribe_en_legible():
    # con 560 mil discordantes el binomtest sublima a 0.0: no puede salir "0"
    assert _texto_pvalue(0.0) == "p < 0.0001"
    assert _texto_pvalue(1e-9) == "p < 0.0001"
    assert _texto_pvalue(0.0386) == "p = 0.0386"


def test_lectura_explica_si_la_diferencia_viene_del_umbral():
    # el caso real: el AUC apenas se mueve y el F1 sube por el umbral nuevo
    v1 = {
        "roc_auc": 0.8239,
        "average_precision": 0.1283,
        "precision": 0.0805,
        "recall": 0.6819,
        "f1_macro": 0.5129,
    }
    v2 = {
        "roc_auc": 0.8248,
        "average_precision": 0.1287,
        "precision": 0.1144,
        "recall": 0.3773,
        "f1_macro": 0.5638,
    }
    texto = lectura_de_la_diferencia(tabla_comparativa(v1, v2), 0.5, 0.7)
    assert "ROC AUC prácticamente no se mueve" in texto
    assert "umbral 0.70" in texto and "0.50" in texto
    assert "precisión" in texto and "recall" in texto

    # si el AUC se mueve de verdad, lo dice
    texto_auc = lectura_de_la_diferencia(
        tabla_comparativa(_metricas(), _metricas(f1=0.70, auc=0.85)), 0.5, 0.5
    )
    assert "ordena mejor" in texto_auc


# --------------------------------------------------------------------------
# datos de prueba
# --------------------------------------------------------------------------


def _parquet_sintetico(ruta: Path) -> pd.DataFrame:
    """Parquet con la misma forma que features_estacion_hora.parquet.

    Una estacion con picos de salida (hora 6) y de llegada (hora 2), de modo
    que el proxy de riesgo tiene las dos clases en todos los splits.
    """
    horas = pd.date_range("2023-05-01", "2026-09-30 23:00", freq="h")
    rng = np.random.default_rng(SEED)
    n = len(horas)
    df = pd.DataFrame(
        {
            "hora": horas,
            "estacion_id": np.full(n, 1, dtype="int32"),
            "salidas": np.where(horas.hour == 6, 100, 0).astype("int32"),
            "llegadas": np.where(horas.hour == 2, 100, 0).astype("int32"),
            "capacidad": np.full(n, 50.0, dtype="float32"),
            "latitud": np.full(n, 20.65, dtype="float32"),
            "longitud": np.full(n, -103.35, dtype="float32"),
            "hora_del_dia": horas.hour.astype("int16"),
            "dia_semana": horas.dayofweek.astype("int16"),
            "mes": horas.month.astype("int16"),
            "es_fin_de_semana": (horas.dayofweek >= 5).astype("int8"),
        }
    )
    historicas = [
        "salidas_lag_1h",
        "salidas_lag_24h",
        "salidas_lag_168h",
        "salidas_media_7d",
        "salidas_prom_4sem",
        "neto_lag_1h",
        "neto_lag_24h",
        "neto_media_7d",
    ]
    for columna in historicas:
        df[columna] = rng.normal(size=n).astype("float32")

    df = assign_split(df)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    return df


def _registrar_v1(df: pd.DataFrame) -> None:
    """Pone una V1 plausible en el alias `champion` (como ya esta en produccion)."""
    etiquetado = add_risk_label(df).dropna(subset=FEATURES + ["riesgo"])
    train = etiquetado.loc[etiquetado["split"] == "train"]
    modelo = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=SEED, max_iter=1000, class_weight="balanced"),
    ).fit(train[FEATURES], train["riesgo"].astype(int))

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="p2-v1-champion-test") as run:
        mlflow.set_tags({"pregunta": "P2", "stage_of_work": "search"})
        mlflow.log_params({"best_C": 1.0, "umbral_final": 0.5, "modelo": "logreg"})
        mlflow.sklearn.log_model(modelo, name="model")

    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", ab_testing.NOMBRE_MODELO)
    MlflowClient().set_registered_model_alias(
        ab_testing.NOMBRE_MODELO, "champion", str(version.version)
    )


@pytest.fixture
def tracking_tmp(tmp_path, monkeypatch):
    """Backend MLflow aislado (sqlite nuevo) para no ensuciar el mlflow.db real.

    Solo se toca la variable de entorno (mismo patron que test_register.py):
    fijar ademas el tracking URI explicitamente dejaria un estado global que
    romperia los tests que corren despues.
    """
    uri = f"sqlite:///{tmp_path}/mlflow_test.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


# --------------------------------------------------------------------------
# flujo integral: train-v2 + compare
# --------------------------------------------------------------------------


def test_flujo_completo_entrena_v2_y_compara_sin_tocar_champion(
    tmp_path, monkeypatch, tracking_tmp
):
    ruta = tmp_path / "features.parquet"
    df = _parquet_sintetico(ruta)
    _registrar_v1(df)

    monkeypatch.setattr(ab_testing, "REPORT_PATH", tmp_path / "ab_testing_p2.md")
    monkeypatch.setattr(ab_testing, "CSV_PATH", tmp_path / "ab_testing_p2.csv")
    monkeypatch.setattr(ab_testing, "FIGURES_PATH", tmp_path / "figuras")

    ab_testing.entrenar_v2(ruta)

    cliente = MlflowClient()
    v2 = cliente.get_model_version_by_alias(ab_testing.NOMBRE_MODELO, "ab-candidate")
    v1 = cliente.get_model_version_by_alias(ab_testing.NOMBRE_MODELO, "champion")

    # la V2 es una version NUEVA del mismo modelo y el champion no se movio
    assert v2.run_id != v1.run_id
    assert v2.version != v1.version
    # tag que impide que register.py la promueva por accidente
    assert cliente.get_run(v2.run_id).data.tags["stage_of_work"] == "ab_candidate"
    # el umbral y el C quedan como parametros del run (los usa el compare)
    params = cliente.get_run(v2.run_id).data.params
    assert "umbral_final" in params and "best_C" in params

    tabla = ab_testing.comparar(ruta)

    assert set(tabla["metrica"]) == {
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1_macro",
    }
    assert (tmp_path / "ab_testing_p2.md").exists()
    assert (tmp_path / "ab_testing_p2.csv").exists()
    assert (tmp_path / "figuras" / "roc_v1_vs_v2.png").exists()
    assert (tmp_path / "figuras" / "metricas_v1_vs_v2.png").exists()

    reporte = (tmp_path / "ab_testing_p2.md").read_text(encoding="utf-8")
    assert "split `test`" in reporte  # un solo set de evaluacion para los dos
    assert "McNemar" in reporte
    assert "@champion" in reporte and "@ab-candidate" in reporte


def test_cargar_con_etiqueta_falla_con_un_path_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="mibici.features"):
        ab_testing.cargar_con_etiqueta(tmp_path / "no_existe.parquet")
