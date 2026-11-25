"""Pruebas del Model Registry (issues #95 y #191): selección, descripción, tags e idempotencia.

Los tests de integración usan un backend MLflow propio en sqlite bajo tmp_path,
así no tocan el mlflow.db del proyecto.
"""

from types import SimpleNamespace

import mlflow
from mlflow.tracking import MlflowClient
import pytest
import typer

from mibici.modeling import register
from mibici.modeling.register import (
    _descripcion,
    _mejor_run_con_modelo,
    _run_tiene_modelo,
    _tags_produccion,
)
from mibici.modeling.tracking import EXPERIMENT_NAME

NOMBRE_P1 = "mibici-equipo01-p1-demanda"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _ClienteFalso:
    """Cliente mínimo para _run_tiene_modelo: solo responde list_artifacts."""

    def __init__(self, artefactos):
        self._artefactos = artefactos
        self.llamadas = 0

    def list_artifacts(self, run_id, path):
        self.llamadas += 1
        return self._artefactos


def _run_falso(pregunta="P1", commit="abc1234", metrica="val_mae", valor=0.8075):
    """Objeto con la forma de un Run de MLflow, para funciones puras."""
    tags = {"pregunta": pregunta}
    if commit is not None:
        tags["commit"] = commit
    return SimpleNamespace(
        data=SimpleNamespace(
            tags=tags,
            params={
                "data_manifest_sha256": "f" * 64,
                "data_parquet_sha256": "e" * 64,
            },
            metrics={metrica: valor},
        ),
        info=SimpleNamespace(run_name="p1-candidate-rf-n200", run_id="run-123"),
    )


@pytest.fixture
def tracking_tmp(tmp_path, monkeypatch):
    """Backend MLflow aislado (sqlite nuevo) para no ensuciar el mlflow.db real."""
    uri = f"sqlite:///{tmp_path}/mlflow_test.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def _crear_run(pregunta, stage, metrica, valor, con_modelo=True):
    """Corrige un run con tags y (opcional) un modelo logueado."""
    with mlflow.start_run(run_name=f"{pregunta}-{stage}") as run:
        mlflow.set_tags({"pregunta": pregunta, "stage_of_work": stage})
        mlflow.log_metric(metrica, valor)
        if con_modelo:
            from sklearn.linear_model import Ridge

            modelo = Ridge().fit([[0.0], [1.0]], [0.0, 1.0])
            mlflow.sklearn.log_model(modelo, name="model")
    return run.info.run_id


# --------------------------------------------------------------------------
# _run_tiene_modelo
# --------------------------------------------------------------------------

def test_run_tiene_modelo_detecta_el_arbol_del_flavor():
    cliente = _ClienteFalso([SimpleNamespace(path="model/MLmodel")])
    assert _run_tiene_modelo(cliente, "r1", set()) is True


def test_run_tiene_modelo_sin_mlmodel_es_false():
    cliente = _ClienteFalso([SimpleNamespace(path="model/otro.txt")])
    assert _run_tiene_modelo(cliente, "r1", set()) is False


def test_run_tiene_modelo_no_consulta_si_ya_es_logueado():
    cliente = _ClienteFalso([])
    assert _run_tiene_modelo(cliente, "r1", {"r1"}) is True
    assert cliente.llamadas == 0  # no gastó una llamada al servidor


# --------------------------------------------------------------------------
# _descripcion
# --------------------------------------------------------------------------

def test_descripcion_cita_run_commit_y_metrica():
    texto = _descripcion(_run_falso(), "val_mae")
    assert "p1-candidate-rf-n200" in texto
    assert "run-123" in texto
    assert "abc1234" in texto
    assert "val_mae=0.8075" in texto
    # linaje de datos: los hashes del manifest y del parquet, truncados a 16
    assert "f" * 16 in texto
    assert "e" * 16 in texto


def test_descripcion_sin_commit_dice_desconocido():
    texto = _descripcion(_run_falso(commit=None), "val_mae")
    assert "desconocido" in texto


# --------------------------------------------------------------------------
# main: validación de la opción
# --------------------------------------------------------------------------

def test_main_rechaza_pregunta_invalida():
    with pytest.raises(typer.BadParameter):
        register.main(question="p9")


def test_main_con_run_inexistente_sale_con_codigo_de_error(tracking_tmp):
    # el experimento no existe en el backend nuevo: _registrar falla y main
    # debe terminar con Exit(1), no con una excepcion sin capturar
    with pytest.raises(typer.Exit) as error:
        register.main(question="p1")
    assert error.value.exit_code == 1


# --------------------------------------------------------------------------
# _mejor_run_con_modelo: selección sobre el backend real (sqlite tmp)
# --------------------------------------------------------------------------

def test_mejor_run_elige_el_menor_val_mae(tracking_tmp):
    mlflow.set_experiment(EXPERIMENT_NAME)
    _crear_run("P1", "candidate", "val_mae", 0.95)
    mejor = _crear_run("P1", "candidate", "val_mae", 0.80)

    run = _mejor_run_con_modelo(MlflowClient(), "p1", "val_mae", "ASC")
    assert run.info.run_id == mejor


def test_mejor_run_prefiere_candidates_aunque_no_sean_los_mejores(tracking_tmp):
    mlflow.set_experiment(EXPERIMENT_NAME)
    _crear_run("P1", "search", "val_mae", 0.50)  # mejor metrica, pero es exploratorio
    candidato = _crear_run("P1", "candidate", "val_mae", 0.70)

    run = _mejor_run_con_modelo(MlflowClient(), "p1", "val_mae", "ASC")
    assert run.info.run_id == candidato


def test_mejor_run_cae_a_las_corridas_previas_si_no_hay_candidato(tracking_tmp):
    mlflow.set_experiment(EXPERIMENT_NAME)
    padre = _crear_run("P1", "search", "val_mae", 0.50)

    run = _mejor_run_con_modelo(MlflowClient(), "p1", "val_mae", "ASC")
    assert run.info.run_id == padre


def test_mejor_run_ignora_runs_sin_la_metrica_de_validacion(tracking_tmp):
    # caso real de P3: candidatos viejos con modelo pero sin val_*; no deben ganar
    mlflow.set_experiment(EXPERIMENT_NAME)
    _crear_run("P3", "candidate", "test_silhouette_score", 0.20)
    con_val = _crear_run("P3", "previo", "val_silhouette_score", 0.33)

    run = _mejor_run_con_modelo(MlflowClient(), "p3", "val_silhouette_score", "DESC")
    assert run.info.run_id == con_val


def test_mejor_run_sin_experimento_lanza_error_claro(tracking_tmp):
    # backend vacio: el error debe ser accionable, no un None silencioso
    with pytest.raises(typer.BadParameter, match="No existe el experimento"):
        _mejor_run_con_modelo(MlflowClient(), "p1", "val_mae", "ASC")


def test_mejor_run_ignora_candidates_sin_modelo(tracking_tmp):
    mlflow.set_experiment(EXPERIMENT_NAME)
    _crear_run("P1", "candidate", "val_mae", 0.50, con_modelo=False)
    con_modelo = _crear_run("P1", "candidate", "val_mae", 0.90, con_modelo=True)

    run = _mejor_run_con_modelo(MlflowClient(), "p1", "val_mae", "ASC")
    assert run.info.run_id == con_modelo


# --------------------------------------------------------------------------
# _registrar: registro real e idempotencia
# --------------------------------------------------------------------------

def test_registrar_es_idempotente_y_apunta_champion(tracking_tmp):
    mlflow.set_experiment(EXPERIMENT_NAME)
    _crear_run("P1", "candidate", "val_mae", 0.80)

    register._registrar("p1")
    cliente = MlflowClient()
    assert len(cliente.search_model_versions(f"name='{NOMBRE_P1}'")) == 1

    # segunda vez: se omite y no crea una version duplicada
    register._registrar("p1")
    assert len(cliente.search_model_versions(f"name='{NOMBRE_P1}'")) == 1

    alias = cliente.get_model_version_by_alias(NOMBRE_P1, "champion")
    modelo = cliente.get_registered_model(NOMBRE_P1)
    assert "val_mae" in modelo.description  # descripcion con linaje y metrica
    assert alias.run_id is not None


# --------------------------------------------------------------------------
# estado de produccion (issue #191): tags, archivado y @ab-candidate
# --------------------------------------------------------------------------

def _tags(cliente, version):
    return cliente.get_model_version(NOMBRE_P1, str(version)).tags


def test_tags_produccion_lleva_stage_pregunta_y_metrica():
    tags = _tags_produccion("p1", "val_mae", 0.80791)
    assert tags == {
        "stage": "production",
        "pregunta": "P1",
        "metrica_seleccion": "val_mae",
        "valor_metrica": "0.8079",
    }


def test_registrar_marca_el_champion_como_produccion(tracking_tmp):
    mlflow.set_experiment(EXPERIMENT_NAME)
    _crear_run("P1", "candidate", "val_mae", 0.80)

    register._registrar("p1")
    cliente = MlflowClient()
    champion = cliente.get_model_version_by_alias(NOMBRE_P1, "champion")
    assert champion.tags["stage"] == "production"
    assert champion.tags["metrica_seleccion"] == "val_mae"
    assert champion.tags["valor_metrica"] == "0.8000"
    assert "val_mae" in champion.description


def test_registrar_archiva_el_champion_anterior(tracking_tmp):
    mlflow.set_experiment(EXPERIMENT_NAME)
    _crear_run("P1", "candidate", "val_mae", 0.90)
    register._registrar("p1")

    # aparece un candidato mejor: nueva version champion, la anterior se archiva
    _crear_run("P1", "candidate", "val_mae", 0.70)
    register._registrar("p1")

    cliente = MlflowClient()
    champion = cliente.get_model_version_by_alias(NOMBRE_P1, "champion")
    assert str(champion.version) == "2"
    assert _tags(cliente, 2)["stage"] == "production"
    assert _tags(cliente, 1)["stage"] == "archived"


def test_registrar_completa_tags_de_una_version_ya_registrada(tracking_tmp):
    # caso de las v2 actuales: el run ya esta registrado, pero sin tags ni alias
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_id = _crear_run("P1", "candidate", "val_mae", 0.80)
    mlflow.register_model(f"runs:/{run_id}/model", NOMBRE_P1)

    register._registrar("p1")
    cliente = MlflowClient()
    assert len(cliente.search_model_versions(f"name='{NOMBRE_P1}'")) == 1
    assert _tags(cliente, 1)["stage"] == "production"
    assert str(cliente.get_model_version_by_alias(NOMBRE_P1, "champion").version) == "1"


def test_registrar_marca_ab_candidate_como_evaluacion(tracking_tmp):
    mlflow.set_experiment(EXPERIMENT_NAME)
    _crear_run("P1", "candidate", "val_mae", 0.80)
    retador = _crear_run("P1", "ab_test", "val_mae", 0.95)
    cliente = MlflowClient()
    version_ab = mlflow.register_model(f"runs:/{retador}/model", NOMBRE_P1).version
    cliente.set_registered_model_alias(NOMBRE_P1, "ab-candidate", version_ab)

    register._registrar("p1")
    ab = cliente.get_model_version_by_alias(NOMBRE_P1, "ab-candidate")
    assert str(ab.version) == str(version_ab)  # el alias no se mueve
    assert ab.tags["stage"] == "evaluation"
    assert "No esta en produccion" in ab.description
    champion = cliente.get_model_version_by_alias(NOMBRE_P1, "champion")
    assert str(champion.version) != str(version_ab)
