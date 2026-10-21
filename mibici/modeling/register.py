"""Registra los modelos candidatos de P1/P2/P3 en el Model Registry (issue #70)."""

from __future__ import annotations

from loguru import logger
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import typer

from mibici.modeling.tracking import EXPERIMENT_NAME

app = typer.Typer(help="Registra el mejor candidato de cada pregunta en el Model Registry.")

# pregunta -> (nombre en el registry, metrica de seleccion, orden)
_MODELOS = {
    "p1": ("mibici-equipo01-p1-demanda", "test_mae", "ASC"),
    "p2": ("mibici-equipo01-p2-riesgo", "test_roc_auc", "DESC"),
    "p3": ("mibici-equipo01-p3-grupos", "test_silhouette_score", "DESC"),
}


def _run_tiene_modelo(cliente: MlflowClient, run_id: str, logueados: set[str]) -> bool:
    """El run tiene un modelo logueado por el flavor o sus artefactos completos en 'model'."""
    if run_id in logueados:
        return True
    hijos = cliente.list_artifacts(run_id, "model")
    return any(a.path == "model/MLmodel" for a in hijos)


def _mejor_run_con_modelo(cliente: MlflowClient, pregunta: str, metrica: str, orden: str):
    """Busca el mejor run candidato de la pregunta que tenga el artefacto 'model'."""
    experimento = cliente.get_experiment_by_name(EXPERIMENT_NAME)
    if experimento is None:
        raise typer.BadParameter(f"No existe el experimento {EXPERIMENT_NAME!r}.")

    modelos = cliente.search_logged_models(experiment_ids=[experimento.experiment_id])
    logueados = {
        m.source_run_id
        for m in modelos
        if m.name == "model" and getattr(m.status, "name", "") == "READY"
    }

    base = f"tags.pregunta = '{pregunta.upper()}'"
    ordenado = [f"metrics.{metrica} {orden}"]
    runs = cliente.search_runs(
        [experimento.experiment_id],
        filter_string=f"{base} AND tags.stage_of_work = 'candidate'",
        order_by=ordenado,
    )
    if not runs:
        # corridas previas a las etiquetas stage_of_work
        runs = cliente.search_runs([experimento.experiment_id], filter_string=base, order_by=ordenado)

    for run in runs:
        if _run_tiene_modelo(cliente, run.info.run_id, logueados):
            return run
    return None


def _descripcion(run, metrica: str) -> str:
    """Arma la descripción con linaje: run de origen, commit y versión de datos."""
    tags, params = run.data.tags, run.data.params
    valor = run.data.metrics.get(metrica)
    return (
        f"Modelo elegido de {tags.get('pregunta', '?')} sobre el experimento {EXPERIMENT_NAME!r}. "
        f"Run de origen: {run.info.run_name} ({run.info.run_id}). "
        f"Commit: {tags.get('commit', 'desconocido')}. "
        f"Datos: manifest {params.get('data_manifest_sha256', '?')[:16]} "
        f"/ parquet {params.get('data_parquet_sha256', '?')[:16]}. "
        f"{metrica}={valor:.4f}. "
        f"Registrado con mibici/modeling/register.py."
    )


def _registrar(pregunta: str) -> None:
    nombre, metrica, orden = _MODELOS[pregunta]
    cliente = MlflowClient()
    run = _mejor_run_con_modelo(cliente, pregunta, metrica, orden)
    if run is None:
        raise typer.BadParameter(
            f"No hay runs con modelo logueable para {pregunta.upper()} en {EXPERIMENT_NAME!r}."
        )

    existentes = cliente.search_model_versions(f"name='{nombre}'")
    if any(v.run_id == run.info.run_id for v in existentes):
        logger.info(f"{nombre} ya esta registrado desde este run; se omite.")
        return

    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", nombre)
    descripcion = _descripcion(run, metrica)
    cliente.update_registered_model(nombre, description=descripcion)
    cliente.update_model_version(nombre, version.version, description=descripcion)
    logger.success(
        f"{nombre} v{version.version} <- {run.info.run_name} "
        f"({metrica}={run.data.metrics.get(metrica):.4f})"
    )


@app.command()
def main(
    question: str = typer.Option("all", "--question", help="Pregunta a registrar: p1 | p2 | p3 | all."),
) -> None:
    """Registra el mejor candidato de cada pregunta (o solo la indicada)."""
    q = question.strip().lower()
    preguntas = list(_MODELOS) if q == "all" else [q]
    if any(p not in _MODELOS for p in preguntas):
        raise typer.BadParameter(f"Opcion invalida: {question!r}. Usa p1, p2, p3 o all.")

    fallidos = []
    for p in preguntas:
        try:
            _registrar(p)
        except (MlflowException, typer.BadParameter) as error:
            logger.error(f"No se pudo registrar {p.upper()}: {error}")
            fallidos.append(p)

    if fallidos:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
