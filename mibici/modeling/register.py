"""Registra los modelos candidatos de P1/P2/P3 en el Model Registry (issues #70 y #191)."""

from __future__ import annotations

from loguru import logger
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import typer

from mibici.modeling.tracking import EXPERIMENT_NAME

app = typer.Typer(help="Registra el mejor candidato de cada pregunta en el Model Registry.")

# pregunta -> (nombre en el registry, metrica de seleccion, orden)
# la seleccion se hace en VALIDACION: test queda solo para el reporte final
_MODELOS = {
    "p1": ("mibici-equipo01-p1-demanda", "val_mae", "ASC"),
    "p2": ("mibici-equipo01-p2-riesgo", "val_roc_auc", "DESC"),
    "p3": ("mibici-equipo01-p3-grupos", "val_silhouette_score", "DESC"),
}

# estado de cada version en el registry (tag "stage")
STAGE_PRODUCCION = "production"
STAGE_ARCHIVADO = "archived"
STAGE_EVALUACION = "evaluacion"

ALIAS_CHAMPION = "champion"
ALIAS_AB = "ab-candidate"

_DESCRIPCION_AB = (
    "Version en evaluacion (alias @ab-candidate): retador del A/B test de P2, "
    "reentrenado con la ventana 2023-05 a 2025-06. No esta en produccion: la API "
    "sirve @champion. Resultados y decision en reports/ab_testing_p2.md y "
    "reports/reporte_actividad9.md."
)


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
    # primero candidatos; si ninguno tiene modelo (p. ej. el padre 'search' lo
    # registra), se busca entre todas las corridas previas de la pregunta
    filtros = [f"{base} AND tags.stage_of_work = 'candidate'", base]
    for filtro in filtros:
        runs = cliente.search_runs(
            [experimento.experiment_id], filter_string=filtro, order_by=ordenado
        )
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


def _tags_produccion(pregunta: str, metrica: str, valor: float) -> dict[str, str]:
    """Tags del estado de produccion: pregunta, metrica de seleccion y su valor."""
    return {
        "stage": STAGE_PRODUCCION,
        "pregunta": pregunta.upper(),
        "metrica_seleccion": metrica,
        "valor_metrica": f"{valor:.4f}",
    }


def _version_por_alias(cliente: MlflowClient, nombre: str, alias: str):
    """Version que apunta el alias, o None si el alias no existe."""
    try:
        return cliente.get_model_version_by_alias(nombre, alias)
    except MlflowException:
        return None


def _marcar_ab_candidate(cliente: MlflowClient, nombre: str) -> None:
    """Deja explicito que la version @ab-candidate esta en evaluacion, no en produccion."""
    version = _version_por_alias(cliente, nombre, ALIAS_AB)
    if version is None:
        return
    cliente.set_model_version_tag(nombre, version.version, "stage", STAGE_EVALUACION)
    cliente.update_model_version(nombre, version.version, description=_DESCRIPCION_AB)
    logger.info(f"{nombre} v{version.version} (@{ALIAS_AB}) marcado como {STAGE_EVALUACION}")


def _registrar(pregunta: str) -> None:
    nombre, metrica, orden = _MODELOS[pregunta]
    cliente = MlflowClient()
    run = _mejor_run_con_modelo(cliente, pregunta, metrica, orden)
    if run is None:
        raise typer.BadParameter(
            f"No hay runs con modelo logueable para {pregunta.upper()} en {EXPERIMENT_NAME!r}."
        )

    anterior = _version_por_alias(cliente, nombre, ALIAS_CHAMPION)
    existentes = cliente.search_model_versions(f"name='{nombre}'")
    ya_registrada = next((v for v in existentes if v.run_id == run.info.run_id), None)

    if ya_registrada is None:
        version = str(mlflow.register_model(f"runs:/{run.info.run_id}/model", nombre).version)
    else:
        # no se duplica la version, pero se completan descripcion, tags y alias
        version = str(ya_registrada.version)
        logger.info(f"{nombre} ya esta registrado desde este run (v{version}); se actualiza.")

    descripcion = _descripcion(run, metrica)
    valor = run.data.metrics.get(metrica)
    cliente.update_registered_model(nombre, description=descripcion)
    cliente.update_model_version(nombre, version, description=descripcion)
    for clave, texto in _tags_produccion(pregunta, metrica, valor).items():
        cliente.set_model_version_tag(nombre, version, clave, texto)
    cliente.set_registered_model_alias(nombre, ALIAS_CHAMPION, version)

    if anterior is not None and str(anterior.version) != version:
        cliente.set_model_version_tag(nombre, anterior.version, "stage", STAGE_ARCHIVADO)
        logger.info(f"{nombre} v{anterior.version} deja @{ALIAS_CHAMPION}: {STAGE_ARCHIVADO}")

    _marcar_ab_candidate(cliente, nombre)
    logger.success(f"{nombre} v{version} <- {run.info.run_name} ({metrica}={valor:.4f})")


@app.command()
def main(
    question: str = typer.Option(
        "all", "--question", help="Pregunta a registrar: p1 | p2 | p3 | all."
    ),
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
