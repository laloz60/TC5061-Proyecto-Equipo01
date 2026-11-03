"""Carga de modelos desde el Model Registry para el contenedor (Actividad 8).

Flujo robusto para el contenedor:
1. Intenta ``models:/<nombre>@champion`` (funciona en el host).
2. Si falla (dentro del contenedor el artifact_uri guardado es una ruta
   local absoluta del host), resuelve el alias, obtiene el id del logged
   model y reconstruye la URI como ``mlflow-artifacts:/`` para descargar
   los artefactos a traves del proxy del servidor MLflow.
"""

from __future__ import annotations

from loguru import logger
import mlflow
import mlflow.pyfunc

_ALIASES = {
    "p1": "mibici-equipo01-p1-demanda",
    "p2": "mibici-equipo01-p2-riesgo",
    "p3": "mibici-equipo01-p3-grupos",
}


def load_champion(key: str) -> mlflow.pyfunc.PyFuncModel:
    """Carga el modelo ``@champion`` de la pregunta indicada (p1, p2 o p3)."""
    name = _ALIASES[key]
    model_uri = f"models:/{name}@champion"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"{key}: modelo cargado via {model_uri}")
        return model
    except Exception as direct_error:  # noqa: BLE001 - fallback documentado
        logger.warning(
            f"{key}: no se pudo cargar via {model_uri} ({direct_error}); probando via proxy mlflow-artifacts"
        )

    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(name, "champion")
    # source guarda el logged model como "models:/m-<id>"
    logged_model_id = version.source.rsplit("/", 1)[-1]
    experiment_id = client.get_run(version.run_id).info.experiment_id
    proxy_uri = f"mlflow-artifacts:/{experiment_id}/models/{logged_model_id}/artifacts"
    model = mlflow.pyfunc.load_model(proxy_uri)
    logger.success(f"{key}: modelo cargado via proxy {proxy_uri}")
    return model
