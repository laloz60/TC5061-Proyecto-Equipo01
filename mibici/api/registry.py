"""Carga de modelos desde el Model Registry para el contenedor (Actividad 8 y 10).

Flujo robusto para el contenedor:
1. Si ``MIBICI_MOCK_REGISTRY=1`` (entorno CI smoke test), devuelve un modelo
   ligero determinista que respeta la interfaz PyFunc y sklearn_model.
2. Intenta ``models:/<nombre>@champion`` (funciona en el host).
3. Si falla (dentro del contenedor el artifact_uri guardado es una ruta
   local absoluta del host), resuelve el alias, obtiene el id del logged
   model y reconstruye la URI como ``mlflow-artifacts:/`` para descargar
   los artefactos a traves del proxy del servidor MLflow.
"""

from __future__ import annotations

import os

from loguru import logger
import mlflow
import mlflow.pyfunc
import numpy as np

_ALIASES = {
    "p1": "mibici-equipo01-p1-demanda",
    "p2": "mibici-equipo01-p2-riesgo",
    "p3": "mibici-equipo01-p3-grupos",
}


class _MockSklearnModel:
    """Simula el estimador subyacente de scikit-learn para P2 en CI."""

    def predict_proba(self, df):
        return np.array([[0.25, 0.75]], dtype="float32")


class _MockModelImpl:
    """Simula el atributo ``_model_impl`` de MLflow PyFunc."""

    def __init__(self) -> None:
        self.sklearn_model = _MockSklearnModel()


class _MockPyFuncModel:
    """Modelo mock determinista para validar contratos de la API en CI (issue #159)."""

    def __init__(self, key: str) -> None:
        self.key = key
        self._model_impl = _MockModelImpl()

    def predict(self, data):
        if self.key == "p1":
            return np.array([5.62], dtype="float32")
        if self.key == "p2":
            return np.array([1], dtype="int32")
        return np.array([2], dtype="int32")


def load_champion(key: str) -> mlflow.pyfunc.PyFuncModel:
    """Carga el modelo ``@champion`` de la pregunta indicada (p1, p2 o p3)."""
    if os.getenv("MIBICI_MOCK_REGISTRY", "0") == "1":
        logger.warning(f"{key}: usando modelo mock para smoke test en CI (MIBICI_MOCK_REGISTRY=1)")
        return _MockPyFuncModel(key)  # type: ignore[return-value]

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
