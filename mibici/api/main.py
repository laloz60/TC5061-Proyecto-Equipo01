from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from loguru import logger
import mlflow
import numpy as np
import pandas as pd

from mibici.api.registry import load_champion
from mibici.api.schemas import ClusterInput, ModelFeaturesInput, P1Response, P2Response, P3Response

ml_models = {}

# Dtypes del input schema de los modelos P1/P2: el modelo los registró como
# int32/float32 y pandas crea int64/float64 por defecto, lo que hace fallar
# la validación de schema de mlflow ("Can not safely convert ...")
_INT_FEATURES = ("hora_del_dia", "dia_semana", "mes", "es_fin_de_semana")
_FLOAT_FEATURES = (
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "salidas_media_7d",
    "salidas_prom_4sem",
    "neto_lag_1h",
    "neto_lag_24h",
    "neto_media_7d",
    "capacidad",
    "latitud",
    "longitud",
)


def _a_dataframe(data: ModelFeaturesInput) -> pd.DataFrame:
    """Convierte el payload a un DataFrame con los dtypes del modelo."""
    df = pd.DataFrame([data.model_dump()])
    ints = list(_INT_FEATURES)
    floats = list(_FLOAT_FEATURES)
    df[ints] = df[ints].astype("int32")
    df[floats] = df[floats].astype("float32")
    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Conectando al Model Registry en: {tracking_uri}")
    # load_champion intenta models:/<nombre>@champion y, dentro del contenedor,
    # cae al proxy mlflow-artifacts (issue #126)
    for key in ("p1", "p2", "p3"):
        try:
            ml_models[key] = load_champion(key)
        except Exception as e:  # noqa: BLE001 - un fallo no debe tumbar el arranque
            logger.error(f"Error al cargar el modelo {key}: {e}")
    if ml_models:
        logger.success(f"Modelos cargados: {sorted(ml_models)}")
    yield
    ml_models.clear()


app = FastAPI(title="MiBici ML API", version="1.0.0", lifespan=lifespan)


@app.get("/health", tags=["Monitoreo"])
def health_check():
    if not ml_models:
        raise HTTPException(status_code=503, detail="Modelos no disponibles")
    return {"status": "healthy", "modelos_cargados": list(ml_models.keys())}


@app.post("/predict/p1-demanda", response_model=P1Response, tags=["P1 - Regresión"])
def predict_p1(data: ModelFeaturesInput):
    if "p1" not in ml_models:
        raise HTTPException(status_code=503, detail="Modelo P1 no disponible")
    pred = ml_models["p1"].predict(_a_dataframe(data))
    return P1Response(demanda_predicha_viajes=float(pred[0]))


@app.post("/predict/p2-riesgo", response_model=P2Response, tags=["P2 - Clasificación"])
def predict_p2(data: ModelFeaturesInput):
    if "p2" not in ml_models:
        raise HTTPException(status_code=503, detail="Modelo P2 no disponible")
    df = _a_dataframe(data)
    base_model = ml_models["p2"]._model_impl.sklearn_model
    return P2Response(
        riesgo_desabasto_o_saturacion=int(ml_models["p2"].predict(df)[0]),
        probabilidad_riesgo=float(base_model.predict_proba(df)[0][1]),
    )


@app.post("/predict/p3-clustering", response_model=P3Response, tags=["P3 - Clustering"])
def predict_p3(data: ClusterInput):
    if "p3" not in ml_models:
        raise HTTPException(status_code=503, detail="Modelo P3 no disponible")
    pred = ml_models["p3"].predict(np.array(data.perfil_horario).reshape(1, 24))
    return P3Response(cluster_asignado=int(pred[0]))
