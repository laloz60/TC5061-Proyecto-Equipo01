import os
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from loguru import logger
from mibici.api.schemas import ModelFeaturesInput, ClusterInput, P1Response, P2Response, P3Response

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Conectando al Model Registry en: {tracking_uri}")
    try:
        ml_models["p1"] = mlflow.pyfunc.load_model("models:/mibici-equipo01-p1-demanda@champion")
        ml_models["p2"] = mlflow.pyfunc.load_model("models:/mibici-equipo01-p2-riesgo@champion")
        ml_models["p3"] = mlflow.pyfunc.load_model("models:/mibici-equipo01-p3-grupos@champion")
        logger.success("Modelos cargados exitosamente")
    except Exception as e:
        logger.error(f"Error al cargar modelos: {e}")
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
    if "p1" not in ml_models: raise HTTPException(status_code=503, detail="Modelo P1 no disponible")
    pred = ml_models["p1"].predict(pd.DataFrame([data.model_dump()]))
    return P1Response(demanda_predicha_viajes=float(pred[0]))

@app.post("/predict/p2-riesgo", response_model=P2Response, tags=["P2 - Clasificación"])
def predict_p2(data: ModelFeaturesInput):
    if "p2" not in ml_models: raise HTTPException(status_code=503, detail="Modelo P2 no disponible")
    df = pd.DataFrame([data.model_dump()])
    base_model = ml_models["p2"]._model_impl.sklearn_model
    return P2Response(riesgo_desabasto_o_saturacion=int(ml_models["p2"].predict(df)[0]),
                      probabilidad_riesgo=float(base_model.predict_proba(df)[0][1]))

@app.post("/predict/p3-clustering", response_model=P3Response, tags=["P3 - Clustering"])
def predict_p3(data: ClusterInput):
    if "p3" not in ml_models: raise HTTPException(status_code=503, detail="Modelo P3 no disponible")
    pred = ml_models["p3"].predict(np.array(data.perfil_horario).reshape(1, 24))
    return P3Response(cluster_asignado=int(pred[0]))