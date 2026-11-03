import os
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from loguru import logger

from mibici.api.schemas import ModelFeaturesInput, ClusterInput, P1Response, P2Response, P3Response

# Diccionario global en memoria para guardar los modelos
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos desde el Model Registry al iniciar la API."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info(f"Conectando al Model Registry en: {tracking_uri}")
    
    try:
        # Cargar los modelos usando el alias '@champion'
        logger.info("Cargando modelo P1 (Demanda)...")
        ml_models["p1"] = mlflow.pyfunc.load_model("models:/mibici-equipo01-p1-demanda@champion")
        
        logger.info("Cargando modelo P2 (Riesgo)...")
        ml_models["p2"] = mlflow.pyfunc.load_model("models:/mibici-equipo01-p2-riesgo@champion")
        
        logger.info("Cargando modelo P3 (Clustering)...")
        ml_models["p3"] = mlflow.pyfunc.load_model("models:/mibici-equipo01-p3-grupos@champion")
        
        logger.success("¡Todos los modelos cargados exitosamente!")
    except Exception as e:
        logger.error(f"Error al cargar los modelos desde MLflow: {e}")
        logger.warning("Asegúrate de que el contenedor tenga acceso a MLFLOW_TRACKING_URI")
    
    yield
    ml_models.clear()

app = FastAPI(
    title="MiBici Machine Learning API",
    description="API REST para consumo de modelos P1, P2 y P3. Maneja validación estricta con Pydantic (errores 422 automáticos).",
    version="1.0.0",
    lifespan=lifespan
)

# =======================================================
# ENDPOINT HEALTH (Requisito Issue #127)
# =======================================================
@app.get("/health", tags=["Monitoreo"])
def health_check():
    """Chequeo de salud para orquestadores de contenedores (Docker/K8s)."""
    # Si por alguna razón MLflow falló y los modelos están vacíos, devuelve 503
    if not ml_models:
        raise HTTPException(status_code=503, detail="Los modelos no están listos o MLflow no responde.")
    
    return {
        "status": "healthy",
        "modelos_cargados": list(ml_models.keys())
    }

@app.get("/", tags=["Monitoreo"])
def read_root():
    return {"mensaje": "MiBici ML API Activa. Navega a /docs para ver la interfaz Swagger."}

# =======================================================
# ENDPOINTS DE PREDICCIÓN
# =======================================================
@app.post("/predict/p1-demanda", response_model=P1Response, tags=["P1 - Regresión"])
def predict_p1(data: ModelFeaturesInput):
    """Predice cuántos viajes iniciarán en la estación en la siguiente hora."""
    if "p1" not in ml_models:
        raise HTTPException(status_code=503, detail="Modelo P1 no disponible") # Requisito Issue 127
    
    df_input = pd.DataFrame([data.model_dump()])
    pred = ml_models["p1"].predict(df_input)
    return P1Response(demanda_predicha_viajes=float(pred[0]))

@app.post("/predict/p2-riesgo", response_model=P2Response, tags=["P2 - Clasificación"])
def predict_p2(data: ModelFeaturesInput):
    """Predice si la estación se quedará vacía o saturada en el horizonte de horas."""
    if "p2" not in ml_models:
        raise HTTPException(status_code=503, detail="Modelo P2 no disponible") # Requisito Issue 127
    
    df_input = pd.DataFrame([data.model_dump()])
    pred_class = ml_models["p2"].predict(df_input)
    
    base_model = ml_models["p2"]._model_impl.sklearn_model
    pred_prob = base_model.predict_proba(df_input)[0][1]
    
    return P2Response(
        riesgo_desabasto_o_saturacion=int(pred_class[0]),
        probabilidad_riesgo=float(pred_prob)
    )

@app.post("/predict/p3-clustering", response_model=P3Response, tags=["P3 - Clustering"])
def predict_p3(data: ClusterInput):
    """Clasifica una estación en un grupo basándose en su perfil de uso de 24h."""
    if "p3" not in ml_models:
        raise HTTPException(status_code=503, detail="Modelo P3 no disponible") # Requisito Issue 127
    
    X = np.array(data.perfil_horario).reshape(1, 24)
    pred = ml_models["p3"].predict(X)
    return P3Response(cluster_asignado=int(pred[0]))