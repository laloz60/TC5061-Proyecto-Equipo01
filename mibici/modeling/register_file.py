import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

def registrar_y_promover_con_ids():
    client = MlflowClient()
    
    # Mapeo de los mejores Run IDs obtenidos en tus corridas
    modelos_runs = {
        "mibici-equipo01-p2-riesgo": "350d144e8665446db9aed28219b864a2",  # Mejor candidata LogReg (C=10.0)
        "mibici-equipo01-p3-grupos": "ae57d281f2674a16a998e7801a95720a"   # Candidata K-Means++ (K=5)
    }
    
    for nombre_reg, run_id in modelos_runs.items():
        try:
            model_uri = f"runs:/{run_id}/model"
            logger.info(f"Registrando modelo desde {model_uri} en {nombre_reg}...")
            
            # Registrar el modelo en el Model Registry
            reg_version = mlflow.register_model(model_uri=model_uri, name=nombre_reg)
            
            # Asignar el alias de producción @champion
            client.set_registered_model_alias(nombre_reg, "champion", str(reg_version.version))
            logger.success(f"Modelo {nombre_reg} versión {reg_version.version} promovido exitosamente a @champion.")
            
        except Exception as e:
            logger.error(f"Error al registrar {nombre_reg}: {e}")

if __name__ == "__main__":
    registrar_y_promover_con_ids()