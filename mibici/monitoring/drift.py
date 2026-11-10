import json
import pandas as pd
from pathlib import Path
from loguru import logger
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mibici.config as config

def generate_drift_report():
    logger.info("Cargando features procesadas para el análisis de Drift...")
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "features_estacion_hora.parquet")
    
    logger.info("Simulando partición: Referencia (Train histórico) vs Actual (Producción)...")
    # Muestreo representativo para no desbordar memoria
    reference_data = df.iloc[:int(len(df)*0.5)].sample(n=30000, random_state=config.SEED)
    current_data = df.iloc[int(len(df)*0.5):].sample(n=30000, random_state=config.SEED)
    
    logger.info("Calculando Data Drift paramétrico con Evidently...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # 1. Guardar HTML (Para el reporte humano / PDF)
    html_path = Path("reports/drift_report.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    drift_report.save_html(str(html_path))
    
    # [MEJORA] 2. Guardar JSON (Para automatización MLOps / GitHub Actions)
    json_path = Path("reports/drift_report.json")
    drift_report.save_json(str(json_path))
    
    # [MEJORA] 3. Leer el JSON programáticamente y tomar decisiones
    with open(json_path, 'r') as f:
        drift_result = json.load(f)
        
    # Extraer las métricas exactas del JSON de Evidently
    dataset_drift = drift_result["metrics"][0]["result"]["dataset_drift"]
    share_of_drifted_columns = drift_result["metrics"][0]["result"]["share_of_drifted_columns"]
    
    logger.success(f"Reportes guardados exitosamente en {html_path.parent}")
    
    # Lógica de decisión
    if dataset_drift:
        logger.warning(f"¡ALERTA DE DRIFT! El {share_of_drifted_columns*100:.1f}% de las variables cambió.")
        logger.warning("Recomendación de Sistema: Se requiere disparar el pipeline de Continuous Training.")
    else:
        logger.info(f"Sistema estable. Solo el {share_of_drifted_columns*100:.1f}% de las variables cambió. No requiere reentrenamiento.")

if __name__ == "__main__":
    generate_drift_report()