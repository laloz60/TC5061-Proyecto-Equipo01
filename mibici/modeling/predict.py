"""
Módulo de Inferencia y Predicción (Predict) - Proyecto MiBici
--------------------------------------------------------------
Implementa la arquitectura MLOps orientada a objetos (POO) para la carga de modelos
serializados y la ejecución de predicciones en lote (batch inference).
Incluye soporte modular:
    P1 (Regresion).
    P2 (Clasificacion).
    P3 (Clustering).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import typer

from mibici.config import (
    MODELS_DIR,
    NOMENCLATURA_PARQUET_PATH,
    PROCESSED_DATA_DIR,
    SEED,
    UNIFIED_TRIPS_PATH,
)

# Inicialización de la aplicación CLI con Typer
app = typer.Typer(help="CLI modular para la inferencia y predicción de modelos - MiBici")


# =====================================================================
# ARQUITECTURA - BASE PREDICTOR
# =====================================================================


class BasePredictor(ABC):
    """
    Clase base abstracta (Superclase) para estandarizar el proceso de inferencia
    bajo principios sólidos de MLOps y POO.
    """

    def __init__(self, question: str, model_type: str):
        self.question = question
        self.model_type = model_type

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Método abstracto para la carga segura del modelo serializado."""

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Método abstracto para la ejecución de predicciones."""


class KMeansClusterPredictor(BasePredictor):
    """
    Implementación concreta para la inferencia de la Pregunta 3 (P3):
    Asignación de estaciones a los clústeres espaciales y temporales entrenados.
    """

    def __init__(self):
        super().__init__(question="P3", model_type="KMeans")

    def load_model(self, model_path: Path):
        """Carga el modelo K-Means desde el disco validando su existencia previa."""
        logger.info(f"Cargando modelo serializado desde: {model_path}")

        if model_path.exists():
            model = joblib.load(model_path)
            logger.success("Modelo K-Means cargado exitosamente.")
            return model
        else:
            logger.error(
                f"No se encontró el modelo en {model_path}. Entrena el modelo primero usando train.py p3"
            )
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    def predict(self, input_path: Path, model_path: Path, output_path: Path):
        """Ejecuta la inferencia batch para P3 asignando cada estación a su clúster óptimo."""
        model = self.load_model(model_path)

        logger.info(f"Cargando nomenclatura desde: {input_path}")
        if input_path.exists():
            df_nom = pd.read_parquet(input_path)
            n_stations = len(df_nom)
        else:
            logger.warning(
                f"No se encontró el archivo en {input_path}. Usando datos sintéticos de prueba."
            )
            n_stations = 300
            df_nom = pd.DataFrame(
                {
                    "estacion_id": range(n_stations),
                    "nombre": [f"Estación {i}" for i in range(n_stations)],
                }
            )

        # Construcción de perfiles horarios (debe seguir la misma lógica que train.py)
        if UNIFIED_TRIPS_PATH.exists():
            logger.info(f"Calculando perfiles horarios reales desde: {UNIFIED_TRIPS_PATH}")
            df_trips = pd.read_parquet(UNIFIED_TRIPS_PATH, columns=["inicio_viaje", "origen_id"])
            df_trips["hora"] = pd.to_datetime(df_trips["inicio_viaje"]).dt.hour

            profile_counts = pd.crosstab(df_trips["origen_id"], df_trips["hora"])
            profile_counts = profile_counts.reindex(columns=range(24), fill_value=0)

            if "estacion_id" in df_nom.columns:
                profile_counts = profile_counts.reindex(df_nom["estacion_id"], fill_value=0)

            row_sums = profile_counts.sum(axis=1)
            row_sums[row_sums == 0] = 1
            X_inference = (profile_counts.div(row_sums, axis=0)).values
        else:
            logger.warning(
                "No se encontró el archivo de viajes unificados. Usando perfiles sintéticos."
            )
            np.random.seed(SEED)
            X_inference = np.random.rand(n_stations, 24)

        logger.info("Generando asignaciones de clústeres (inferencia batch)...")
        for _ in tqdm(range(5), total=5, desc="Calculando predicciones P3"):
            pass

        cluster_assignments = model.predict(X_inference)
        df_nom["cluster_assigned"] = cluster_assignments

        # Guardado de resultados en formato Parquet optimizado
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_nom.to_parquet(output_path, index=False)
        logger.success(f"Predicciones P3 guardadas exitosamente en: {output_path}")

        # Guardado de resultados en formato Parquet optimizado
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_nom.to_parquet(output_path, index=False)
        logger.success(f"Predicciones P3 guardadas exitosamente en: {output_path}")

        # EVALUACIÓN Y REPORTE
        print("\n" + "=" * 50)
        print("REPORTE DE EVALUACIÓN DE PREDICCIONES P3")
        print("=" * 50)
        print(f"Total de estaciones evaluadas: {len(df_nom)}")

        counts = df_nom["cluster_assigned"].value_counts().sort_index()
        print("\n--- Distribución de Estaciones por Clúster ---")
        for cluster_id, count in counts.items():
            pct = (count / len(df_nom)) * 100
            print(f" Clúster {cluster_id}: {count} estaciones ({pct:.2f}%)")

        sil_infer = None
        if UNIFIED_TRIPS_PATH.exists():
            df_trips = pd.read_parquet(UNIFIED_TRIPS_PATH, columns=["inicio_viaje", "origen_id"])
            df_trips["hora"] = pd.to_datetime(df_trips["inicio_viaje"]).dt.hour
            profile_counts = pd.crosstab(df_trips["origen_id"], df_trips["hora"])
            profile_counts = profile_counts.reindex(columns=range(24), fill_value=0)
            if "estacion_id" in df_nom.columns:
                profile_counts = profile_counts.reindex(df_nom["estacion_id"], fill_value=0)
            row_sums = profile_counts.sum(axis=1)
            row_sums[row_sums == 0] = 1
            X_inf = (profile_counts.div(row_sums, axis=0)).values

            sil_infer = silhouette_score(X_inf, cluster_assignments)
            print("\n--- Métrica de Calidad ---")
            print(f" Silhouette Score en Inferencia: {sil_infer:.4f}")
        print("=" * 50 + "\n")

        # Guardar reporte físico en la carpeta reports/
        report_path = Path("reports/p3_evaluation_summary.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== REPORTE DE EVALUACIÓN DE PREDICCIONES P3 ===\n")
            f.write(f"Total de estaciones evaluadas: {len(df_nom)}\n\n")
            f.write("--- Distribución de Estaciones por Clúster ---\n")
            for cluster_id, count in counts.items():
                pct = (count / len(df_nom)) * 100
                f.write(f"Clúster {cluster_id}: {count} estaciones ({pct:.2f}%)\n")
            if sil_infer is not None:
                f.write(f"\nSilhouette Score en Inferencia: {sil_infer:.4f}\n")
        logger.success(f"Reporte físico guardado en: {report_path}")

        return df_nom


# =====================================================================
# INTERFAZ DE LÍNEA DE COMANDOS (TYPER CLI)
# =====================================================================


@app.command("p3")
def predict_p3(
    model_path: Path = typer.Option(
        MODELS_DIR / "kmeans_p3.pkl", help="Ruta al modelo K-Means entrenado."
    ),
    input_path: Path = typer.Option(
        NOMENCLATURA_PARQUET_PATH, help="Ruta al archivo de nomenclatura."
    ),
    output_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "p3_predictions.parquet", help="Ruta de salida para las predicciones."
    ),
):
    """Comando CLI para ejecutar la inferencia de la Pregunta 3 (P3) con rutas seguras por defecto."""
    logger.info("Iniciando ejecución de predicción para P3 con valores por defecto...")
    predictor = KMeansClusterPredictor()
    predictor.predict(input_path=input_path, model_path=model_path, output_path=output_path)
    logger.success("Inferencia P3 finalizada correctamente.")


@app.command("p1")
def predict_p1(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_features.csv", help="Características P1."
    ),
    model_path: Path = typer.Option(MODELS_DIR / "model_p1.pkl", help="Modelo P1."),
    predictions_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "p1_predictions.csv", help="Predicciones P1."
    ),
):
    """Comando reservado para la inferencia de la Pregunta P1 (Regresión)."""
    logger.info("Inferencia P1 pendiente de implementación modular por el equipo.")


@app.command("p2")
def predict_p2(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_features.csv", help="Características P2."
    ),
    model_path: Path = typer.Option(MODELS_DIR / "model_p2.pkl", help="Modelo P2."),
    predictions_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "p2_predictions.csv", help="Predicciones P2."
    ),
):
    """Comando reservado para la inferencia de la Pregunta P2 (Clasificación)."""
    logger.info("Inferencia P2 pendiente de implementación modular por el equipo.")


@app.command()
def main(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_features.csv", help="Características base."
    ),
    model_path: Path = typer.Option(MODELS_DIR / "model.pkl", help="Modelo base genérico."),
    predictions_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_predictions.csv", help="Predicciones base genéricas."
    ),
):
    """Comando principal por defecto heredado de la plantilla CCDS (Safety Check)."""
    logger.info("Ejecutando pipeline de inferencia por defecto (main)...")
    for i in tqdm(range(10), total=10, desc="Ejecutando inferencia base"):
        if i == 5:
            logger.info("Paso intermedio de inferencia base de la plantilla.")
    logger.success("Inferencia por defecto completada.")


if __name__ == "__main__":
    app()
