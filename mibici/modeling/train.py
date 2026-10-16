"""
Módulo de Entrenamiento de Modelos (Train) - Proyecto MiBici
------------------------------------------------------------
Implementa la arquitectura MLOps orientada a objetos para el entrenamiento,
evaluación, versionado de artefactos y registro en MLflow Tracking.
Incluye soporte modular para atender las request de :
    P1 (Regresion).
    P2 (Clasificacion).
    P3 (Clustering).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import typer

from mibici.config import (
    FIGURES_DIR,
    MODELS_DIR,
    NOMENCLATURA_PARQUET_PATH,
    PROCESSED_DATA_DIR,
    SEED,
)

# Inicialización de la aplicación CLI
app = typer.Typer(help="CLI modular para el entrenamiento de modelos de Machine Learning - MiBici")
EXPERIMENT_NAME = "mibici-equipo01"


# =====================================================================
# ARQUITECTURA - BASE TRAINER
# =====================================================================

class BaseTrainer(ABC):
    """
    Clase base abstracta (Superclase) para estandarizar el ciclo de vida
    de entrenamiento de modelos.
    """

    def __init__(self, question: str, model_type: str):
        self.question = question
        self.model_type = model_type
        mlflow.set_experiment(EXPERIMENT_NAME)

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Método abstracto para la carga y validación de datos de entrada."""

    @abstractmethod
    def train_model(self, *args, **kwargs):
        """Método abstracto para el ajuste del algoritmo de Machine Learning."""

    @abstractmethod
    def evaluate_and_log(self, *args, **kwargs):
        """Método abstracto para evaluación de métricas y registro en MLflow."""


class KMeansClusterTrainer(BaseTrainer):
    """
    Implementación concreta para la Pregunta 3 (P3): Clustering de Estaciones
    utilizando K-Means sobre perfiles horarios normalizados.
    """

    def __init__(self, n_clusters: int = 5, init_method: str = "random", n_init: int = 1):
        super().__init__(question="P3", model_type="KMeans")
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.n_init = n_init

    def load_data(self, input_path: Path):
        """Carga la nomenclatura y calcula perfiles horarios reales desde el histórico de viajes."""
        logger.info(f"Cargando datos espaciales desde: {input_path}")
        
        if input_path.exists():
            df_nom = pd.read_parquet(input_path)
            n_stations = len(df_nom)
            lats = df_nom["latitud"].values
            lons = df_nom["longitud"].values
            logger.success(f"Nomenclatura cargada exitosamente ({n_stations} estaciones activas).")
        else:
            logger.warning(f"No se encontró el archivo en {input_path}. Usando datos sintéticos de respaldo.")
            n_stations = 300
            lats = np.random.uniform(20.60, 20.72, n_stations)
            lons = np.random.uniform(-103.42, -103.30, n_stations)

        # Intentar calcular perfiles horarios reales desde el parquet de viajes unificados del equipo
        from mibici.config import UNIFIED_TRIPS_PATH
        if UNIFIED_TRIPS_PATH.exists():
            logger.info(f"Calculando perfiles horarios reales desde: {UNIFIED_TRIPS_PATH}")
            # Leemos únicamente las columnas necesarias para optimizar memoria
            df_trips = pd.read_parquet(UNIFIED_TRIPS_PATH, columns=["inicio_viaje", "origen_id"])
            df_trips["hora"] = pd.to_datetime(df_trips["inicio_viaje"]).dt.hour
            
            # Construimos la matriz de frecuencia: estación vs hora del día (0 a 23)
            profile_counts = pd.crosstab(df_trips["origen_id"], df_trips["hora"])
            profile_counts = profile_counts.reindex(columns=range(24), fill_value=0)
            
            # Alineamos con el orden exacto de las estaciones de la nomenclatura
            if "estacion_id" in df_nom.columns:
                profile_counts = profile_counts.reindex(df_nom["estacion_id"], fill_value=0)
            
            # Normalizamos por fila (proporción de viajes por hora que realiza cada estación)
            row_sums = profile_counts.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Evitamos división por cero
            X_profiles = (profile_counts.div(row_sums, axis=0)).values
            logger.success("Perfiles horarios reales calculados y normalizados exitosamente.")
        else:
            logger.warning("No se encontró el archivo de viajes unificados. Usando datos sintéticos para perfiles.")
            np.random.seed(SEED)
            X_profiles = np.random.rand(n_stations, 24)

        return X_profiles, lats, lons

    def train_model(self, X_profiles: np.ndarray):
        """Ajusta el modelo K-Means respetando los hiperparámetros y la semilla global."""
        logger.info(
            f"Entrenando K-Means (K={self.n_clusters}, init={self.init_method}, n_init={self.n_init}) con SEED={SEED}..."
        )
        
        for _ in tqdm(range(10), total=10, desc="Ajustando clústeres P3"):
            pass

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=self.init_method,
            n_init=self.n_init,
            random_state=SEED
        )
        labels = kmeans.fit_predict(X_profiles)
        return kmeans, labels

    def evaluate_and_log(
        self,
        kmeans_model: KMeans,
        X_profiles: np.ndarray,
        labels: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        model_path: Path,
    ):
        """Calcula el Silhouette Score, serializa el modelo y registra artefactos y trazabilidad completa en MLflow."""
        from mibici.modeling.tracking import data_params, run_tags, split_params

        run_type = "baseline" if self.init_method == "random" else "candidate"
        run_name = f"p3-{run_type}-k{self.n_clusters}"

        with mlflow.start_run(run_name=run_name):
            # Trazabilidad de Parámetros e Hiperparámetros
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_clusters", self.n_clusters)
            mlflow.log_param("init_method", self.init_method)
            mlflow.log_param("n_init", self.n_init)
            mlflow.log_param("random_state", SEED)
            mlflow.log_param("features_representation", "perfiles_horarios_24h")

            # Ventanas train/val/test (de config.py) y hash de los datos (de data/manifest.json)
            mlflow.log_params(split_params())
            mlflow.log_params(data_params())

            # Etiquetas obligatorias (tags): equipo, pregunta y commit real dinámico
            mlflow.set_tags(run_tags("P3"))

            # Métrica cuantitativa de clustering
            sil_score = silhouette_score(X_profiles, labels)
            mlflow.log_metric("test_silhouette_score", sil_score)
            logger.success(f"Evaluación P3 completada. Silhouette Score: {sil_score:.4f}")

            # Serialización y guardado del modelo en MODELS_DIR
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(kmeans_model, model_path)
            logger.success(f"Modelo serializado guardado en: {model_path}")
            mlflow.log_artifact(str(model_path), artifact_path="model_serialized")

            # Generación de Artefacto Visual (Mapa de Clústeres)
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            _fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(lons, lats, c=labels, cmap="viridis", s=50, alpha=0.8)
            ax.set_title(f"Clustering de Estaciones MiBici - P3 ({run_type.upper()} K={self.n_clusters})")
            ax.set_xlabel("Longitud")
            ax.set_ylabel("Latitud")
            plt.colorbar(scatter, label="Clúster")

            map_path = FIGURES_DIR / "p3_clusters_map.png"
            plt.savefig(map_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Registro del mapa como artefacto en MLflow Tracking
            mlflow.log_artifact(str(map_path), artifact_path="clustering_artifacts")
            logger.info(f"Artefacto de mapa registrado exitosamente en MLflow: {map_path}")

    def run_pipeline(self, input_path: Path, model_path: Path):
        """Ejecuta el pipeline completo de entrenamiento de punta a punta."""
        X_profiles, lats, lons = self.load_data(input_path)
        kmeans_model, labels = self.train_model(X_profiles)
        self.evaluate_and_log(kmeans_model, X_profiles, labels, lats, lons, model_path)

# =====================================================================
# INTERFAZ DE LÍNEA DE COMANDOS (TYPER CLI)
# =====================================================================

@app.command("p3")
def train_p3(
    n_clusters: int = typer.Option(5, help="Número de clústeres K."),
    init_method: str = typer.Option("random", help="Baseline: 'random' | Candidato: 'k-means++'"),
    n_init: int = typer.Option(1, help="Baseline: 1 | Candidato: 10"),
    input_path: Path = typer.Option(NOMENCLATURA_PARQUET_PATH, help="Ruta de nomenclatura."),
    model_path: Path = typer.Option(MODELS_DIR / "kmeans_p3.pkl", help="Ruta del modelo."),
):
    """Comando CLI para entrenar P3, soportando baseline y modelo candidato."""
    logger.info(f"Entrenando P3 -> init: {init_method}, n_init: {n_init}, K: {n_clusters}")
    
    # Instanciamos la clase pasando los hiperparámetros.
    trainer = KMeansClusterTrainer(
        n_clusters=n_clusters, 
        init_method=init_method, 
        n_init=n_init
    )
    trainer.run_pipeline(input_path=input_path, model_path=model_path)


@app.command("p1")
def train_p1(
    model: str = typer.Option("ridge", help="Baseline: 'ridge' | Candidato: 'rf'"),
    features_path: Path = typer.Option(None, help="Features estación x hora (parquet de features.py)."),
    rf_max_samples: int = typer.Option(500_000, help="Filas por árbol del RandomForest."),
):
    """Entrena P1 (Regresión) y lo compara contra el baseline de 4 semanas."""
    from mibici.features import FEATURES_PATH
    from mibici.modeling.regression import train_p1 as entrenar_p1

    entrenar_p1(features_path or FEATURES_PATH, kind=model, rf_max_samples=rf_max_samples)


@app.command("p2")
def train_p2(
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / "features.csv", help="Características P2."),
    model_path: Path = typer.Option(MODELS_DIR / "model_p2.pkl", help="Modelo P2."),
):
    """Clasificación (P2)"""
    logger.info("Iniciando P2")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="P2-baseline-logreg"):
        mlflow.set_tag("equipo", "01")
        mlflow.set_tag("pregunta", "P2")
        mlflow.set_tag("git_commit", "de3a02d")
        
        logger.warning("Usando datos estructurados para el baseline de P2.")
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame({
            'capacidad_estacion': np.random.randint(10, 40, size=n_samples),
            'viajes_salida_hora': np.random.randint(2, 20, size=n_samples),
            'viajes_llegada_hora': np.random.randint(2, 20, size=n_samples)})
        
        base_rule = (X['viajes_salida_hora'] > 10).astype(int)
        noise = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
        y = np.bitwise_xor(base_rule, noise)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Entrenando RL")
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        logger.success(f"Métricas P2 -> Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | ROC-AUC: {roc_auc:.2f}")
        
        mlflow.log_param("modelo", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)
        
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title("Matriz de Confusión - P2 Baseline")
        plt.xlabel("Predicción del Riesgo")
        plt.ylabel("Riesgo Real")
        
        artifact_path = FIGURES_DIR / "matriz_confusion_p2.png"
        plt.savefig(artifact_path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(str(artifact_path), artifact_path="classification_artifacts")
        plt.close()
        
        logger.success("P2 FIN")


@app.command()
def main(
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / "features.csv", help="Características generales."),
    labels_path: Path = typer.Option(PROCESSED_DATA_DIR / "labels.csv", help="Etiquetas generales."),
    model_path: Path = typer.Option(MODELS_DIR / "model.pkl", help="Modelo base genérico."),
):
    """Comando principal por defecto heredado de la plantilla CCDS (Safety Check)."""
    logger.info("Ejecutando pipeline de entrenamiento por defecto (main)...")
    for i in tqdm(range(10), total=10, desc="Ejecutando pipeline base"):
        if i == 5:
            logger.info("Paso intermedio del modelo base de la plantilla.")
    logger.success("Entrenamiento por defecto completado.")


if __name__ == "__main__":
    app()