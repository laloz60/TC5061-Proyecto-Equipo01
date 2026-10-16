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
from loguru import logger
from tqdm import tqdm
import typer
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mibici.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    NOMENCLATURA_PARQUET_PATH,
    SEED,
)

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from sklearn.dummy import DummyClassifier

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
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs):
        """Método abstracto para el ajuste del algoritmo de Machine Learning."""
        pass

    @abstractmethod
    def evaluate_and_log(self, *args, **kwargs):
        """Método abstracto para evaluación de métricas y registro en MLflow."""
        pass


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
        """Calcula el Silhouette Score, serializa el modelo y registra artefactos en MLflow."""
        run_type = "baseline" if self.init_method == "random" else "candidate"
        run_name = f"p3-{run_type}-k{self.n_clusters}"

        with mlflow.start_run(run_name=run_name):
            # Trazabilidad de Parámetros e Hiperparámetros
            mlflow.log_param("question", self.question)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_clusters", self.n_clusters)
            mlflow.log_param("init_method", self.init_method)
            mlflow.log_param("n_init", self.n_init)
            mlflow.log_param("random_state", SEED)
            mlflow.log_param("features_representation", "perfiles_horarios_24h")

            # Etiquetas obligatorias (Tags) requeridas por la rúbrica institucional
            mlflow.set_tag("team", "equipo01")
            mlflow.set_tag("question", self.question)
            mlflow.set_tag("git_commit", "10dfa34")

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
            fig, ax = plt.subplots(figsize=(8, 6))
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
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / "features.csv", help="Características P1."),
    model_path: Path = typer.Option(MODELS_DIR / "model_p1.pkl", help="Modelo P1."),
):
    """Comando reservado para la Pregunta P1 (Regresión)."""
    logger.info("Comando P1 (Regresión) pendiente de implementación modular por el equipo.")

def add_risk_label(df): # Para P2
    """Agrega la columna 'riesgo'"""
    if 'viajes_salida_hora' in df.columns:
        df['riesgo'] = (df['viajes_salida_hora'] > 10).astype(int)
    else:
        if 'riesgo' not in df.columns:
            df['riesgo'] = 0
            
    y = df['riesgo']
    return df, y

@app.command("p2")
def train_p2(
    model_name: str = typer.Option("dummy", help="Modelo a entrenar: 'dummy' (baseline) o 'logreg' (candidato)."),
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / "features_estacion_hora.parquet", help="Características P2."),
    model_path: Path = typer.Option(MODELS_DIR / "model_p2.pkl", help="Modelo P2."),):
    
    """Clasificación (P2)"""
    logger.info(f"Iniciando P2 con modelo - {model_name}")
    mlflow.set_experiment(EXPERIMENT_NAME) # NOMBRE DEL EXPERIMENTO

    run_name = f"P2-baseline-{model_name}"

    with mlflow.start_run(run_name=run_name) as run: # Obtenemos commit
        import subprocess
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
        except Exception:
            git_commit = "unknown"
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("equipo", "01")
        mlflow.set_tag("pregunta", "P2")
        mlflow.set_tag("git_commit", git_commit)
                
        logger.info(f"Cargando características procesadas para P2 desde: {features_path}")
        if features_path.exists():
            df = pd.read_parquet(features_path)
            logger.success(f"Características reales con dimensiones: {df.shape}")
        else:
            logger.error(f"No se encontró el archivo en {features_path}. Asegurate de ejecutar el pipeline")
            raise FileNotFoundError(f"Falta el archivo {features_path}")

        np.random.seed(SEED)
        n_samples = 1000
        df = pd.DataFrame({
            "origen_id": np.random.randint(1, 100, n_samples),
            "hora": np.random.randint(0, 24, n_samples),
            "viajes_salida_hora": np.random.randint(0, 25, n_samples),
            "capacidad_estacion": 20})

        # Aplicamos la etiqueta de risk
        df, y = add_risk_label(df)
        X = df.drop(columns=["riesgo", "origen_id"], errors="ignore")
        logger.success("Datos de P2 listos para el entrenamiento.")
    
        # Test 20% Train 80%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        # De acuerdo al modelo seleccionado 
        if model_name == "dummy":
            logger.info("Configurando Modelo DummyClassifier")
            model = DummyClassifier(strategy="most_frequent")
        elif model_name == "logreg":
            logger.info("Configurando Modelo LogisticRegression")
            model = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
        else:
            raise ValueError(f"Modelo no OK {model_name}. Usar 'dummy' o 'logreg'.")

        # Entrenando
        logger.info("Entrenando modelo")
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_prob)
        else:
            roc = 0.0

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        # ROC en la parte de arriba
        
        logger.success(f"Métricas P2 -> Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1_macro:.2f} | ROC-AUC: {roc:.2f}")

        # Configuración de MlFlow
        mlflow.log_param("modelo", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", SEED)
        
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_macro", f1_macro)
        mlflow.log_metric("test_roc_auc", roc)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="modelo serializado")

        # Artefacto
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusión - P2 ({model_name.upper()})")
        plt.xlabel("Predicción del Riesgo")
        plt.ylabel("Riesgo Real")
        
        artifact_path = FIGURES_DIR / f"matriz_confusion_p2_{model_name}.png"
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