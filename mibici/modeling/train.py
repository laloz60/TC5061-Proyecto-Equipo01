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
from datetime import timedelta
from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import typer

from mibici.config import (
    FIGURES_DIR,
    MANIFEST_PATH,
    MODELS_DIR,
    NOMENCLATURA_PARQUET_PATH,
    PROCESSED_DATA_DIR,
    RISK_ALPHA,
    RISK_HORIZON_HOURS,
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

    @staticmethod
    def _perfil_ventana(df_trips, estaciones, n_stations):
        """Matriz estacion x hora normalizada para las filas recibidas."""
        if len(df_trips) == 0:
            return np.zeros((n_stations, 24))

        # Construimos la matriz de frecuencia: estación vs hora del día (0 a 23)
        profile_counts = pd.crosstab(df_trips["origen_id"], df_trips["hora"])
        profile_counts = profile_counts.reindex(columns=range(24), fill_value=0)

        # Alineamos con el orden exacto de las estaciones de la nomenclatura
        if estaciones is not None:
            profile_counts = profile_counts.reindex(estaciones, fill_value=0)

        # Normalizamos por fila (proporción de viajes por hora que realiza cada estación)
        row_sums = profile_counts.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Evitamos división por cero
        return profile_counts.div(row_sums, axis=0).values

    @staticmethod
    def _silhouette_o_nan(X, labels):
        """Silhouette score; NaN si la ventana no da condiciones para calcularlo."""
        if len(X) < 2 or len(np.unique(labels)) < 2:
            return float("nan")
        return silhouette_score(X, labels)

    def load_data(self, input_path: Path):
        """Carga la nomenclatura y calcula perfiles horarios POR VENTANA (train/val/test).

        El split temporal es el de config.py: train 2023-05..2024-12,
        val 2025-01..2025-06, test 2025-07..2026-09. Así el KMeans se ajusta
        en train y la selección se evalúa en val, sin mirar test.
        """
        logger.info(f"Cargando datos espaciales desde: {input_path}")

        from mibici.config import DATA_END, DATA_START, TRAIN_END, UNIFIED_TRIPS_PATH, VAL_END

        estaciones = None
        if input_path.exists():
            df_nom = pd.read_parquet(input_path)
            n_stations = len(df_nom)
            lats = df_nom["latitud"].values
            lons = df_nom["longitud"].values
            if "estacion_id" in df_nom.columns:
                estaciones = df_nom["estacion_id"]
            logger.success(f"Nomenclatura cargada exitosamente ({n_stations} estaciones activas).")
        else:
            logger.warning(
                f"No se encontró el archivo en {input_path}. Usando datos sintéticos de respaldo."
            )
            n_stations = 300
            lats = np.random.uniform(20.60, 20.72, n_stations)
            lons = np.random.uniform(-103.42, -103.30, n_stations)

        # Intentar calcular perfiles horarios reales desde el parquet de viajes unificados del equipo
        if UNIFIED_TRIPS_PATH.exists():
            logger.info(f"Calculando perfiles horarios reales desde: {UNIFIED_TRIPS_PATH}")
            # Leemos únicamente las columnas necesarias para optimizar memoria
            df_trips = pd.read_parquet(UNIFIED_TRIPS_PATH, columns=["inicio_viaje", "origen_id"])
            df_trips["inicio_viaje"] = pd.to_datetime(df_trips["inicio_viaje"])
            df_trips["hora"] = df_trips["inicio_viaje"].dt.hour

            ventanas = {
                "train": (DATA_START, TRAIN_END),
                "val": (TRAIN_END + timedelta(days=1), VAL_END),
                "test": (VAL_END + timedelta(days=1), DATA_END),
            }
            perfiles = {}
            for nombre, (inicio, fin) in ventanas.items():
                mask = (df_trips["inicio_viaje"] >= pd.Timestamp(inicio)) & (
                    df_trips["inicio_viaje"] < pd.Timestamp(fin) + pd.Timedelta(days=1)
                )
                perfiles[nombre] = self._perfil_ventana(df_trips[mask], estaciones, n_stations)
            logger.success(
                "Perfiles horarios calculados por ventana: train/val/test (split temporal)."
            )
        else:
            logger.warning(
                "No se encontró el archivo de viajes unificados. Usando datos sintéticos para perfiles."
            )
            np.random.seed(SEED)
            sintetico = np.random.rand(n_stations, 24)
            perfiles = {nombre: sintetico for nombre in ("train", "val", "test")}

        return perfiles, lats, lons

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
            random_state=SEED,
        )
        labels = kmeans.fit_predict(X_profiles)
        return kmeans, labels

    def evaluate_and_log(
        self,
        kmeans_model: KMeans,
        perfiles: dict,
        labels: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        model_path: Path,
    ):
        """Silhouette en VAL para la selección, en TEST solo para el reporte.

        El modelo ya viene ajustado con los perfiles de train; aquí se predice
        sobre las ventanas de val y test para medir sin fuga de información.
        """

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

            # Métricas de clustering: VAL decide, TEST solo se reporta
            labels_val = kmeans_model.predict(perfiles["val"])
            labels_test = kmeans_model.predict(perfiles["test"])
            sil_val = self._silhouette_o_nan(perfiles["val"], labels_val)
            sil_test = self._silhouette_o_nan(perfiles["test"], labels_test)
            mlflow.log_metric("val_silhouette_score", sil_val)
            mlflow.log_metric("test_silhouette_score", sil_test)
            logger.success(
                f"Evaluación P3 completada. Silhouette val={sil_val:.4f} test={sil_test:.4f}"
            )

            signature = infer_signature(perfiles["train"], labels)
            mlflow.sklearn.log_model(
                kmeans_model, 
                artifact_path="model", 
                signature=signature, 
                input_example=perfiles["train"][:5]
            )

            if MANIFEST_PATH.exists():
                dataset_source = mlflow.data.from_pandas(
                    pd.DataFrame(perfiles["train"]), 
                    source=str(MANIFEST_PATH), 
                    name="mibici_p3_profiles"
                )
                mlflow.log_input(dataset_source, context="clustering_training")

            # Serialización y guardado del modelo en MODELS_DIR
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(kmeans_model, model_path)
            logger.success(f"Modelo serializado guardado en: {model_path}")
            mlflow.log_artifact(str(model_path), artifact_path="model_serialized")
            mlflow.sklearn.log_model(
                sk_model=kmeans_model,
                name="model",
                signature=infer_signature(perfiles["train"], labels),
                input_example=perfiles["train"][:5],
            )

            # Generación de Artefacto Visual (Mapa de Clústeres)
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            _fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(lons, lats, c=labels, cmap="viridis", s=50, alpha=0.8)
            ax.set_title(
                f"Clustering de Estaciones MiBici - P3 ({run_type.upper()} K={self.n_clusters})"
            )
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
        perfiles, lats, lons = self.load_data(input_path)
        # el ajuste es SOLO con la ventana de train
        kmeans_model, labels = self.train_model(perfiles["train"])
        self.evaluate_and_log(kmeans_model, perfiles, labels, lats, lons, model_path)


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
    trainer = KMeansClusterTrainer(n_clusters=n_clusters, init_method=init_method, n_init=n_init)
    trainer.run_pipeline(input_path=input_path, model_path=model_path)

@app.command("p3-search")
def train_p3_search(
    n_clusters_list: str = typer.Option("3,4,5,6", help="Valores de K separados por comas"),
    input_path: Path = typer.Option(NOMENCLATURA_PARQUET_PATH, help="Ruta de nomenclatura."),):
    """P3 usando Nested Runs y artefactos MLflow"""

    import json

    import matplotlib.pyplot as plt

    from mibici.config import FIGURES_DIR, MODELS_DIR
    from mibici.modeling.tracking import EXPERIMENT_NAME, data_params, run_tags, split_params
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # TRAINER
    trainer = KMeansClusterTrainer(n_clusters=5, init_method="k-means++", n_init=10)
    perfiles, lats, lons = trainer.load_data(input_path)
    
    # CORRIDA PADRE
    with mlflow.start_run(run_name="p3-parent-search-kmeans"):
        mlflow.set_tags(run_tags("P3"))
        mlflow.set_tag("stage_of_work", "search")
        mlflow.set_tag("model_family", "kmeans")
        mlflow.log_params(split_params())
        mlflow.log_params(data_params())
        
        k_values = [int(k.strip()) for k in n_clusters_list.split(",")]
        
        # CORRIDAS HIJAS ITERADAS
        for k in k_values:
            with mlflow.start_run(run_name=f"p3-candidate-k{k}", nested=True):
                # ACTUAL CANDIDADTO: se ajusta en train y se evalúa en val
                kmeans_model, labels = KMeansClusterTrainer(
                    n_clusters=k, init_method="k-means++", n_init=10
                ).train_model(perfiles["train"])
                
                labels_val = kmeans_model.predict(perfiles["val"])
                sil_val = KMeansClusterTrainer._silhouette_o_nan(perfiles["val"], labels_val)
                sil_test = KMeansClusterTrainer._silhouette_o_nan(
                    perfiles["test"], kmeans_model.predict(perfiles["test"])
                )
                
                # Registramos parámetros y métricas
                mlflow.log_param("model_type", "KMeans")
                mlflow.log_param("n_clusters", k)
                mlflow.log_param("init_method", "k-means++")
                mlflow.log_param("n_init", 10)
                mlflow.log_metric("val_silhouette_score", sil_val)
                mlflow.log_metric("test_silhouette_score", sil_test)
                
                # Etiquetas obligatorias
                mlflow.set_tags(run_tags("P3"))
                mlflow.set_tag("stage_of_work", "candidate")
                mlflow.set_tag("model_family", "kmeans")

                # Firma
                signature = infer_signature(perfiles["train"], labels)
                input_example = perfiles["train"][:5]

                # Carpeta de datos
                signature_dir = MODELS_DIR / f"signature_metadata_k{k}"
                signature_dir.mkdir(parents=True, exist_ok=True)

                # Guardado en JSON ENTRADA Y SALIDA
                sig_path = signature_dir / "signature.json"
                with open(sig_path, "w", encoding="utf-8") as f:
                    json.dump(signature.to_dict(), f, indent=4)

                ex_path = signature_dir / "input_example.json"
                with open(ex_path, "w", encoding="utf-8") as f:
                    json.dump(input_example.tolist(), f)

                mlflow.log_artifacts(str(signature_dir), artifact_path="model_signature")

                mlflow.sklearn.log_model(sk_model=kmeans_model, name="model",
                    signature=signature, input_example=input_example)
                
                # FIGURAS
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                
                # 1. Mapa
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(lons, lats, c=labels, cmap="viridis", s=50, alpha=0.8)
                ax.set_title(f"Mapa Geográfico - P3 (K={k})")
                ax.set_xlabel("Longitud")
                ax.set_ylabel("Latitud")
                plt.colorbar(scatter, label="Clúster")
                map_path = FIGURES_DIR / f"p3_mapa_geografico_k{k}.png"
                plt.savefig(map_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # 2. Distribución de estaciones
                fig, ax = plt.subplots(figsize=(8, 5))
                unique_clusters, counts = np.unique(labels, return_counts=True)
                ax.bar(unique_clusters, counts, color="skyblue", edgecolor="black")
                ax.set_title(f"Distribución de Estaciones por Clúster - P3 (K={k})")
                ax.set_xlabel("Clúster")
                ax.set_ylabel("Número de Estaciones")
                dist_path = FIGURES_DIR / f"p3_distribucion_clusters_k{k}.png"
                plt.savefig(dist_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # 3. Clústeres combinados
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(lons, lats, c=labels, cmap="tab10", s=40, alpha=0.7)
                ax.set_title(f"Clustering de Estaciones MiBici - P3 (K={k})")
                ax.set_xlabel("Longitud")
                ax.set_ylabel("Latitud")
                clusters_path = FIGURES_DIR / f"p3_clusters_map_k{k}.png"
                plt.savefig(clusters_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                
                mlflow.log_artifact(str(map_path), artifact_path="clustering_artifacts")
                mlflow.log_artifact(str(dist_path), artifact_path="clustering_artifacts")
                mlflow.log_artifact(str(clusters_path), artifact_path="clustering_artifacts")
                
                logger.success(
                    f"Nested run y artefactos guardados para K={k} "
                    f"con Silhouette val={sil_val:.4f} test={sil_test:.4f}"
                )

    # Nuevo modelo "Gaussian Mixture Model (GMM)"
    with mlflow.start_run(run_name="p3-parent-search-gmm"):
        mlflow.set_tags(run_tags("P3"))
        mlflow.set_tag("stage_of_work", "search")
        mlflow.set_tag("model_family", "gaussian_mixture")
        mlflow.log_params(split_params())
        mlflow.log_params(data_params())
        
        from sklearn.mixture import GaussianMixture
        for k in k_values:
            with mlflow.start_run(run_name=f"p3-candidate-gmm-k{k}", nested=True):
                gmm = GaussianMixture(n_components=k, random_state=SEED, n_init=5)
                # se ajusta en train; val decide y test solo se reporta
                gmm.fit(perfiles["train"])
                sil_val_gmm = KMeansClusterTrainer._silhouette_o_nan(
                    perfiles["val"], gmm.predict(perfiles["val"])
                )
                sil_test_gmm = KMeansClusterTrainer._silhouette_o_nan(
                    perfiles["test"], gmm.predict(perfiles["test"])
                )
                
                mlflow.log_param("model_type", "GaussianMixture")
                mlflow.log_param("n_clusters", k)
                mlflow.log_metric("val_silhouette_score", sil_val_gmm)
                mlflow.log_metric("test_silhouette_score", sil_test_gmm)
                
                tags_gmm = run_tags("P3")
                tags_gmm.update({"stage_of_work": "candidate", "model_family": "gaussian_mixture"})
                mlflow.set_tags(tags_gmm)

@app.command("p1")
def train_p1(
    model: str = typer.Option("regla", help="Baseline: 'regla' | Candidatos: 'ridge', 'rf'"),
    features_path: Path = typer.Option(
        None, help="Features estación x hora (parquet de features.py)."
    ),
    rf_max_samples: int = typer.Option(500_000, help="Filas por árbol del RandomForest."),
    n_estimators: int = typer.Option(100, help="RandomForest: número de árboles (rf)."),
):
    """Entrena P1 (Regresión) y lo compara contra el baseline de 4 semanas."""
    from mibici.features import FEATURES_PATH
    from mibici.modeling.regression import train_p1 as entrenar_p1

    entrenar_p1(
        features_path or FEATURES_PATH,
        kind=model,
        rf_max_samples=rf_max_samples,
        n_estimators=n_estimators,
    )

@app.command("p1-search")
def train_p1_search(
    n_estimators_list: str = typer.Option("50,100,200", help="Valores de n_estimators separados por comas"),
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / "features_estacion_hora.parquet", help="Features estación x hora."),):

    """P1"""
    import json
    import warnings

    from mibici.config import FIGURES_DIR, MODELS_DIR
    from mibici.modeling.regression import (FEATURES, TARGET,
        build_model, load_p1_data, regression_metrics,)
    from mibici.modeling.tracking import EXPERIMENT_NAME, data_params, run_tags, split_params

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel") # Ignorar la advertencia
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    train, val, _ = load_p1_data(features_path) # Cargar train y val
    
    df_full = pd.read_parquet(features_path)  # Cargar test
    test_df = df_full[df_full["split"] == "test"].copy()

    test_df[FEATURES] = test_df[FEATURES].fillna(0) # Rellena NAS
    
    y_real = test_df[TARGET].to_numpy()
    y_val = val[TARGET].to_numpy()
    
    print(f"\nTest shape: {test_df.shape} | y_real min: {y_real.min()}, max: {y_real.max()}, mean: {y_real.mean():.2f}\n")

    estimators_values = [int(n.strip()) for n in n_estimators_list.split(",")]

    # Padre
    with mlflow.start_run(run_name="p1-parent-search-rf"):
        mlflow.set_tags(run_tags("P1"))
        mlflow.set_tag("stage_of_work", "search")
        mlflow.set_tag("model_family", "random_forest")
        mlflow.log_params(split_params())
        mlflow.log_params(data_params())
        
        # Hijas iteradas
        for n_est in estimators_values:
            with mlflow.start_run(run_name=f"p1-candidate-rf-n{n_est}", nested=True):
                
                model = build_model("rf", n_estimators=n_est)
                logger.info(f"Entrenando candidato RF con n_estimators={n_est}...")
                model.fit(train[FEATURES], train[TARGET])
                
                # seleccion en VALIDACION: el test queda para el reporte final
                y_pred_val = np.clip(model.predict(val[FEATURES]), 0, None)
                val_metrics = regression_metrics(y_val, y_pred_val)

                y_pred = model.predict(test_df[FEATURES])
                y_pred = np.clip(y_pred, 0, None)
                
                metrics = regression_metrics(y_real, y_pred)
                
                mlflow.log_param("model_type", "RandomForestRegressor")
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", 16)
                mlflow.log_param("min_samples_leaf", 20)
                mlflow.log_metric("val_mae", val_metrics["mae"])
                mlflow.log_metric("val_rmse", val_metrics["rmse"])
                mlflow.log_metric("val_r2", val_metrics["r2"])
                mlflow.log_metric("test_mae", metrics["mae"])
                mlflow.log_metric("test_rmse", metrics["rmse"])
                mlflow.log_metric("test_r2", metrics["r2"])
                
                mlflow.set_tags(run_tags("P1"))
                mlflow.set_tag("stage_of_work", "candidate")
                mlflow.set_tag("model_family", "random_forest")

                signature = infer_signature(train[FEATURES], y_pred[:len(train[FEATURES])])
                input_example = train[FEATURES].head(5)

                # SUBCARPETAS
                temp_dir = Path(f"temp_artifacts_n{n_est}")
                temp_plots = temp_dir / "plots"
                temp_data = temp_dir / "data"
                temp_config = temp_dir / "config"
                
                temp_plots.mkdir(parents=True, exist_ok=True)
                temp_data.mkdir(parents=True, exist_ok=True)
                temp_config.mkdir(parents=True, exist_ok=True)
                
                # Firma entrada/salida
                signature = infer_signature(train[FEATURES], y_pred[:len(train[FEATURES])])
                input_example = train[FEATURES].head(5)

                mlflow.sklearn.log_model(
                    sk_model=model, 
                    artifact_path="model", 
                    signature=signature,
                    input_example=input_example, 
                    skops_trusted_types=["sklearn.tree._tree.Tree"])

                # Data/ JSON
                if MANIFEST_PATH.exists():
                    df_mlflow_input = train[FEATURES].copy()
                    df_mlflow_input[TARGET] = train[TARGET]
                    
                    dataset_source = mlflow.data.from_pandas(
                        df_mlflow_input, 
                        source=str(MANIFEST_PATH), 
                        name="mibici_p1_train", 
                        targets=TARGET) 
                    mlflow.log_input(dataset_source, context="training")
                    
                    manifiesto_data = {
                        "version_datos": "v2.0-entregable2", 
                        "fuente_origen": str(MANIFEST_PATH),
                        "registros_entrenamiento": len(train), 
                        "columnas_features": FEATURES, 
                        "anonimizado": True}
                    with open(temp_data / "manifiesto_ingesta.json", "w", encoding="utf-8") as f:
                        json.dump(manifiesto_data, f, indent=4)
                    
                    mlflow.log_artifacts(str(temp_data), artifact_path="data")

                # Config/
                config_corrida = {
                    "n_estimators": n_est, 
                    "max_depth": 16,
                    "min_samples_leaf": 20, 
                    "global_seed": SEED,
                    "features_count": len(FEATURES)}
                with open(temp_config / "config_ejecucion.json", "w", encoding="utf-8") as f:
                    json.dump(config_corrida, f, indent=4)
                
                mlflow.log_artifacts(str(temp_config), artifact_path="config")

                # Graficas plots/
                np.random.seed(SEED)
                idx = np.random.choice(len(y_real), size=min(50_000, len(y_real)), replace=False)
                y_r_sample = y_real[idx]
                y_p_sample = y_pred[idx]
                residuos = y_r_sample - y_p_sample
                
                limite = float(np.percentile(y_real, 99.9)) if len(y_real) > 0 and np.max(y_real) > 0 else 25.0
                limite = max(limite, 5.0)

                # Gráfica Real vs Pred
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                ax1.scatter(y_r_sample, y_p_sample, s=4, alpha=0.2, color="tab:blue")
                ax1.plot([0, limite], [0, limite], color="black", linestyle="--")
                ax1.set_xlim(0, limite)
                ax1.set_ylim(0, limite)
                ax1.set_xlabel("Viajes reales por hora")
                ax1.set_ylabel("Viajes predichos por hora")
                ax1.set_title(f"Predicción vs Real (n_estimators={n_est})")
                fig1.savefig(temp_plots / "real_vs_pred.png", dpi=150, bbox_inches="tight")
                plt.close(fig1)

                # Gráfica Residuos
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                res_lim = float(np.percentile(np.abs(residuos), 99)) if len(residuos) > 0 else 1.0
                res_lim = max(res_lim, 1.0)
                ax2.hist(residuos, bins=80, range=(-res_lim, res_lim), color="tab:orange", edgecolor="black", linewidth=0.5)
                ax2.axvline(0, color="black", linestyle="--")
                ax2.set_xlabel("Residuo (real - predicho)")
                ax2.set_ylabel("Frecuencia")
                ax2.set_title(f"Distribución de Residuos (n_estimators={n_est})")
                fig2.savefig(temp_plots / "residuos.png", dpi=150, bbox_inches="tight")
                plt.close(fig2)
                
                mlflow.log_artifacts(str(temp_plots), artifact_path="plots")

                # Firma en json dentro de model_signature
                signature_dir = MODELS_DIR / f"signature_metadata_rf_n{n_est}"
                signature_dir.mkdir(parents=True, exist_ok=True)
                with open(signature_dir / "signature.json", "w", encoding="utf-8") as f:
                    json.dump(signature.to_dict(), f, indent=4)
                with open(signature_dir / "input_example.json", "w", encoding="utf-8") as f:
                    json.dump(input_example.values.tolist(), f)
                mlflow.log_artifacts(str(signature_dir), artifact_path="model_signature")

                import shutil
                # Estructura
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                logger.success(f"Nested run - n_estimators={n_est} - Test MAE: {metrics['mae']:.3f}")

                # Graficos juntos se puede descomentar en caso de requerir

                # FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                # fig_path = FIGURES_DIR / f"p1_rf_n{n_est}_pred_vs_real.png"
                
                # np.random.seed(SEED)
                # idx = np.random.choice(len(y_real), size=min(50_000, len(y_real)), replace=False)
                # y_r_sample = y_real[idx]
                # y_p_sample = y_pred[idx]
                # residuos = y_r_sample - y_p_sample
                
                # limite = float(np.percentile(y_real, 99.9)) if len(y_real) > 0 and np.max(y_real) > 0 else 25.0
                # limite = max(limite, 5.0)

                # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # ax1.scatter(y_r_sample, y_p_sample, s=4, alpha=0.2, color="tab:blue")
                # ax1.plot([0, limite], [0, limite], color="black", linestyle="--")
                # ax1.set_xlim(0, limite)
                # ax1.set_ylim(0, limite)
                # ax1.set_xlabel("Viajes reales por hora")
                # ax1.set_ylabel("Viajes predichos por hora")
                # ax1.set_title("Prediccion vs real (test)")

                # res_lim = float(np.percentile(np.abs(residuos), 99)) if len(residuos) > 0 else 1.0
                # res_lim = max(res_lim, 1.0)
                
                # ax2.hist(residuos, bins=80, range=(-res_lim, res_lim), color="tab:blue", edgecolor="black", linewidth=0.5)
                # ax2.axvline(0, color="black", linestyle="--")
                # ax2.set_xlabel("Residuo (real - predicho)")
                # ax2.set_ylabel("Frecuencia")
                # ax2.set_title("Residuos (test)")

                # fig.suptitle(f"P1 - RF (n_estimators={n_est})")
                # fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                # plt.close(fig)
                
                # mlflow.log_artifact(str(fig_path), artifact_path="figures")
               # logger.success(f"Nested run - n_estimators={n_est} - Test MAE: {metrics['mae']:.3f}")
    
    # Modelo de "Gradient Boosting"
    with mlflow.start_run(run_name="p1-parent-search-gbr"):
        mlflow.set_tags(run_tags("P1"))
        mlflow.set_tag("stage_of_work", "search")
        mlflow.set_tag("model_family", "gradient_boosting")
        mlflow.log_params(split_params())
        mlflow.log_params(data_params())
        
        for n_est in estimators_values:
            with mlflow.start_run(run_name=f"p1-candidate-gbr-n{n_est}", nested=True):
                model_gbr = GradientBoostingRegressor(n_estimators=n_est, max_depth=5, random_state=SEED)
                logger.info(f"Entrenando candidato GBR con n_estimators={n_est}...")
                model_gbr.fit(train[FEATURES], train[TARGET])
                
                y_pred_gbr = np.clip(model_gbr.predict(test_df[FEATURES]), 0, None)
                metrics_gbr = regression_metrics(y_real, y_pred_gbr)
                
                mlflow.log_param("model_type", "GradientBoostingRegressor")
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_metric("test_mae", metrics_gbr["mae"])
                mlflow.log_metric("test_rmse", metrics_gbr["rmse"])
                mlflow.log_metric("test_r2", metrics_gbr["r2"])
                
                tags_gbr = run_tags("P1")
                tags_gbr.update({"stage_of_work": "candidate", "model_family": "gradient_boosting"})
                mlflow.set_tags(tags_gbr)

@app.command("p2")
def train_p2(
    model_name: str = typer.Option(
        "dummy", help="Modelo a entrenar: 'dummy' (baseline) o 'logreg' (candidato)."
    ),
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features_estacion_hora.parquet",
        help="Features estacion x hora (se generan con python -m mibici.features).",
    ),
    model_path: Path = typer.Option(MODELS_DIR / "model_p2.pkl", help="Modelo P2."),
):
    """Clasificacion (P2): la estacion quedara vacia o saturada en las proximas horas?

    La etiqueta es el proxy de riesgo de features.add_risk_label
    (ver references/proxy_riesgo_p2.md). Se usa el split temporal de config.py:
    se entrena con train (2023-05..2024-12), se valida con val (2025-01..2025-06)
    y se evalua con test (2025-07..2026-09).
    """
    from mibici.config import MANIFEST_PATH
    from mibici.features import add_risk_label
    from mibici.modeling.regression import FEATURES
    from mibici.modeling.tracking import data_params, run_tags, split_params

    logger.info(f"Iniciando P2 con modelo - {model_name}")
    if not features_path.exists():
        raise FileNotFoundError(f"Falta {features_path}. Corre antes: python -m mibici.features")

    columnas = ["estacion_id", "salidas", "llegadas", "split"] + FEATURES
    df = pd.read_parquet(features_path, columns=columnas)
    
    # La etiqueta va antes del dropna: necesita las 3 horas siguientes de cada estacion
    df = add_risk_label(df)
    df = df.dropna(subset=FEATURES + ["riesgo"])

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    
    X_train, y_train = train[FEATURES], train["riesgo"].astype(int)
    X_val, y_val = val[FEATURES], val["riesgo"].astype(int)
    X_test, y_test = test[FEATURES], test["riesgo"].astype(int)

    logger.info(
        f"P2 train={len(train):,} filas ({y_train.mean():.2%} en riesgo) | "
        f"val={len(val):,} filas ({y_val.mean():.2%} en riesgo) | "
        f"test={len(test):,} filas ({y_test.mean():.2%} en riesgo)"
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    if model_name == "dummy":
        model = DummyClassifier(strategy="most_frequent")
        run_name = "P2-baseline-dummy"
        with mlflow.start_run(run_name=run_name):
            tags = run_tags("P2")
            tags.update({"model_family": "dummy", "stage_of_work": "baseline"})
            mlflow.set_tags(tags)
            mlflow.log_params(split_params())
            mlflow.log_params(data_params())
            mlflow.log_params({
                "modelo": model_name,
                "random_state": SEED,
                "risk_horizon_hours": RISK_HORIZON_HOURS,
                "risk_alpha": RISK_ALPHA,
            })
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_val = model.predict(X_val)
            
            metricas = {
                "val_precision": precision_score(y_val, y_pred_val, zero_division=0),
                "val_recall": recall_score(y_val, y_pred_val, zero_division=0),
                "val_f1_macro": f1_score(y_val, y_pred_val, average="macro", zero_division=0),
                "val_roc_auc": 0.5,
                "test_precision": precision_score(y_test, y_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_pred, zero_division=0),
                "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "test_roc_auc": 0.5,
            }
            mlflow.log_metrics(metricas)
            
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=X_test.head(5))
            if MANIFEST_PATH.exists():
                ds = mlflow.data.from_pandas(X_train, source=str(MANIFEST_PATH), name="mibici_p2_train")
                mlflow.log_input(ds, context="training")

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

            # Matriz de confusión para dummy
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(6, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
            plt.title("Matriz de Confusion - P2 (DUMMY)")
            plt.xlabel("Prediccion del Riesgo")
            plt.ylabel("Riesgo Real")
            artifact_path = FIGURES_DIR / "matriz_confusion_p2_dummy.png"
            plt.savefig(artifact_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(str(artifact_path), artifact_path="classification_artifacts")
            plt.close()

    elif model_name == "logreg":
        # Búsqueda organizada como Nested Runs con selección en VALIDACIÓN (Evita fuga en test)
        with mlflow.start_run(run_name="P2-parent-search-logreg"):
            parent_tags = run_tags("P2")
            parent_tags.update({"model_family": "logreg", "stage_of_work": "search"})
            mlflow.set_tags(parent_tags)
            mlflow.log_params(split_params())
            mlflow.log_params(data_params())
            mlflow.log_params({
                "modelo": model_name,
                "random_state": SEED,
                "risk_horizon_hours": RISK_HORIZON_HOURS,
                "risk_alpha": RISK_ALPHA,
            })

            C_values = [0.1, 1.0, 10.0]
            mejor_f1_val = -1
            mejor_C = None
            mejor_umbral_global = 0.5 # Guardará el umbral óptimo

            for c_val in C_values:
                with mlflow.start_run(run_name=f"p2-child-logreg-C{c_val}", nested=True):
                    child_tags = run_tags("P2")
                    child_tags.update({"model_family": "logreg", "stage_of_work": "candidate"})
                    mlflow.set_tags(child_tags)
                    
                    sub_model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=c_val, class_weight="balanced", random_state=SEED, max_iter=1000),
                    )
                    sub_model.fit(X_train, y_train)
                    y_val_prob = sub_model.predict_proba(X_val)[:, 1]
                    
                    # Buscar el mejor umbral en Validación
                    f1_max_umbral = 0
                    mejor_umbral = 0.5
                    for umbral in np.linspace(0.3, 0.7, 41): # Prueba 41 umbrales entre 0.3 y 0.7
                        y_val_pred_u = (y_val_prob >= umbral).astype(int)
                        f1_u = f1_score(y_val, y_val_pred_u, average="macro", zero_division=0)
                        if f1_u > f1_max_umbral:
                            f1_max_umbral = f1_u
                            mejor_umbral = umbral
                    
                    # Evaluar con el mejor umbral encontrado
                    y_val_pred_opt = (y_val_prob >= mejor_umbral).astype(int)
                    val_metrics = {
                        "val_precision": precision_score(y_val, y_val_pred_opt, zero_division=0),
                        "val_recall": recall_score(y_val, y_val_pred_opt, zero_division=0),
                        "val_f1_macro": f1_max_umbral,
                        "val_roc_auc": roc_auc_score(y_val, y_val_prob),
                    }
                    mlflow.log_param("C", c_val)
                    mlflow.log_param("umbral_optimo_val", round(mejor_umbral, 3))
                    mlflow.log_metrics(val_metrics)

                    if val_metrics["val_f1_macro"] > mejor_f1_val:
                        mejor_f1_val = val_metrics["val_f1_macro"]
                        mejor_C = c_val
                        mejor_umbral_global = mejor_umbral

            # Entrenar modelo final y evaluar en TEST con el umbral óptimo
            logger.info(f"Mejor C (Val): {mejor_C} | Mejor Umbral (Val): {mejor_umbral_global:.3f}")
            mejor_modelo = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=mejor_C, class_weight="balanced", random_state=SEED, max_iter=1000),
            )
            mejor_modelo.fit(X_train, y_train)
            
            y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= mejor_umbral_global).astype(int) # Predicción con umbral ajustado

            # val: con el mejor C y umbral ya elegidos, se reporta en val para la seleccion
            y_val_prob = mejor_modelo.predict_proba(X_val)[:, 1]
            y_val_pred = (y_val_prob >= mejor_umbral_global).astype(int)
            val_metrics = {
                "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                "val_f1_macro": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
                "val_roc_auc": roc_auc_score(y_val, y_val_prob),
            }

            test_metrics = {
                "test_precision": precision_score(y_test, y_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_pred, zero_division=0),
                "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, y_prob),
            }
            mlflow.log_param("best_C", mejor_C)
            mlflow.log_param("umbral_final", round(mejor_umbral_global, 3))
            mlflow.log_metrics(val_metrics)
            mlflow.log_metrics(test_metrics)

            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(mejor_modelo, artifact_path="model", signature=signature, input_example=X_test.head(5))
            
            if MANIFEST_PATH.exists():
                ds = mlflow.data.from_pandas(X_train, source=str(MANIFEST_PATH), name="mibici_p2_train")
                mlflow.log_input(ds, context="training")

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(mejor_modelo, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")
            
            # Artefactos
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(6, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
            plt.title(f"Matriz de Confusion - P2 (LOGREG Umbral {mejor_umbral_global:.2f})")
            plt.xlabel("Prediccion del Riesgo")
            plt.ylabel("Riesgo Real")
            artifact_path = FIGURES_DIR / "matriz_confusion_p2_logreg.png"
            plt.savefig(artifact_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(str(artifact_path), artifact_path="classification_artifacts")
            plt.close()

            # Curva PR con el umbral marcado
            fig, ax = plt.subplots(figsize=(6, 5))
            PrecisionRecallDisplay.from_predictions(
                y_test, y_prob, name="LogReg", plot_chance_level=True, drop_intermediate=True, ax=ax
            )
            # Dibujar el punto exacto de nuestro umbral óptimo (ya calculado en y_pred)
            precision_opt = precision_score(y_test, y_pred, zero_division=0)
            recall_opt = recall_score(y_test, y_pred, zero_division=0)
            
            ax.plot(recall_opt, precision_opt, 'ro', markersize=8, label=f'Umbral {mejor_umbral_global:.2f} (Val)')
            ax.legend()
                
            ax.set_title("Curva Precision-Recall - P2 (LOGREG)")
            pr_path = FIGURES_DIR / "curva_pr_p2_logreg.png"
            fig.savefig(pr_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(str(pr_path), artifact_path="classification_artifacts")
            plt.close(fig)
            mlflow.log_metric("test_average_precision", average_precision_score(y_test, y_prob))

            logger.success(f"Búsqueda nested runs P2 finalizada. Mejor F1 Macro (Val): {mejor_f1_val:.3f}")
    # Nuevo modelo elgido "rf"
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        with mlflow.start_run(run_name="P2-parent-search-rf"):
            parent_tags = run_tags("P2")
            parent_tags.update({"model_family": "random_forest", "stage_of_work": "search"})
            mlflow.set_tags(parent_tags)
            mlflow.log_params(split_params())
            mlflow.log_params(data_params())
            mlflow.log_params({
                "modelo": model_name,
                "random_state": SEED,
                "risk_horizon_hours": RISK_HORIZON_HOURS,
                "risk_alpha": RISK_ALPHA,
            })

            n_estimators_list = [50, 100, 200]
            mejor_f1_val = -1
            mejor_n = None

            for n_est in n_estimators_list:
                with mlflow.start_run(run_name=f"p2-child-rf-n{n_est}", nested=True):
                    child_tags = run_tags("P2")
                    child_tags.update({"model_family": "random_forest", "stage_of_work": "candidate"})
                    mlflow.set_tags(child_tags)
                    
                    sub_model = RandomForestClassifier(n_estimators=n_est, class_weight="balanced", random_state=SEED, max_depth=10)
                    sub_model.fit(X_train, y_train)
                    
                    y_val_pred = sub_model.predict(X_val)
                    y_val_prob = sub_model.predict_proba(X_val)[:, 1]
                    
                    val_metrics = {
                        "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                        "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                        "val_f1_macro": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
                        "val_roc_auc": roc_auc_score(y_val, y_val_prob),
                    }
                    mlflow.log_param("n_estimators", n_est)
                    mlflow.log_metrics(val_metrics)

                    if val_metrics["val_f1_macro"] > mejor_f1_val:
                        mejor_f1_val = val_metrics["val_f1_macro"]
                        mejor_n = n_est

            logger.info(f"Mejor n_estimators encontrado en validación: {mejor_n}")
            mejor_modelo = RandomForestClassifier(n_estimators=mejor_n, class_weight="balanced", random_state=SEED, max_depth=10)
            mejor_modelo.fit(X_train, y_train)
            y_pred = mejor_modelo.predict(X_test)
            y_prob = mejor_modelo.predict_proba(X_test)[:, 1]

            # val: con el mejor n_estimators ya elegido, se reporta en val para la seleccion
            y_val_pred = mejor_modelo.predict(X_val)
            y_val_prob = mejor_modelo.predict_proba(X_val)[:, 1]
            val_metrics = {
                "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                "val_f1_macro": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
                "val_roc_auc": roc_auc_score(y_val, y_val_prob),
            }

            test_metrics = {
                "test_precision": precision_score(y_test, y_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_pred, zero_division=0),
                "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, y_prob),
            }
            mlflow.log_param("best_n_estimators", mejor_n)
            mlflow.log_metrics(val_metrics)
            mlflow.log_metrics(test_metrics)

            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(mejor_modelo, artifact_path="model", signature=signature, input_example=X_test.head(5))
            logger.success(f"Búsqueda nested runs P2 RF finalizada. Mejor F1 Macro en val: {mejor_f1_val:.3f}")
    else:
        raise ValueError(f"Modelo no valido: {model_name}. Usar 'dummy' o 'logreg'.")

    logger.success("P2 FIN")


@app.command()
def main(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features.csv", help="Características generales."
    ),
    labels_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "labels.csv", help="Etiquetas generales."
    ),
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
