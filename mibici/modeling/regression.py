"""P1: viajes que se iniciaran en cada estacion en la siguiente hora.

Target: `salidas` = viajes con origen en la estacion que empiezan en la hora h
(hora local de Guadalajara). Se predice con features calculadas hasta la hora h-1.
Se quitan las filas sin historia completa (primeras 4 semanas de cada estacion).

Pipeline: ingesta (dataset.py) -> estacion x hora y features (features.py)
-> entrenamiento en train -> evaluacion en val y test -> MLflow.

Metricas: MAE (error en viajes por hora), RMSE (castiga mas los errores en
horas pico) y R2 (varianza explicada).
"""

from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mibici.config import FIGURES_DIR, MODELS_DIR, SEED
from mibici.modeling.tracking import EXPERIMENT_NAME, data_params, run_tags, split_params

TARGET = "salidas"
BASELINE = "salidas_prom_4sem"

CATEGORICAS = ["hora_del_dia", "dia_semana", "mes"]
NUMERICAS = [
    "es_fin_de_semana",
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
]
FEATURES = CATEGORICAS + NUMERICAS


def load_p1_data(features_path):
    columnas = ["estacion_id", "hora", TARGET] + FEATURES + ["split"]
    df = pd.read_parquet(features_path, columns=columnas)
    df = df.dropna(subset=FEATURES)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    return train, val, test


def build_model(kind, rf_max_samples=500_000, n_estimators=100):
    if kind == "ridge":
        # la hora y el dia van en one-hot porque su efecto no es lineal
        preprocesamiento = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                ("num", StandardScaler(), NUMERICAS),
            ]
        )
        return Pipeline([("pre", preprocesamiento), ("model", Ridge(alpha=1.0))])

    if kind == "rf":
        # max_samples limita las filas por arbol para que no tarde tanto
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=16,
            min_samples_leaf=20,
            max_samples=rf_max_samples,
            n_jobs=-1,
            random_state=SEED,
        )
        return Pipeline([("model", rf)])

    raise ValueError(f"Modelo no soportado: {kind}")


def regression_metrics(y_real, y_pred):
    return {
        "mae": mean_absolute_error(y_real, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_real, y_pred)),
        "r2": r2_score(y_real, y_pred),
    }


def plot_predictions(y_real, y_pred, titulo, path):
    # se grafica una muestra porque test tiene millones de filas
    np.random.seed(SEED)
    idx = np.random.choice(len(y_real), size=min(50_000, len(y_real)), replace=False)
    y_real = y_real[idx]
    y_pred = y_pred[idx]
    residuos = y_real - y_pred
    limite = np.percentile(y_real, 99.9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_real, y_pred, s=4, alpha=0.2)
    ax1.plot([0, limite], [0, limite], color="black")
    ax1.set_xlim(0, limite)
    ax1.set_ylim(0, limite)
    ax1.set_xlabel("Viajes reales por hora")
    ax1.set_ylabel("Viajes predichos por hora")
    ax1.set_title("Prediccion vs real (test)")

    ax2.hist(
        residuos, bins=80, range=(np.percentile(residuos, 0.5), np.percentile(residuos, 99.5))
    )
    ax2.axvline(0, color="black")
    ax2.set_xlabel("Residuo (real - predicho)")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Residuos (test)")

    fig.suptitle(titulo)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_p1(features_path: Path, kind="ridge", rf_max_samples=500_000, n_estimators=100):
    train, val, test = load_p1_data(features_path)
    logger.info(f"train={len(train):,} val={len(val):,} test={len(test):,}")

    model = build_model(kind, rf_max_samples, n_estimators)
    logger.info(f"Entrenando {kind}...")
    model.fit(train[FEATURES], train[TARGET])

    metricas = {}
    predicciones = {}
    for nombre, datos in [("val", val), ("test", test)]:
        y_real = datos[TARGET].to_numpy()
        y_pred = model.predict(datos[FEATURES])
        y_pred = np.clip(y_pred, 0, None)  # no puede haber viajes negativos
        predicciones[nombre] = y_pred

        for metrica, valor in regression_metrics(y_real, y_pred).items():
            metricas[f"{nombre}_{metrica}"] = valor

        y_baseline = datos[BASELINE].to_numpy()
        for metrica, valor in regression_metrics(y_real, y_baseline).items():
            metricas[f"baseline_{nombre}_{metrica}"] = valor

    if kind == "ridge":
        run_name = "P1-baseline-ridge"
    else:
        run_name = "P1-candidato-rf"

    params = {
        "model_type": type(model[-1]).__name__,
        "target": "salidas por estacion y hora (t+1)",
        "baseline": "promedio misma hora 4 semanas previas",
        "features": ",".join(FEATURES),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "random_state": SEED,
    }
    if kind == "ridge":
        params["alpha"] = 1.0
    else:
        params["n_estimators"] = n_estimators
        params["max_depth"] = 16
        params["min_samples_leaf"] = 20
        params["max_samples"] = rf_max_samples
    params.update(split_params())
    params.update(data_params())

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(run_tags("P1"))
        mlflow.log_params(params)
        mlflow.log_metrics(metricas)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"p1_{kind}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_serialized")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(train[FEATURES], train[TARGET]),
            input_example=train[FEATURES].head(5),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        fig_path = FIGURES_DIR / f"p1_{kind}_pred_vs_real.png"
        plot_predictions(test[TARGET].to_numpy(), predicciones["test"], run_name, fig_path)
        mlflow.log_artifact(str(fig_path), artifact_path="figures")

    logger.success(
        f"{run_name}: test MAE={metricas['test_mae']:.3f} "
        f"(baseline {metricas['baseline_test_mae']:.3f}), "
        f"RMSE={metricas['test_rmse']:.3f}, R2={metricas['test_r2']:.3f}"
    )
    return metricas
