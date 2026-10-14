"""P1 — Regresión: ¿cuántos viajes se iniciarán en una estación en la próxima hora?

Construcción del target (#5)
----------------------------
- Fuente: Parquet unificado (`inicio_viaje`, `origen_id`), hora local de
  Guadalajara sin zona horaria (sin horario de verano desde 2022).
- Agregación: `salidas` = número de viajes con `origen_id` = estación y
  `inicio_viaje` dentro de la hora [h, h+1). Las horas sin viajes cuentan como
  0 (malla completa desde la primera actividad de la estación). Ver
  `mibici.features.aggregate_station_hour`.
- Horizonte t+1: al cierre de la hora h-1 se predice `salidas` de la hora h.
  Todas las features de la fila h usan solo horas <= h-1 (sin fuga temporal).
- Filtros: se descartan las filas con features NaN (las primeras 4 semanas de
  cada estación, sin historia suficiente, y 2 estaciones sin capacidad en la
  nomenclatura), para que baseline y modelos se evalúen sobre las mismas filas.

Pipeline: limpieza (ingesta) → agregación estación × hora → feature
engineering (`features.py`) → entrenamiento en train → evaluación en val y
test → registro en MLflow.

Métricas (guía de scikit-learn, "Regression metrics"):
- MAE: error medio en viajes por hora; es la más interpretable para el
  operador ("nos equivocamos en ~0.8 bicis por estación y hora").
- RMSE: penaliza más los errores grandes, que en operación son los picos de
  demanda que dejan una estación sin bicis.
- R²: proporción de varianza explicada, comparable entre modelos; un valor
  cercano a 0 equivale a predecir la media.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mibici.config import FIGURES_DIR, MODELS_DIR, SEED
from mibici.modeling.tracking import (
    EXPERIMENT_NAME,
    data_params,
    run_tags,
    split_params,
)

TARGET = "salidas"
BASELINE_COLUMN = "salidas_prom_4sem"
CATEGORICAL = ["hora_del_dia", "dia_semana", "mes"]
NUMERIC = [
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
FEATURES = CATEGORICAL + NUMERIC

RUN_NAMES = {"ridge": "P1-baseline-ridge", "rf": "P1-candidato-rf"}


def load_p1_data(features_path: Path) -> dict[str, pd.DataFrame]:
    """Lee la tabla de features y la separa por partición, sin filas con NaN."""
    df = pd.read_parquet(
        features_path, columns=["estacion_id", "hora", TARGET, *FEATURES, "split"]
    )
    before = len(df)
    df = df.dropna(subset=FEATURES)
    logger.info(f"Filas con historia completa: {len(df):,} de {before:,}")
    return {name: part for name, part in df.groupby("split", observed=True)}


def build_model(kind: str, rf_max_samples: int = 500_000) -> Pipeline:
    """Ridge (lineal, con hora/día/mes en one-hot) o RandomForest (no lineal).

    Ridge necesita one-hot porque el efecto de la hora no es lineal (picos a
    las 8 y 18 h). El RandomForest captura esas interacciones por sí solo y se
    entrena con submuestreo por árbol para que quepa en una laptop.
    """
    if kind == "ridge":
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                ("num", StandardScaler(), NUMERIC),
            ]
        )
        return Pipeline([("pre", pre), ("model", Ridge(alpha=1.0, random_state=SEED))])
    if kind == "rf":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=20,
            max_samples=rf_max_samples,
            n_jobs=-1,
            random_state=SEED,
        )
        return Pipeline([("model", model)])
    raise ValueError(f"Modelo no soportado: {kind!r} (usa 'ridge' o 'rf')")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE y R²."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> Path:
    """Predicción vs real y distribución de residuos (muestra de 50k puntos)."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_true), size=min(50_000, len(y_true)), replace=False)
    yt, yp = y_true[idx], y_pred[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    top = float(np.percentile(yt, 99.9))
    ax1.scatter(yt, yp, s=4, alpha=0.2)
    ax1.plot([0, top], [0, top], color="black", linewidth=1)
    ax1.set(
        xlim=(0, top),
        ylim=(0, top),
        xlabel="Viajes reales por hora",
        ylabel="Viajes predichos por hora",
        title="Predicción vs real (test)",
    )
    residuos = yt - yp
    ax2.hist(residuos, bins=80, range=tuple(np.percentile(residuos, [0.5, 99.5])))
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set(xlabel="Residuo (real − predicho)", ylabel="Frecuencia", title="Residuos (test)")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def train_p1(features_path: Path, kind: str = "ridge", rf_max_samples: int = 500_000) -> dict:
    """Entrena un modelo P1, lo compara con el baseline y registra el run en MLflow.

    El run guarda las métricas del baseline (promedio de la misma hora en las 4
    semanas previas) junto a las del modelo, en val y test, para que la mejora
    se lea directamente en la UI de MLflow.
    """
    parts = load_p1_data(features_path)
    train, val, test = parts["train"], parts["val"], parts["test"]
    logger.info(f"train={len(train):,} | val={len(val):,} | test={len(test):,}")

    model = build_model(kind, rf_max_samples=rf_max_samples)
    logger.info(f"Entrenando {kind} sobre {len(train):,} filas…")
    model.fit(train[FEATURES], train[TARGET])

    metrics = {}
    for name, part in (("val", val), ("test", test)):
        y = part[TARGET].to_numpy()
        pred = np.clip(model.predict(part[FEATURES]), 0, None)  # no hay viajes negativos
        for key, value in regression_metrics(y, pred).items():
            metrics[f"{name}_{key}"] = value
        for key, value in regression_metrics(y, part[BASELINE_COLUMN].to_numpy()).items():
            metrics[f"baseline_{name}_{key}"] = value
        if name == "test":
            test_pred = pred

    run_name = RUN_NAMES[kind]
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(run_tags("P1"))
        mlflow.log_params(
            {
                "model_type": type(model[-1]).__name__,
                "target": f"{TARGET} (estación × hora, horizonte t+1)",
                "baseline": f"{BASELINE_COLUMN} (misma hora, 4 semanas previas)",
                "features": ",".join(FEATURES),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "random_state": SEED,
                **{
                    f"model__{k}": v
                    for k, v in model[-1].get_params().items()
                    if k
                    in {"alpha", "n_estimators", "max_depth", "min_samples_leaf", "max_samples"}
                },
                **split_params(),
                **data_params(),
            }
        )
        mlflow.log_metrics(metrics)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"p1_{kind}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        fig_path = plot_predictions(
            test[TARGET].to_numpy(),
            test_pred,
            f"P1 — {run_name}",
            FIGURES_DIR / f"p1_{kind}_pred_vs_real.png",
        )
        mlflow.log_artifact(str(fig_path), artifact_path="figures")

    logger.success(
        f"{run_name} ({run.info.run_id[:8]}): test MAE={metrics['test_mae']:.3f} "
        f"vs baseline {metrics['baseline_test_mae']:.3f} | test RMSE={metrics['test_rmse']:.3f} "
        f"| test R²={metrics['test_r2']:.3f}"
    )
    return metrics
