"""Regenera el reporte de P3 (summary + figuras) sin depender de binarios versionados.

El resumen y las figuras de P3 se alimentan de `data/processed/p3_predictions.parquet`,
que a su vez sale de `python -m mibici.modeling.train p3` + `predict p3`. Esos
artefactos (el modelo serializado y el parquet de predicciones) no se versionan por
tamaño, así que este módulo permite reconstruir el reporte desde el modelo ya
entrenado, de forma ejecutable desde terminal y determinista (SEED=42).

Uso (desde la raíz del repo):
    python -m mibici.reportes_p3

Salidas (todas bajo `reports/`):
    p3_evaluation_summary.txt          reparto por clúster + silhouette en inferencia
    figures/p3_distribucion_clusters.png
    figures/p3_mapa_geografico.png
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")  # sin display: el reporte debe correr en CI/terminal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score

from mibici.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    UNIFIED_TRIPS_PATH,
)

SUMMARY_PATH = REPORTS_DIR / "p3_evaluation_summary.txt"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "p3_predictions.parquet"
MODEL_PATH = MODELS_DIR / "kmeans_p3.pkl"


def cargar_predicciones() -> pd.DataFrame:
    """Carga las predicciones de P3 y valida que traigan lo mínimo necesario.

    Se exige `cluster_assigned`, `latitud` y `longitud` porque el summary y el mapa
    se construyen sobre ellas; si falta el archivo, el mensaje indica el comando que
    lo genera en vez de fallar con un traceback críptico.
    """
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Falta {PREDICTIONS_PATH}. Genera las predicciones con:\n"
            "  python -m mibici.dataset\n"
            "  python -m mibici.modeling.train p3\n"
            "  python -m mibici.modeling.predict p3"
        )
    df = pd.read_parquet(PREDICTIONS_PATH)
    faltantes = {"cluster_assigned", "latitud", "longitud"} - set(df.columns)
    if faltantes:
        raise ValueError(f"p3_predictions.parquet no trae las columnas {sorted(faltantes)}")
    return df


def perfiles_horarios(df_nom: pd.DataFrame):
    """Reconstruye el perfil horario (proporción de salidas por hora) de cada estación.

    Es exactamente el mismo features sobre el que se entrena y se evalúa P3: una fila
    por estación con 24 columnas que reparten las salidas del día. Se calcula aquí (y
    no se guarda en `p3_predictions.parquet`) porque es derivable desde los viajes
    crudos; persistirlo duplicaría datos sin necesidad.
    """
    if not UNIFIED_TRIPS_PATH.exists():
        raise FileNotFoundError(
            f"Falta {UNIFIED_TRIPS_PATH}. Sin los viajes crudos no se puede reconstruir "
            "el perfil horario ni el silhouette. Corre: python -m mibici.dataset"
        )
    viajes = pd.read_parquet(UNIFIED_TRIPS_PATH, columns=["inicio_viaje", "origen_id"])
    viajes["hora"] = pd.to_datetime(viajes["inicio_viaje"]).dt.hour
    conteos = (
        pd.crosstab(viajes["origen_id"], viajes["hora"])
        .reindex(columns=range(24), fill_value=0)
        .reindex(df_nom["estacion_id"], fill_value=0)
    )
    totales = conteos.sum(axis=1)
    totales[totales == 0] = 1  # estación sin viajes: evita división entre cero
    return conteos.div(totales, axis=0).values


def silhouette_en_inferencia(df: pd.DataFrame) -> float | None:
    """Silhouette sobre perfiles horarios, con el mismo features con que se entrenó P3.

    Devuelve `None` (y no truena) si hay un solo clúster: el silhouette no está
    definido para k=1 y el reporte debe poder escribirse igualmente.
    """
    if df["cluster_assigned"].nunique() < 2:
        return None
    X = perfiles_horarios(df)
    return float(silhouette_score(X, df["cluster_assigned"].values))


def escribir_summary(df: pd.DataFrame, silhouette: float | None) -> str:
    """Escribe el summary de P3 y devuelve su contenido como texto."""
    total = len(df)
    # Formato idéntico al que escribe predict.py, para que el archivo sea el mismo
    # salga del script de inferencia o de este regenerador de reportes.
    lineas = [
        "=== REPORTE DE EVALUACIÓN DE PREDICCIONES P3 ===",
        f"Total de estaciones evaluadas: {total}",
        "",
        "--- Distribución de Estaciones por Clúster ---",
    ]
    conteo = df["cluster_assigned"].value_counts().sort_index()
    for cluster, n in conteo.items():
        lineas.append(f"Clúster {cluster}: {n} estaciones ({n / total:.2%})")
    lineas.append("")
    if silhouette is None:
        lineas.append("Silhouette Score en Inferencia: n/a (menos de 2 clústeres)")
    else:
        lineas.append(f"Silhouette Score en Inferencia: {silhouette:.4f}")
    contenido = "\n".join(lineas) + "\n"
    SUMMARY_PATH.write_text(contenido, encoding="utf-8")
    return contenido


def graficar_distribucion(df: pd.DataFrame) -> Path:
    """Histograma de estaciones por clúster (la figura que se pega en el Word)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # countplot sin hue (figura byte-idéntica a la original); el FutureWarning de
    # palette es cosmético y se descarta con -W ignore en el CLI si molesta.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.countplot(data=df, x="cluster_assigned", palette="viridis", ax=ax)
    ax.set_title("Distribución de Estaciones de MiBici por Clúster (P3)")
    ax.set_xlabel("ID de Clúster")
    ax.set_ylabel("Número de Estaciones")
    out = FIGURES_DIR / "p3_distribucion_clusters.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def graficar_mapa(df: pd.DataFrame) -> Path:
    """Mapa espacial coloreado por clúster (la figura de ubicación en la ciudad)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        df["longitud"],
        df["latitud"],
        c=df["cluster_assigned"],
        cmap="viridis",
        s=60,
        alpha=0.8,
    )
    fig.colorbar(scatter, ax=ax, label="Clúster Asignado")
    ax.set_title("Mapa Espacial de Clústeres de Estaciones - MiBici P3")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    out = FIGURES_DIR / "p3_mapa_geografico.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = cargar_predicciones()

    # El modelo no es insumo del reporte, pero se avisa si no está: sin él no se pueden
    # regenerar las predicciones desde cero en otra máquina.
    if not MODEL_PATH.exists():
        print(f"[aviso] No encontré {MODEL_PATH}; las predicciones no serán reproducibles.")

    sil = silhouette_en_inferencia(df)
    print(escribir_summary(df, sil))
    print(f"Figura escrita: {graficar_distribucion(df)}")
    print(f"Figura escrita: {graficar_mapa(df)}")


if __name__ == "__main__":
    main()
