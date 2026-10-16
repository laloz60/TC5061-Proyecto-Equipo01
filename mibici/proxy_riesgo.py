"""Reporte del proxy de riesgo de P2: sensibilidad, estaciones con mas riesgo y grafica.

Reproduce las cifras de references/proxy_riesgo_p2.md con un solo comando:

    python -m mibici.proxy_riesgo
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import typer

from mibici.config import (
    FIGURES_DIR,
    NOMENCLATURA_PARQUET_PATH,
    REPORTS_DIR,
    RISK_ALPHA,
    UNIFIED_TRIPS_PATH,
)
from mibici.features import (
    add_risk_label,
    add_station_features,
    aggregate_station_hour,
    assign_split,
)

app = typer.Typer()

ALPHAS = (0.3, 0.4, 0.5, 0.6, 0.7)


def build_risk_table(trips, nomenclatura):
    """Tabla estacion x hora con capacidad, etiqueta de riesgo y split (sin filas en NA)."""
    df = aggregate_station_hour(trips)
    df = add_station_features(df, nomenclatura)
    df = add_risk_label(df)
    df = df[df["riesgo"].notna()]
    df = assign_split(df)
    return df


def sensitivity(df, alphas=ALPHAS):
    """Porcentaje de horas en riesgo para cada alpha, total y por split.

    Sirve para justificar alpha: un umbral muy bajo marca casi todo como riesgo y uno
    muy alto deja al modelo sin ejemplos positivos.
    """
    filas = []
    for alpha in alphas:
        umbral = alpha * df["capacidad"]
        vaciado = df["neto_fut"] >= umbral
        saturacion = df["neto_fut"] <= -umbral
        riesgo = vaciado | saturacion
        fila = {
            "alpha": alpha,
            "vaciado_pct": 100 * vaciado.mean(),
            "saturacion_pct": 100 * saturacion.mean(),
            "riesgo_pct": 100 * riesgo.mean(),
        }
        for split in ("train", "val", "test"):
            fila[f"{split}_pct"] = 100 * riesgo[df["split"] == split].mean()
        filas.append(fila)
    return pd.DataFrame(filas).round(2)


def top_stations(df, nomenclatura, n=10):
    """Estaciones con mayor porcentaje de horas en riesgo (prioridad para rebalanceo)."""
    top = df.groupby("estacion_id").agg(
        horas=("riesgo", "size"),
        riesgo_pct=("riesgo", "mean"),
        horas_vaciado=("riesgo_vaciado", "sum"),
        horas_saturacion=("riesgo_saturacion", "sum"),
    )
    top["riesgo_pct"] = (100 * top["riesgo_pct"]).round(2)
    nombres = nomenclatura[["estacion_id", "nombre", "capacidad"]].astype({"estacion_id": "int32"})
    top = top.reset_index().merge(nombres, on="estacion_id")
    return top.sort_values("riesgo_pct", ascending=False).head(n)


def plot_risk_by_hour(df, path):
    """Grafica las horas en riesgo por hora del dia, separando vaciado y saturacion."""
    por_hora = df.groupby(df["hora"].dt.hour)[["riesgo_vaciado", "riesgo_saturacion"]].sum()
    ax = por_hora.plot(kind="bar", figsize=(10, 4))
    ax.set_xlabel("Hora del dia (t)")
    ax.set_ylabel("Horas-estacion en riesgo")
    ax.set_title(f"P2: horas en riesgo por hora del dia (alpha = {RISK_ALPHA})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@app.command()
def main(
    trips_path: Path = UNIFIED_TRIPS_PATH,
    nomenclatura_path: Path = NOMENCLATURA_PARQUET_PATH,
    output_dir: Path = REPORTS_DIR,
):
    columnas = ["inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    trips = pd.read_parquet(trips_path, columns=columnas)
    nomenclatura = pd.read_parquet(nomenclatura_path)

    df = build_risk_table(trips, nomenclatura)
    logger.info(
        f"Estacion x hora con etiqueta: {len(df):,} filas, {df['estacion_id'].nunique()} estaciones"
    )

    sens = sensitivity(df)
    top = top_stations(df, nomenclatura)
    logger.info(f"Sensibilidad a alpha:\n{sens.to_string(index=False)}")
    logger.info(f"Estaciones con mas riesgo:\n{top.to_string(index=False)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sens.to_csv(output_dir / "p2_proxy_sensibilidad.csv", index=False)
    top.to_csv(output_dir / "p2_proxy_top_estaciones.csv", index=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_risk_by_hour(df, FIGURES_DIR / "p2_riesgo_por_hora.png")
    logger.success(f"Reporte del proxy de P2 guardado en {output_dir}")


if __name__ == "__main__":
    app()
