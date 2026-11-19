"""Graficas de latencia, RPS y tasa de error de las pruebas de carga (issue #168).

Lee los *_stats_history.csv que genera Locust (scripts/load_test.sh) y guarda
en reports/load/figures/:
    latencia_p95.png   p95 en el tiempo, A/B/C, 1 contra 2 replicas
    rps.png            peticiones por segundo en el tiempo
    tasa_error.png     % de peticiones fallidas en el tiempo; en C incluye la
                       corrida con 2 replicas ANTES del arreglo de nginx
                       (reports/load/c2r_antes_fix/)

Uso: python scripts/graficas_carga.py [carpeta]   (por defecto reports/load)
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ESCENARIOS = {
    "A": "A · rampa 50 usuarios",
    "B": "B · 50 usuarios sostenidos",
    "C": "C · pico 100 usuarios",
}
# Paleta categorica (orden fijo): 1 replica, 2 replicas, 2 replicas antes del fix
SERIES = {
    "1": ("1 réplica", "#2a78d6"),
    "2": ("2 réplicas + nginx", "#eb6834"),
    "antes": ("2 réplicas, antes del fix de nginx", "#1baf7a"),
}
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def leer_historial(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta, na_values="N/A")
    df = df[df["Name"] == "Aggregated"].copy()
    df["seg"] = df["Timestamp"] - df["Timestamp"].iloc[0]
    df["error_pct"] = (100 * df["Failures/s"] / df["Requests/s"]).where(df["Requests/s"] > 0)
    return df


def corridas(carpeta: Path, esc: str) -> list[tuple[str, pd.DataFrame]]:
    """Series a dibujar en el panel de un escenario."""
    rutas = [
        ("1", carpeta / f"{esc}_stats_history.csv"),
        ("2", carpeta / f"{esc}_2r_stats_history.csv"),
    ]
    if esc == "C":
        rutas.append(("antes", carpeta / "c2r_antes_fix" / "C_2r_stats_history.csv"))
    return [(clave, leer_historial(r)) for clave, r in rutas if r.exists()]


def estilo(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def grafica(
    carpeta: Path, columna: str, titulo: str, eje_y: str, archivo: str, series: tuple[str, ...]
) -> Path:
    fig, ejes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (esc, nombre) in zip(ejes, ESCENARIOS.items(), strict=True):
        for clave, df in corridas(carpeta, esc):
            if clave not in series:
                continue
            etiqueta, color = SERIES[clave]
            ax.plot(df["seg"], df[columna], color=color, linewidth=1.6, label=etiqueta)
        ax.set_title(nombre, loc="left", fontsize=10, color=TINTA)
        ax.set_xlabel("segundos desde el inicio", fontsize=9, color=TINTA_2)
        estilo(ax)
    ejes[0].set_ylabel(eje_y, fontsize=9, color=TINTA_2)
    # Leyenda unica con todas las series presentes (el panel C tiene las 3)
    manejadores, etiquetas = ejes[2].get_legend_handles_labels()
    fig.legend(
        manejadores, etiquetas, loc="upper right", ncol=len(etiquetas), frameon=False, fontsize=9
    )
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    salida = carpeta / "figures" / archivo
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)
    return salida


def main() -> None:
    carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/load")
    (carpeta / "figures").mkdir(exist_ok=True)
    salidas = [
        grafica(
            carpeta,
            "95%",
            "Latencia p95 por escenario",
            "p95 (ms)",
            "latencia_p95.png",
            ("1", "2"),
        ),
        grafica(carpeta, "Requests/s", "Peticiones por segundo", "RPS", "rps.png", ("1", "2")),
        grafica(
            carpeta,
            "error_pct",
            "Tasa de error (1 y 2 réplicas: 0 % en todas las corridas; la línea azul queda bajo la naranja)",
            "% de peticiones fallidas",
            "tasa_error.png",
            ("1", "2", "antes"),
        ),
    ]
    for s in salidas:
        print(f"Guardado en {s}")


if __name__ == "__main__":
    main()
