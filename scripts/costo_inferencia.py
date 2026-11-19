"""Costo por inferencia por escenario y numero de replicas (issue #163).

Cruza los picos de recursos (reports/load/docker_stats_picos.csv, de
scripts/resumen_stats.py) con el resumen de Locust ({ESC}[_2r]_stats.csv)
y estima el costo por inferencia en AWS EC2 (us-east-1, Linux).

Formula (la del issue, "mas defendible"):
    costo/inferencia = precio_hora * duracion_h / peticiones_exitosas

Uso: python scripts/costo_inferencia.py [carpeta]   (por defecto reports/load)
"""

from pathlib import Path
import sys

import pandas as pd

# Precios on-demand y spot (USD/h), us-east-1, Linux. Consultados el 2026-11-19
# en compute.doit.com y calculator.holori.com. Spot cambia a diario: es aproximado
# (t3.large medido; los demas, misma proporcion spot/on-demand de ~38 %).
INSTANCIAS = {
    # nombre: (vCPU, memoria GiB, on-demand USD/h, spot USD/h)
    "t3.small": (2, 2, 0.0208, 0.0078),
    "t3.medium": (2, 4, 0.0416, 0.0157),
    "t3.large": (2, 8, 0.0832, 0.0314),
    "t3.xlarge": (4, 16, 0.1664, 0.0628),
}
DURACION_S = {"A": 120, "B": 180, "C": 60}  # igual que scripts/load_test.sh
MEM_SO_GIB = 0.5  # reserva para el sistema operativo y Docker
MARGEN = 1.2  # 20 % de holgura sobre el pico observado


def instancia_minima(mem_gib: float, vcpu: float) -> str:
    """La instancia mas barata en la que caben el pico de memoria y de CPU."""
    for nombre, (cpus, mem, _, _) in INSTANCIAS.items():
        if mem_gib * MARGEN + MEM_SO_GIB <= mem and vcpu * MARGEN <= cpus:
            return nombre
    raise ValueError(f"Ninguna instancia alcanza: {mem_gib:.2f} GiB, {vcpu:.2f} vCPU")


def resumen_locust(ruta: Path) -> pd.Series:
    df = pd.read_csv(ruta)
    return df[df["Name"] == "Aggregated"].iloc[0]


def main() -> None:
    carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/load")
    picos = pd.read_csv(carpeta / "docker_stats_picos.csv")

    filas = []
    for (esc, rep), grupo in picos.groupby(["escenario", "replicas"]):
        ruta_locust = carpeta / f"{esc}{'_2r' if rep == 2 else ''}_stats.csv"
        if not ruta_locust.exists():
            print(f"Aviso: falta {ruta_locust}, se omite {esc} x{rep}")
            continue
        loc = resumen_locust(ruta_locust)
        exitosas = int(loc["Request Count"] - loc["Failure Count"])
        dur_h = DURACION_S[esc] / 3600

        mem_gib = grupo["mem_max_mib"].sum() / 1024  # suma de servicios = host
        vcpu = grupo["vcpu_pico"].sum()
        inst = instancia_minima(mem_gib, vcpu)
        cpus, mem_inst, od, spot = INSTANCIAS[inst]

        filas.append(
            {
                "escenario": esc,
                "replicas": rep,
                "peticiones_ok": exitosas,
                "fallas_pct": round(100 * loc["Failure Count"] / loc["Request Count"], 2),
                "rps": round(loc["Requests/s"], 2),
                "p95_ms": loc["95%"],
                "mem_pico_gib": round(mem_gib, 2),
                "vcpu_pico": round(vcpu, 2),
                "uso_mem_pct": round(100 * mem_gib / mem_inst, 1),
                "uso_cpu_pct": round(100 * vcpu / cpus, 1),
                "instancia": inst,
                "costo_prueba_usd": round(od * dur_h, 4),
                "usd_por_1M_inf": round(1e6 * od * dur_h / exitosas, 4),
                "usd_por_1M_inf_spot": round(1e6 * spot * dur_h / exitosas, 4),
                # Formula base del issue, para comparar
                "usd_por_1M_inf_rps": round(1e6 * od / (loc["Requests/s"] * 3600), 4),
            }
        )

    if not filas:
        sys.exit("No hay escenarios con datos de Locust.")
    tabla = pd.DataFrame(filas)
    # Tabla Markdown lista para pegar en finops.md (sin depender de tabulate)
    print("| " + " | ".join(tabla.columns) + " |")
    print("|" + "---|" * len(tabla.columns))
    for fila in tabla.itertuples(index=False):
        print("| " + " | ".join(str(v) for v in fila) + " |")
    salida = carpeta / "costos_inferencia.csv"
    tabla.to_csv(salida, index=False)
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
