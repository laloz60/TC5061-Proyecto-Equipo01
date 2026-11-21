"""Tabla de picos de CPU y memoria por escenario (issue #162).

Lee reports/load/docker_stats.csv (generado por scripts/load_test.sh) y
guarda reports/load/docker_stats_picos.csv.

Con 2 replicas, CPU y memoria de la API se SUMAN por muestra: es el
consumo total que habria que provisionar (insumo del analisis FinOps).
En docker stats 100 % = 1 vCPU completa.

Uso: python scripts/resumen_stats.py [ruta_csv]
"""

import sys

import pandas as pd

A_MIB = {"GiB": 1024, "MiB": 1, "KiB": 1 / 1024, "B": 1 / 1024**2}


def a_mib(valor: str) -> float:
    valor = valor.strip()
    for unidad, factor in A_MIB.items():  # GiB antes que B: el orden importa
        if valor.endswith(unidad):
            return float(valor[: -len(unidad)]) * factor
    raise ValueError(f"Unidad de memoria no reconocida: {valor}")


def main() -> None:
    ruta = sys.argv[1] if len(sys.argv) > 1 else "reports/load/docker_stats.csv"
    df = pd.read_csv(ruta)
    df["cpu"] = df["cpu_perc"].str.rstrip("%").astype(float)
    df["mem_mib"] = df["mem_usada"].map(a_mib)
    # mibici-api-1 / mibici-api-2 -> api ; mibici-mlflow-1 -> mlflow
    df["servicio"] = df["contenedor"].str.replace(r"^mibici-|-\d+$", "", regex=True)

    por_muestra = df.groupby(["escenario", "replicas", "servicio", "timestamp"], as_index=False)[
        ["cpu", "mem_mib"]
    ].sum()

    tabla = por_muestra.groupby(["escenario", "replicas", "servicio"]).agg(
        cpu_max_pct=("cpu", "max"),
        cpu_prom_pct=("cpu", "mean"),
        mem_max_mib=("mem_mib", "max"),
        muestras=("cpu", "size"),
    )
    tabla["vcpu_pico"] = tabla["cpu_max_pct"] / 100
    tabla = tabla.round(2)

    print(tabla.to_string())
    salida = ruta.replace(".csv", "_picos.csv")
    tabla.to_csv(salida)
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
