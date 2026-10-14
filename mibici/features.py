"""Feature engineering por estación × hora sin fuga temporal.

Convierte el Parquet unificado de viajes en una tabla estación × hora con los
conteos de salidas y llegadas y las variables explicativas que comparten P1
(regresión: viajes que se iniciarán en la hora siguiente) y P2 (clasificación:
riesgo de estación vacía/saturada, cuyo proxy usa el saldo neto y la
capacidad). Cada fila representa la hora `hora` de la estación `estacion_id`:
los targets (`salidas`, `llegadas`) son lo que ocurre DURANTE esa hora, y
todas las features se calculan solo con horas anteriores, es decir, con lo
que se conoce al momento de predecir (fin de la hora previa).

Pipeline: viajes crudos → agregación estación × hora (malla completa con
ceros) → calendario → ventanas históricas desplazadas → atributos fijos de
la estación → etiqueta de partición temporal.

Las fechas del dataset son hora local de Guadalajara sin zona horaria; México
eliminó el horario de verano en 2022, así que no hay horas duplicadas ni
faltantes en el periodo 2023-05..2026-09.

Uso:
    python -m mibici.features
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import (
    DATA_END,
    DATA_START,
    NOMENCLATURA_PARQUET_PATH,
    PROCESSED_DATA_DIR,
    TRAIN_END,
    UNIFIED_TRIPS_PATH,
    VAL_END,
)

app = typer.Typer()

FEATURES_PATH = PROCESSED_DATA_DIR / "features_estacion_hora.parquet"

# Lags en horas: 1h (inercia inmediata), 24h (misma hora de ayer) y 168h
# (misma hora de la semana pasada: captura el ciclo semanal laboral/fin).
LAGS_HORAS = (1, 24, 168)
# Ventana del promedio móvil: 7 días.
VENTANA_MOVIL_HORAS = 7 * 24
# Semanas para el promedio "misma hora y día de la semana" (baseline de P1, #5).
SEMANAS_BASELINE = 4


def aggregate_station_hour(
    trips: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cuenta salidas y llegadas por estación y hora, con ceros explícitos.

    Las salidas se asignan a la hora de `inicio_viaje` en `origen_id` (target
    de P1) y las llegadas a la hora de `fin_viaje` en `destino_id`. Con ambas
    se obtiene el saldo neto que usa el proxy de riesgo de P2.

    La malla es completa (una fila por hora, aunque no haya viajes) desde la
    primera actividad de cada estación hasta `end`. Sin ceros explícitos,
    `shift(1)` regresaría la última hora CON viajes en lugar de la hora
    anterior, y los lags y promedios quedarían sesgados hacia arriba. La
    malla empieza en la primera actividad porque 173 estaciones abrieron
    después de 2023-06 y antes de eso no existían (no son horas con cero
    demanda).

    Args:
        trips: viajes con `inicio_viaje`, `fin_viaje`, `origen_id`, `destino_id`.
        start: primera hora de la malla (default: `DATA_START`).
        end: última hora de la malla (default: última hora de `DATA_END`).

    Returns:
        DataFrame ordenado por `estacion_id`, `hora` con `salidas` y `llegadas`.
    """
    start = pd.Timestamp(start or DATA_START)
    end = pd.Timestamp(end or pd.Timestamp(DATA_END) + pd.Timedelta(hours=23))

    salidas = (
        trips.assign(hora=trips["inicio_viaje"].dt.floor("h"))
        .groupby(["origen_id", "hora"], observed=True)
        .size()
        .rename_axis(["estacion_id", "hora"])
        .rename("salidas")
    )
    llegadas = (
        trips.assign(hora=trips["fin_viaje"].dt.floor("h"))
        .groupby(["destino_id", "hora"], observed=True)
        .size()
        .rename_axis(["estacion_id", "hora"])
        .rename("llegadas")
    )
    counts = pd.concat([salidas, llegadas], axis=1).fillna(0).reset_index()
    # Viajes que terminan después del corte (p. ej. 2026-10-01 00:05) caen fuera.
    counts = counts[(counts["hora"] >= start) & (counts["hora"] <= end)]

    apertura = counts.groupby("estacion_id")["hora"].min()
    horas = pd.date_range(start, end, freq="h")
    grid = pd.concat(
        [
            pd.DataFrame({"estacion_id": estacion, "hora": horas[horas >= primera]})
            for estacion, primera in apertura.items()
        ],
        ignore_index=True,
    )
    grid = grid.merge(counts, on=["estacion_id", "hora"], how="left")
    grid[["salidas", "llegadas"]] = grid[["salidas", "llegadas"]].fillna(0).astype("int16")
    grid["estacion_id"] = grid["estacion_id"].astype("int32")
    return grid.sort_values(["estacion_id", "hora"], ignore_index=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables de calendario derivadas de `hora`.

    La demanda de MiBici tiene picos de traslado laboral (mañana y tarde entre
    semana) que desaparecen en fin de semana, y una estacionalidad por mes
    (vacaciones, lluvias). Estas variables solo dependen del reloj, así que se
    conocen con anticipación y no introducen fuga.
    """
    df = df.copy()
    df["hora_del_dia"] = df["hora"].dt.hour.astype("int8")
    df["dia_semana"] = df["hora"].dt.dayofweek.astype("int8")  # 0 = lunes
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["mes"] = df["hora"].dt.month.astype("int8")
    return df


def add_history_features(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("salidas", "neto"),
    lags: tuple[int, ...] = LAGS_HORAS,
    window: int = VENTANA_MOVIL_HORAS,
    weeks: int = SEMANAS_BASELINE,
) -> pd.DataFrame:
    """Agrega lags, promedio móvil y promedio de la misma hora en semanas previas.

    Todas las ventanas se desplazan al menos una hora (`shift(1)` o más) dentro
    de cada estación, así que la fila de la hora h solo ve las horas h-1 hacia
    atrás: es lo que el operador conoce al predecir la hora h. Requiere la
    malla completa de `aggregate_station_hour` para que desplazar k filas
    equivalga a retroceder k horas.

    - `{col}_lag_{k}h`: valor de hace k horas (P1: inercia y ciclos diario y
      semanal; P2: el saldo neto reciente anticipa vaciado o saturación).
    - `{col}_media_7d`: promedio de las 168 horas previas (nivel típico de la
      estación, suaviza días atípicos).
    - `{col}_prom_{weeks}sem`: promedio de la misma hora y día de la semana en
      las `weeks` semanas previas; es el baseline ingenuo de P1 (#5).

    Las primeras horas de cada estación quedan en NaN cuando no hay historia
    suficiente; el modelo decide si las descarta o las imputa.

    `neto` = salidas − llegadas: positivo significa que la estación pierde
    bicicletas en esa hora.
    """
    df = df.copy()
    if "neto" in columns and "neto" not in df:
        df["neto"] = df["salidas"].astype("int32") - df["llegadas"].astype("int32")

    by_station = df.groupby("estacion_id", sort=False)
    for col in columns:
        series = by_station[col]
        for k in lags:
            df[f"{col}_lag_{k}h"] = series.shift(k).astype("float32")
        df[f"{col}_media_7d"] = series.transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        ).astype("float32")
        same_hour = [series.shift(24 * 7 * w) for w in range(1, weeks + 1)]
        df[f"{col}_prom_{weeks}sem"] = (
            pd.concat(same_hour, axis=1).mean(axis=1, skipna=False).astype("float32")
        )
    return df


def add_station_features(df: pd.DataFrame, nomenclatura: pd.DataFrame) -> pd.DataFrame:
    """Une los atributos fijos de la estación: capacidad y coordenadas.

    La capacidad (`dpcapacity`) es el umbral físico del proxy de riesgo de P2
    (una estación con 15 anclajes se satura antes que una de 40) y también
    acota la demanda de P1. Latitud y longitud dejan que el modelo aprenda
    patrones por zona (centro, zonas corporativas, residenciales). Son
    atributos estáticos, así que no introducen fuga. Las estaciones con viajes
    que no aparecen en la nomenclatura vigente quedan con NaN.
    """
    station = nomenclatura[["estacion_id", "capacidad", "latitud", "longitud"]].astype(
        {
            "estacion_id": "int32",
            "capacidad": "float32",
            "latitud": "float32",
            "longitud": "float32",
        }
    )
    return df.merge(station, on="estacion_id", how="left")


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta cada hora con su partición temporal (train/val/test).

    Usa los cortes de `config.py` (train 2023-05..2024-12, val 2025-01..2025-06,
    test 2025-07..2026-09). Las features de las primeras horas de val y test
    usan horas de la partición anterior; es correcto, porque en producción esa
    historia también estaría disponible al predecir.
    """
    df = df.copy()
    day = df["hora"].dt.normalize()
    split = pd.Series("test", index=df.index)
    split[day <= pd.Timestamp(VAL_END)] = "val"
    split[day <= pd.Timestamp(TRAIN_END)] = "train"
    df["split"] = split.astype("category")
    return df


def build_features(trips: pd.DataFrame, nomenclatura: pd.DataFrame) -> pd.DataFrame:
    """Corre el pipeline completo: agregación → calendario → historia → estación → split."""
    df = aggregate_station_hour(trips)
    logger.info(
        f"Malla estación × hora: {len(df):,} filas, {df['estacion_id'].nunique()} estaciones"
    )
    df = add_calendar_features(df)
    df = add_history_features(df)
    df = add_station_features(df, nomenclatura)
    return assign_split(df)


@app.command()
def main(
    trips_path: Path = typer.Option(UNIFIED_TRIPS_PATH, help="Parquet unificado de viajes."),
    nomenclatura_path: Path = typer.Option(NOMENCLATURA_PARQUET_PATH, help="Nomenclatura."),
    output_path: Path = typer.Option(FEATURES_PATH, help="Parquet de salida."),
):
    """Genera la tabla de features estación × hora para P1 y P2."""
    logger.info(f"Leyendo viajes desde {trips_path}")
    trips = pd.read_parquet(
        trips_path, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    )
    nomenclatura = pd.read_parquet(nomenclatura_path)

    features = build_features(trips, nomenclatura)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.success(
        f"Features: {len(features):,} filas × {features.shape[1]} columnas -> {output_path}"
    )


if __name__ == "__main__":
    app()
