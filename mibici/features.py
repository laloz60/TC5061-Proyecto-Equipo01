from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import (
    DATA_END,
    DATA_START,
    NOMENCLATURA_PARQUET_PATH,
    PROCESSED_DATA_DIR,
    RISK_ALPHA,
    RISK_HORIZON_HOURS,
    TRAIN_END,
    UNIFIED_TRIPS_PATH,
    VAL_END,
)

app = typer.Typer()

FEATURES_PATH = PROCESSED_DATA_DIR / "features_estacion_hora.parquet"


def quitar_viajes_duplicados(trips):
    """Quita los viajes con `viaje_id` repetido (758 filas exactas en 2026-09)."""
    sin_duplicados = trips.drop_duplicates(subset="viaje_id")
    logger.info(f"Viajes duplicados eliminados: {len(trips) - len(sin_duplicados):,}")
    return sin_duplicados


def aggregate_station_hour(trips, start=None, end=None):
    """Cuenta salidas y llegadas por estacion y hora (target de P1 y saldo para P2)."""
    if start is None:
        start = pd.Timestamp(DATA_START)
    if end is None:
        end = pd.Timestamp(DATA_END) + pd.Timedelta(hours=23)

    trips = trips.copy()
    trips["hora_inicio"] = trips["inicio_viaje"].dt.floor("h")
    trips["hora_fin"] = trips["fin_viaje"].dt.floor("h")

    salidas = trips.groupby(["origen_id", "hora_inicio"]).size().reset_index()
    salidas.columns = ["estacion_id", "hora", "salidas"]

    llegadas = trips.groupby(["destino_id", "hora_fin"]).size().reset_index()
    llegadas.columns = ["estacion_id", "hora", "llegadas"]

    conteos = salidas.merge(llegadas, on=["estacion_id", "hora"], how="outer")
    conteos = conteos[(conteos["hora"] >= start) & (conteos["hora"] <= end)]

    # se agregan las horas sin viajes con 0 para que los lags sean de horas seguidas.
    # cada estacion empieza en su primer viaje porque varias abrieron despues de 2023-05
    horas = pd.date_range(start, end, freq="h")
    tablas = []
    for estacion, primera_hora in conteos.groupby("estacion_id")["hora"].min().items():
        tabla = pd.DataFrame({"hora": horas[horas >= primera_hora]})
        tabla["estacion_id"] = estacion
        tablas.append(tabla)
    df = pd.concat(tablas, ignore_index=True)

    df = df.merge(conteos, on=["estacion_id", "hora"], how="left")
    df["salidas"] = df["salidas"].fillna(0).astype("int16")
    df["llegadas"] = df["llegadas"].fillna(0).astype("int16")
    df["estacion_id"] = df["estacion_id"].astype("int32")
    df = df[["estacion_id", "hora", "salidas", "llegadas"]]
    df = df.sort_values(["estacion_id", "hora"]).reset_index(drop=True)
    return df


def add_calendar_features(df):
    """Hora, dia de la semana, fin de semana y mes."""
    df = df.copy()
    df["hora_del_dia"] = df["hora"].dt.hour.astype("int8")
    df["dia_semana"] = df["hora"].dt.dayofweek.astype("int8")
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["mes"] = df["hora"].dt.month.astype("int8")
    return df


def add_history_features(df, columns=("salidas", "neto"), lags=(1, 24, 168), window=168, weeks=4):
    """Lags, promedio movil de 7 dias y promedio de la misma hora en las semanas anteriores.

    Todo se recorre al menos 1 hora para no usar informacion de la hora que se predice.
    """
    df = df.copy()
    if "neto" not in df.columns:
        df["neto"] = df["salidas"].astype("int32") - df["llegadas"].astype("int32")

    grupos = df.groupby("estacion_id")
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = grupos[col].shift(lag).astype("float32")

        anterior = grupos[col].shift(1)
        media = anterior.groupby(df["estacion_id"]).rolling(window, min_periods=window).mean()
        df[f"{col}_media_7d"] = media.reset_index(level=0, drop=True).astype("float32")

        # promedio de la misma hora y dia en las semanas pasadas (baseline de P1)
        semanas = pd.DataFrame(index=df.index)
        for semana in range(1, weeks + 1):
            semanas[semana] = grupos[col].shift(168 * semana)
        df[f"{col}_prom_{weeks}sem"] = semanas.mean(axis=1, skipna=False).astype("float32")
    return df


def add_station_features(df, nomenclatura):
    """Capacidad y coordenadas de cada estacion."""
    estaciones = nomenclatura[["estacion_id", "capacidad", "latitud", "longitud"]].copy()
    estaciones["estacion_id"] = estaciones["estacion_id"].astype("int32")
    estaciones["capacidad"] = estaciones["capacidad"].astype("float32")
    estaciones["latitud"] = estaciones["latitud"].astype("float32")
    estaciones["longitud"] = estaciones["longitud"].astype("float32")
    return df.merge(estaciones, on="estacion_id", how="left")


def add_risk_label(df, horizon=RISK_HORIZON_HOURS, alpha=RISK_ALPHA):
    """Etiqueta de riesgo de P2: la estacion se vacia o se satura en las proximas horas.

    MiBici no publica el inventario de bicis, asi que el riesgo se aproxima con el
    saldo neto (salidas - llegadas) de las horas t+1 .. t+horizon. Partiendo de una
    estacion a media capacidad, si salen mas de alpha * capacidad bicis se queda vacia
    y si llegan mas de alpha * capacidad se queda sin anclajes.
    Ver references/proxy_riesgo_p2.md.

    Usa horas futuras a proposito porque es el objetivo de P2, no una feature:
    neto_fut y las etiquetas nunca deben entrar como variables del modelo.
    Requiere la columna capacidad (add_station_features). Las ultimas horas de cada
    estacion y las estaciones sin capacidad quedan en NA.
    """
    df = df.copy()
    neto = df["salidas"].astype("int32") - df["llegadas"].astype("int32")
    grupos = neto.groupby(df["estacion_id"])
    futuras = pd.DataFrame(index=df.index)
    for k in range(1, horizon + 1):
        futuras[k] = grupos.shift(-k)
    df["neto_fut"] = futuras.sum(axis=1, skipna=False).astype("float32")

    umbral = alpha * df["capacidad"]
    vaciado = df["neto_fut"] >= umbral
    saturacion = df["neto_fut"] <= -umbral
    valido = df["neto_fut"].notna() & (df["capacidad"] > 0)
    df["riesgo_vaciado"] = vaciado.astype("Int8").where(valido)
    df["riesgo_saturacion"] = saturacion.astype("Int8").where(valido)
    df["riesgo"] = (vaciado | saturacion).astype("Int8").where(valido)
    return df


def assign_split(df):
    """Marca cada fila como train, val o test segun las fechas de config.py."""
    df = df.copy()
    dia = df["hora"].dt.normalize()
    df["split"] = "test"
    df.loc[dia <= pd.Timestamp(VAL_END), "split"] = "val"
    df.loc[dia <= pd.Timestamp(TRAIN_END), "split"] = "train"
    df["split"] = df["split"].astype("category")
    return df


def build_features(trips, nomenclatura):
    df = aggregate_station_hour(trips)
    logger.info(f"Estacion x hora: {len(df):,} filas, {df['estacion_id'].nunique()} estaciones")
    df = add_calendar_features(df)
    df = add_history_features(df)
    df = add_station_features(df, nomenclatura)
    df = assign_split(df)
    return df


@app.command()
def main(
    trips_path: Path = UNIFIED_TRIPS_PATH,
    nomenclatura_path: Path = NOMENCLATURA_PARQUET_PATH,
    output_path: Path = FEATURES_PATH,
):
    columnas = ["viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    trips = pd.read_parquet(trips_path, columns=columnas)
    trips = quitar_viajes_duplicados(trips)
    nomenclatura = pd.read_parquet(nomenclatura_path)

    df = build_features(trips, nomenclatura)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Features guardadas en {output_path} ({len(df):,} filas)")


if __name__ == "__main__":
    app()
