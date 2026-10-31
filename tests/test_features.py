import numpy as np
import pandas as pd

from mibici.features import (
    add_calendar_features,
    add_history_features,
    add_risk_label,
    add_station_features,
    aggregate_station_hour,
    assign_split,
    quitar_viajes_duplicados,
)

INICIO = pd.Timestamp("2024-01-01 00:00")
FIN = pd.Timestamp("2024-01-01 03:00")


def crear_viajes(filas):
    df = pd.DataFrame(filas, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"])
    df["inicio_viaje"] = pd.to_datetime(df["inicio_viaje"])
    df["fin_viaje"] = pd.to_datetime(df["fin_viaje"])
    return df


def crear_horas(salidas, estacion=1):
    df = pd.DataFrame()
    df["hora"] = pd.date_range(INICIO, periods=len(salidas), freq="h")
    df["estacion_id"] = estacion
    df["salidas"] = salidas
    df["llegadas"] = 0
    return df


def test_quitar_viajes_duplicados_conserva_una_fila_por_id():
    trips = pd.DataFrame({"viaje_id": [1, 1, 2], "origen_id": [10, 10, 20]})

    resultado = quitar_viajes_duplicados(trips)

    assert resultado["viaje_id"].tolist() == [1, 2]


def test_horas_sin_viajes_quedan_en_cero():
    viajes = crear_viajes(
        [
            ("2024-01-01 00:10", "2024-01-01 00:20", 1, 2),
            ("2024-01-01 00:40", "2024-01-01 01:05", 1, 2),
            ("2024-01-01 03:00", "2024-01-01 03:10", 1, 1),
        ]
    )
    df = aggregate_station_hour(viajes, INICIO, FIN)

    estacion1 = df[df["estacion_id"] == 1]
    assert estacion1["salidas"].tolist() == [2, 0, 0, 1]
    assert estacion1["llegadas"].tolist() == [0, 0, 0, 1]

    estacion2 = df[df["estacion_id"] == 2]
    assert estacion2["llegadas"].tolist() == [1, 1, 0, 0]


def test_estacion_empieza_en_su_primer_viaje():
    viajes = crear_viajes(
        [
            ("2024-01-01 00:10", "2024-01-01 00:20", 1, 1),
            ("2024-01-01 02:10", "2024-01-01 02:20", 3, 3),
        ]
    )
    df = aggregate_station_hour(viajes, INICIO, FIN)

    assert len(df[df["estacion_id"] == 1]) == 4
    assert len(df[df["estacion_id"] == 3]) == 2


def test_llegadas_despues_del_fin_no_se_cuentan():
    viajes = crear_viajes([("2024-01-01 03:50", "2024-01-01 04:10", 1, 1)])
    df = aggregate_station_hour(viajes, INICIO, FIN)

    assert df["hora"].max() == FIN
    assert df["llegadas"].sum() == 0


def test_calendario():
    df = add_calendar_features(crear_horas([0] * 24 * 7))

    # 2024-01-06 fue sabado
    sabado = df[df["hora"] == pd.Timestamp("2024-01-06 08:00")].iloc[0]
    assert sabado["dia_semana"] == 5
    assert sabado["es_fin_de_semana"] == 1
    assert sabado["hora_del_dia"] == 8
    assert sabado["mes"] == 1


def test_lags_y_media_movil():
    df = pd.concat([crear_horas(list(range(1, 31)), 1), crear_horas([100] * 30, 2)])
    df = df.reset_index(drop=True)
    df = add_history_features(df, lags=(1, 24), window=3, weeks=1)

    estacion1 = df[df["estacion_id"] == 1].reset_index(drop=True)
    assert np.isnan(estacion1.loc[0, "salidas_lag_1h"])
    assert estacion1.loc[5, "salidas_lag_1h"] == 5
    assert estacion1.loc[24, "salidas_lag_24h"] == 1
    assert estacion1.loc[5, "salidas_media_7d"] == 4  # promedio de 3, 4 y 5

    # la estacion 2 no debe tomar valores de la estacion 1
    estacion2 = df[df["estacion_id"] == 2].reset_index(drop=True)
    assert np.isnan(estacion2.loc[0, "salidas_lag_1h"])


def test_promedio_misma_hora_semanas_anteriores():
    salidas = [0] * (168 * 3)
    salidas[10] = 4
    salidas[168 + 10] = 8
    df = add_history_features(crear_horas(salidas), lags=(1,), window=3, weeks=2)

    assert df.loc[2 * 168 + 10, "salidas_prom_2sem"] == 6
    assert np.isnan(df.loc[168 + 10, "salidas_prom_2sem"])


def test_neto():
    df = crear_horas([5, 1, 0])
    df["llegadas"] = [2, 3, 0]
    df = add_history_features(df, lags=(1,), window=1, weeks=1)

    assert df["neto"].tolist() == [3, -2, 0]
    assert df.loc[1, "neto_lag_1h"] == 3


def test_sin_fuga_de_informacion():
    # si cambian los datos desde la hora h, las features hasta h deben ser iguales
    np.random.seed(0)
    salidas = np.random.randint(0, 10, size=168 * 6)
    h = 168 * 5
    salidas_cambiadas = salidas.copy()
    salidas_cambiadas[h:] = salidas_cambiadas[h:] + 50

    df1 = add_history_features(crear_horas(salidas))
    df2 = add_history_features(crear_horas(salidas_cambiadas))

    columnas = [c for c in df1.columns if "lag" in c or "media" in c or "prom" in c]
    pd.testing.assert_frame_equal(df1.loc[:h, columnas], df2.loc[:h, columnas])


def test_datos_de_estacion():
    nomenclatura = pd.DataFrame(
        {"estacion_id": [1], "capacidad": [15], "latitud": [20.67], "longitud": [-103.35]}
    )
    df = pd.concat([crear_horas([1], 1), crear_horas([1], 99)])
    df = add_station_features(df, nomenclatura)

    assert df[df["estacion_id"] == 1]["capacidad"].iloc[0] == 15
    assert np.isnan(df[df["estacion_id"] == 99]["capacidad"].iloc[0])


def test_split():
    df = pd.DataFrame()
    df["hora"] = pd.to_datetime(
        ["2024-12-31 23:00", "2025-01-01 00:00", "2025-06-30 23:00", "2025-07-01 00:00"]
    )
    df = assign_split(df)
    assert df["split"].tolist() == ["train", "val", "val", "test"]

def test_etiqueta_de_riesgo():
    df = crear_horas([0, 6, 3, 0, 0])
    df["llegadas"] = [0, 0, 0, 9, 1]
    df["capacidad"] = 10.0
    df = add_risk_label(df, horizon=2, alpha=0.5)

    # neto = [0, 6, 3, -9, -1]; neto_fut suma las 2 horas siguientes
    assert df["neto_fut"].tolist()[:3] == [9, -6, -10]
    assert df["neto_fut"].iloc[3:].isna().all()
    assert df["riesgo_vaciado"].tolist()[:3] == [1, 0, 0]
    assert df["riesgo_saturacion"].tolist()[:3] == [0, 1, 1]
    assert df["riesgo"].tolist()[:3] == [1, 1, 1]
    assert df["riesgo"].iloc[3:].isna().all()


def test_riesgo_no_mezcla_estaciones_ni_usa_estaciones_sin_capacidad():
    df = pd.concat([crear_horas([0, 0, 0], 1), crear_horas([9, 9, 9], 2)]).reset_index(drop=True)
    df["capacidad"] = [10.0, 10.0, 10.0, np.nan, np.nan, np.nan]
    df = add_risk_label(df, horizon=1, alpha=0.5)

    # la ultima hora de la estacion 1 no debe tomar las salidas de la estacion 2
    assert pd.isna(df.loc[2, "riesgo"])
    assert df.loc[[0, 1], "riesgo"].tolist() == [0, 0]
    assert df.loc[3:, "riesgo"].isna().all()
