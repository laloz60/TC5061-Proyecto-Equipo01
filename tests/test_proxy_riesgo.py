"""Pruebas del reporte del proxy de riesgo de P2 (issue #95).

Usan viajes sintéticos chicos: los fixtures reales tienen millones de filas.
"""

import pandas as pd

from mibici.proxy_riesgo import build_risk_table, plot_risk_by_hour, sensitivity, top_stations

NOMENCLATURA = pd.DataFrame(
    {
        "estacion_id": [1, 2, 3],
        "nombre": ["Obelisco", "Arcos", "Andares"],
        "capacidad": [10, 12, 8],
        "latitud": [20.67, 20.68, 20.69],
        "longitud": [-103.35, -103.36, -103.37],
    }
)


def crear_viajes(filas):
    df = pd.DataFrame(filas, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"])
    df["inicio_viaje"] = pd.to_datetime(df["inicio_viaje"])
    df["fin_viaje"] = pd.to_datetime(df["fin_viaje"])
    return df


def viajes_con_riesgo():
    """Dos casos claros con alpha=0.5 y horizonte=3:

    - la estacion 1 recibe 40 llegadas en las horas 13 y 14 (saturacion
      en la hora 12, cuando mira 3 horas hacia adelante);
    - la estacion 3 hace 40 salidas en las horas 14 y 15 (vaciado en la 14).
    """
    filas = (
        [("2024-03-01 12:10", "2024-03-01 13:05", 2, 1)] * 20
        + [("2024-03-01 12:20", "2024-03-01 14:05", 2, 1)] * 20
        + [("2024-03-01 14:10", "2024-03-01 15:30", 3, 1)] * 20
        + [("2024-03-01 15:10", "2024-03-01 16:30", 3, 1)] * 20
    )
    return crear_viajes(filas)


# --------------------------------------------------------------------------
# build_risk_table
# --------------------------------------------------------------------------

def test_build_risk_table_no_deja_filas_sin_etiqueta():
    df = build_risk_table(viajes_con_riesgo(), NOMENCLATURA)

    assert df["riesgo"].notna().all()
    assert df["capacidad"].notna().all()
    assert set(df["split"]) <= {"train", "val", "test"}
    assert df["riesgo"].isin([0, 1]).all()


def test_build_risk_table_marca_saturacion_y_vaciado():
    df = build_risk_table(viajes_con_riesgo(), NOMENCLATURA)
    df = df.set_index(["estacion_id", "hora"])

    # la estacion 1 empieza a tener filas en la hora 13 (su primera llegada)
    # y ve neto_fut = -60 ahí (llegadas de las 13-15)
    saturada = df.loc[(1, pd.Timestamp("2024-03-01 13:00"))]
    assert saturada["riesgo_saturacion"] == 1
    assert saturada["riesgo_vaciado"] == 0
    assert saturada["riesgo"] == 1

    # la estacion 3 ve neto_fut = +40 en la hora 14 (salidas de las 15-17)
    vaciada = df.loc[(3, pd.Timestamp("2024-03-01 14:00"))]
    assert vaciada["riesgo_vaciado"] == 1
    assert vaciada["riesgo_saturacion"] == 0


def test_build_risk_table_estacion_sin_desbalance_no_marca():
    df = build_risk_table(viajes_con_riesgo(), NOMENCLATURA)

    # la estacion 2 solo tuvo salidas en la hora 12: su neto_fut posterior es 0
    estacion2 = df[df["estacion_id"] == 2]
    assert estacion2["riesgo"].sum() == 0


def test_build_risk_table_estacion_inexistente_en_nomenclatura():
    # un id de viaje que no esta en la nomenclatura no debe romper el reporte
    viajes = crear_viajes([("2024-03-01 12:10", "2024-03-01 13:05", 99, 1)] * 20)
    df = build_risk_table(viajes, NOMENCLATURA)

    assert 99 not in df["estacion_id"].values
    assert df["riesgo"].notna().all()


# --------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------

def test_sensitivity_porcentajes_exactos():
    df = pd.DataFrame(
        {
            "capacidad": [10.0] * 4,
            "neto_fut": [5.0, -5.0, 9.0, 0.0],
            "split": ["train", "train", "val", "test"],
        }
    )
    sens = sensitivity(df, alphas=(0.5,)).iloc[0]

    # umbral = 0.5 * 10 = 5: vaciado en [5, 9], saturacion en [-5]
    assert sens["alpha"] == 0.5
    assert sens["vaciado_pct"] == 50.0
    assert sens["saturacion_pct"] == 25.0
    assert sens["riesgo_pct"] == 75.0
    # por split: train tiene 2 filas y ambas en riesgo; test ninguna
    assert sens["train_pct"] == 100.0
    assert sens["val_pct"] == 100.0
    assert sens["test_pct"] == 0.0


def test_sensitivity_alpha_marcas_mas_a_umbral_bajo():
    # justifica la eleccion de alpha: 0.3 marca mas horas que 0.7
    df = pd.DataFrame(
        {
            "capacidad": [10.0] * 6,
            "neto_fut": [4.0, 4.0, -4.0, -4.0, 8.0, 0.0],
            "split": ["train"] * 6,
        }
    )
    sens = sensitivity(df, alphas=(0.3, 0.7)).set_index("alpha")

    assert sens.loc[0.3, "riesgo_pct"] > sens.loc[0.7, "riesgo_pct"]


def test_sensitivity_una_fila_por_alpha():
    sens = sensitivity(
        pd.DataFrame({"capacidad": [10.0], "neto_fut": [0.0], "split": ["train"]})
    )
    assert sens["alpha"].tolist() == [0.3, 0.4, 0.5, 0.6, 0.7]


# --------------------------------------------------------------------------
# top_stations
# --------------------------------------------------------------------------

def test_top_stations_ordena_por_riesgo_y_trae_nombre():
    df = pd.DataFrame(
        {
            "estacion_id": [1, 1, 2, 2],
            "riesgo": [1, 0, 0, 0],
            "riesgo_vaciado": [1, 0, 0, 0],
            "riesgo_saturacion": [0, 0, 0, 0],
        }
    )
    top = top_stations(df, NOMENCLATURA, n=1)

    assert len(top) == 1
    assert top.iloc[0]["estacion_id"] == 1
    assert top.iloc[0]["nombre"] == "Obelisco"
    assert top.iloc[0]["riesgo_pct"] == 50.0  # 1 de 2 horas
    assert top.iloc[0]["horas_vaciado"] == 1


def test_top_stations_respeta_el_corte_n():
    df = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3],
            "riesgo": [1, 0, 0],
            "riesgo_vaciado": [0, 0, 0],
            "riesgo_saturacion": [1, 0, 0],
        }
    )
    top = top_stations(df, NOMENCLATURA, n=2)

    assert len(top) == 2
    assert top.iloc[0]["estacion_id"] == 1  # el de mayor riesgo va primero


def test_top_stations_sin_estacion_en_nomenclatura_no_rompe():
    df = pd.DataFrame(
        {
            "estacion_id": [1, 99],
            "riesgo": [1, 1],
            "riesgo_vaciado": [1, 0],
            "riesgo_saturacion": [0, 1],
        }
    )
    top = top_stations(df, NOMENCLATURA, n=10)

    # el merge inner con nomenclatura deja fuera a la 99
    assert 99 not in top["estacion_id"].values
    assert 1 in top["estacion_id"].values


# --------------------------------------------------------------------------
# plot_risk_by_hour
# --------------------------------------------------------------------------

def test_plot_risk_by_hour_crea_el_archivo(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    df = pd.DataFrame(
        {
            "hora": pd.date_range("2024-03-01 08:00", periods=4, freq="h"),
            "riesgo_vaciado": [1, 0, 0, 1],
            "riesgo_saturacion": [0, 1, 0, 0],
        }
    )
    salida = tmp_path / "riesgo.png"

    plot_risk_by_hour(df, salida)

    assert salida.exists()
    assert salida.stat().st_size > 0
