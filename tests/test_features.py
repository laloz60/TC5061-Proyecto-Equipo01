"""Tests del feature engineering estación × hora (datos sintéticos, sin red)."""

import numpy as np
import pandas as pd

from mibici.features import (
    add_calendar_features,
    add_history_features,
    add_station_features,
    aggregate_station_hour,
    assign_split,
)

START = pd.Timestamp("2024-01-01 00:00")


def _trips(rows):
    """rows: (inicio, fin, origen, destino)."""
    df = pd.DataFrame(rows, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"])
    df["inicio_viaje"] = pd.to_datetime(df["inicio_viaje"])
    df["fin_viaje"] = pd.to_datetime(df["fin_viaje"])
    return df


def _hourly(values, estacion_id=1):
    """Malla de una estación con `salidas` dadas y llegadas en cero."""
    return pd.DataFrame(
        {
            "estacion_id": estacion_id,
            "hora": pd.date_range(START, periods=len(values), freq="h"),
            "salidas": np.asarray(values, dtype="int16"),
            "llegadas": np.zeros(len(values), dtype="int16"),
        }
    )


def test_aggregate_fills_missing_hours_with_zeros():
    trips = _trips(
        [
            ("2024-01-01 00:10", "2024-01-01 00:20", 1, 2),
            ("2024-01-01 00:40", "2024-01-01 01:05", 1, 2),
            ("2024-01-01 03:00", "2024-01-01 03:10", 1, 1),
        ]
    )
    grid = aggregate_station_hour(trips, start=START, end=START + pd.Timedelta(hours=3))
    est1 = grid[grid["estacion_id"] == 1].set_index("hora")
    assert est1["salidas"].tolist() == [2, 0, 0, 1]
    assert est1["llegadas"].tolist() == [0, 0, 0, 1]
    est2 = grid[grid["estacion_id"] == 2].set_index("hora")
    assert est2["llegadas"].tolist() == [1, 1, 0, 0]


def test_aggregate_starts_each_station_at_first_activity():
    """Una estación que abre después no se rellena con ceros previos a su apertura."""
    trips = _trips(
        [
            ("2024-01-01 00:10", "2024-01-01 00:20", 1, 1),
            ("2024-01-01 02:10", "2024-01-01 02:20", 3, 3),
        ]
    )
    grid = aggregate_station_hour(trips, start=START, end=START + pd.Timedelta(hours=3))
    assert grid.loc[grid["estacion_id"] == 3, "hora"].min() == START + pd.Timedelta(hours=2)
    assert (grid["estacion_id"] == 1).sum() == 4


def test_aggregate_drops_arrivals_after_end():
    trips = _trips([("2024-01-01 03:50", "2024-01-01 04:10", 1, 1)])
    grid = aggregate_station_hour(trips, start=START, end=START + pd.Timedelta(hours=3))
    assert grid["hora"].max() == START + pd.Timedelta(hours=3)
    assert grid["llegadas"].sum() == 0


def test_calendar_features():
    df = add_calendar_features(_hourly([0] * 24 * 7))  # 2024-01-01 es lunes
    sabado = df[df["hora"] == pd.Timestamp("2024-01-06 08:00")].iloc[0]
    assert sabado["dia_semana"] == 5
    assert sabado["es_fin_de_semana"] == 1
    assert sabado["hora_del_dia"] == 8
    assert sabado["mes"] == 1
    assert df.loc[0, "es_fin_de_semana"] == 0


def test_lags_use_previous_hours_within_station():
    df = pd.concat([_hourly(range(1, 31), estacion_id=1), _hourly([100] * 30, estacion_id=2)])
    out = add_history_features(df.reset_index(drop=True), lags=(1, 24), window=3, weeks=1)
    est1 = out[out["estacion_id"] == 1].reset_index(drop=True)
    assert np.isnan(est1.loc[0, "salidas_lag_1h"])
    assert est1.loc[5, "salidas_lag_1h"] == 5  # hora 5 ve el valor de la hora 4
    assert est1.loc[24, "salidas_lag_24h"] == 1
    assert est1.loc[5, "salidas_media_7d"] == np.mean([3, 4, 5])  # horas 2, 3 y 4
    # El lag de la estación 2 no hereda valores de la estación 1.
    est2 = out[out["estacion_id"] == 2].reset_index(drop=True)
    assert np.isnan(est2.loc[0, "salidas_lag_1h"])


def test_same_hour_average_over_previous_weeks():
    semana = 24 * 7
    values = np.zeros(semana * 3, dtype=int)
    values[[10, semana + 10]] = [4, 8]  # misma hora en las semanas 1 y 2
    out = add_history_features(_hourly(values), lags=(1,), window=3, weeks=2)
    assert out.loc[2 * semana + 10, "salidas_prom_2sem"] == 6
    assert np.isnan(out.loc[semana + 10, "salidas_prom_2sem"])  # solo 1 semana de historia


def test_neto_is_salidas_minus_llegadas():
    df = _hourly([5, 1, 0])
    df["llegadas"] = np.array([2, 3, 0], dtype="int16")
    out = add_history_features(df, lags=(1,), window=1, weeks=1)
    assert out["neto"].tolist() == [3, -2, 0]
    assert out.loc[1, "neto_lag_1h"] == 3


def test_no_leakage_future_values_do_not_change_past_features():
    """Regla de oro de #4: alterar la hora h no puede cambiar ninguna feature de horas <= h."""
    rng = np.random.default_rng(0)
    values = rng.integers(0, 10, size=24 * 7 * 6)
    h = 24 * 7 * 5
    altered = values.copy()
    altered[h:] += 50

    def features(v):
        out = add_history_features(_hourly(v))
        return out.drop(columns=["salidas", "llegadas", "neto"])

    base, changed = features(values), features(altered)
    pd.testing.assert_frame_equal(base.iloc[: h + 1], changed.iloc[: h + 1])
    assert not base.iloc[h + 1 :].equals(changed.iloc[h + 1 :])


def test_station_features_join():
    nom = pd.DataFrame(
        {"estacion_id": [1], "capacidad": [15], "latitud": [20.67], "longitud": [-103.35]}
    )
    df = pd.concat([_hourly([1]), _hourly([1], estacion_id=99)], ignore_index=True)
    out = add_station_features(df, nom)
    assert out.loc[out["estacion_id"] == 1, "capacidad"].iloc[0] == 15
    assert np.isnan(out.loc[out["estacion_id"] == 99, "capacidad"].iloc[0])


def test_assign_split_follows_config_cutoffs():
    df = pd.DataFrame(
        {
            "hora": pd.to_datetime(
                ["2024-12-31 23:00", "2025-01-01 00:00", "2025-06-30 23:00", "2025-07-01 00:00"]
            )
        }
    )
    assert assign_split(df)["split"].tolist() == ["train", "val", "val", "test"]
