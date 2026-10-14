"""Tests de humo para la ingesta de datos MiBici (sin red)."""

import json

import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from mibici import dataset as ds
from mibici.config import DATA_END, DATA_START, MANIFEST_PATH, UNIFIED_TRIPS_PATH
from mibici.dataset import (
    app,
    catalog,
    normalize_columns,
    parse_trip_datetime,
    read_csv_normalized,
)

runner = CliRunner()


def test_catalog_cover_post_pandemic_history():
    entries = catalog(
        start=(DATA_START.year, DATA_START.month),
        end=(DATA_END.year, DATA_END.month),
    )
    assert len(entries) == 41
    assert entries[0].label == "2023-05"
    assert entries[-1].label == "2026-09"


def test_normalize_columns_canonical_schema():
    raw = [
        "Viaje_Id",
        "Usuario_Id",
        "Genero",
        "Año_de_nacimiento",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]
    assert normalize_columns(raw) == [
        "viaje_id",
        "usuario_id",
        "genero",
        "anio_nacimiento",
        "inicio_viaje",
        "fin_viaje",
        "origen_id",
        "destino_id",
    ]


def test_normalize_columns_mangled_header_2023_04():
    """El header de 4 archivos llega dañado por doble conversión de charset."""
    raw = [
        '"Viaje_Id"',
        '"Usuario_Id"',
        '"Genero"',
        '"A}\u00e4e_nacimiento"',
        '"Inicio_del_viaje"',
        '"Fin_del_viaje"',
        '"Origen_Id"',
        '"Destino_Id"',
    ]
    assert "anio_nacimiento" in normalize_columns(raw)


def test_unified_parquet_exists_and_is_non_empty():
    if not UNIFIED_TRIPS_PATH.exists():
        pytest.skip("ingesta no ejecutada todavía")

    rows = pq.ParquetFile(UNIFIED_TRIPS_PATH).metadata.num_rows
    assert rows > 0
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert rows == manifest["total_rows"]


def test_manifest_is_versioned_and_complete():
    if not MANIFEST_PATH.exists():
        pytest.skip("ingesta no ejecutada todavía")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["n_months"] == 41
    assert manifest["first_month"] == "2023-05"
    assert manifest["last_month"] == "2026-09"
    assert manifest["total_rows"] > 0
    assert manifest["total_rows"] == manifest["unified_parquet"]["rows"]
    assert len(manifest["files"]) == 41
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])
    assert len(manifest["unified_parquet"]["sha256"]) == 64


def test_read_csv_normalized_handles_null_literals(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text(
        "Viaje_Id,Usuario_Id,Genero,Año_de_nacimiento,Inicio_del_viaje,"
        "Fin_del_viaje,Origen_Id,Destino_Id\n"
        '1,10,"NULL","NULL","2024-01-01 00:00:00","2024-01-01 00:10:00",1,2\n',
        encoding="utf-8",
    )
    df = read_csv_normalized(csv)
    assert df["genero"].isna().all()
    assert df["anio_nacimiento"].isna().all()
    assert pd.api.types.is_datetime64_any_dtype(df["inicio_viaje"])


def _make_fake_unified(tmp_path):
    """Parquet unificado falso de 1 fila en tmp_path."""
    fake = tmp_path / "mibici_viajes_test.parquet"
    pd.DataFrame(
        {
            "viaje_id": [1],
            "usuario_id": [10],
            "genero": ["M"],
            "anio_nacimiento": [1990],
            "inicio_viaje": pd.to_datetime(["2024-01-01"]),
            "fin_viaje": pd.to_datetime(["2024-01-01"]),
            "origen_id": [1],
            "destino_id": [2],
        }
    ).to_parquet(fake, index=False)
    return fake


def test_cli_aborts_without_force_when_parquet_exists(tmp_path, monkeypatch):
    """Sin --force, responder 'n' aborta sin descargar ni sobrescribir."""
    fake = _make_fake_unified(tmp_path)
    monkeypatch.setattr(ds, "UNIFIED_TRIPS_PATH", fake)
    downloaded = []
    monkeypatch.setattr(ds, "download_all", lambda *a, **k: downloaded.append(a))
    before = fake.read_bytes()

    result = runner.invoke(app, [], input="n\n")

    assert result.exit_code == 1
    assert downloaded == []
    assert fake.read_bytes() == before


def test_cli_force_skips_confirmation_and_runs(tmp_path, monkeypatch):
    """Con --force no se pregunta y la ingesta procede (con mocks, sin red)."""
    fake = _make_fake_unified(tmp_path)
    monkeypatch.setattr(ds, "UNIFIED_TRIPS_PATH", fake)
    monkeypatch.setattr(ds, "download_all", lambda *a, **k: None)
    monkeypatch.setattr(ds, "build_nomenclatura", lambda: 484)
    monkeypatch.setattr(ds, "build_unified_parquet", lambda entries: 1_000_000)
    monkeypatch.setattr(ds, "write_manifest", lambda *a, **k: None)

    result = runner.invoke(app, ["--force"])

    assert result.exit_code == 0, result.output


def test_parse_trip_datetime_dayfirst_format():
    """2023-11, 2024-09 y 2024-10 vienen como dd/mm/yyyy H:MM; no se deben invertir."""
    parsed = parse_trip_datetime(pd.Series(["01/11/2023 0:01", "16/11/2023 9:57", None]))
    assert parsed.iloc[0] == pd.Timestamp("2023-11-01 00:01")
    assert parsed.iloc[1] == pd.Timestamp("2023-11-16 09:57")
    assert pd.isna(parsed.iloc[2])


def test_parse_trip_datetime_iso_format():
    parsed = parse_trip_datetime(pd.Series(["2024-01-01 00:05:25", "2024-01-13 18:00:00"]))
    assert parsed.tolist() == [
        pd.Timestamp("2024-01-01 00:05:25"),
        pd.Timestamp("2024-01-13 18:00"),
    ]


def test_parse_trip_datetime_fails_on_unknown_format():
    with pytest.raises(ValueError, match="formato no reconocido"):
        parse_trip_datetime(pd.Series(["2024-01-01 00:05:25", "ayer a las 5"]))
