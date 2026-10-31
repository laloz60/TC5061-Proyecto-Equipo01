"""Validación de datos: esquema y corrida completa con DeepChecks en su propio entorno."""

from datetime import datetime, timedelta
import json
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mibici.config import PROJ_ROOT
from mibici.data_validation import TIPOS_VIAJES, revisar_tipos

CARPETA_BIN = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
PYTHON_VALIDACION = PROJ_ROOT / ".venv-validacion" / CARPETA_BIN
USUARIO_BASE = 987654


def escribir_viajes(path):
    """Parquet con 200 viajes en train, 200 en test y una fila duplicada a propósito."""
    fechas = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(200)]
    fechas += [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(200)]
    fechas.append(fechas[-1])
    ids = list(range(400)) + [399]

    tabla = pa.table(
        {
            "viaje_id": pa.array(ids, type=pa.int64()),
            "usuario_id": pa.array([USUARIO_BASE + i for i in ids], type=pa.int64()),
            "genero": pa.array(["M" if i % 2 else "F" for i in ids]),
            "anio_nacimiento": pa.array([1980 + i % 30 for i in ids], type=pa.int16()),
            "inicio_viaje": pa.array(fechas, type=pa.timestamp("us")),
            "fin_viaje": pa.array(
                [f + timedelta(minutes=10) for f in fechas], type=pa.timestamp("us")
            ),
            "origen_id": pa.array([i % 5 for i in ids], type=pa.int32()),
            "destino_id": pa.array([i % 7 for i in ids], type=pa.int32()),
        }
    )
    pq.write_table(tabla, path)


def test_revisar_tipos_detecta_columna_ausente():
    df = pd.DataFrame({"viaje_id": [1, 2]})

    resultado = revisar_tipos(df, TIPOS_VIAJES)

    assert not resultado["ok"]
    assert resultado["diferencias"]["genero"]["encontrado"] == "ausente"


def test_validacion_completa_con_deepchecks(tmp_path):
    if not PYTHON_VALIDACION.exists():
        pytest.skip("falta el entorno: corre `make validation_env`")

    viajes = tmp_path / "viajes.parquet"
    escribir_viajes(viajes)

    corrida = subprocess.run(
        [
            str(PYTHON_VALIDACION),
            "-W",
            "ignore",
            "-m",
            "mibici.data_validation",
            "--sample-rows",
            "500",
            "--viajes-path",
            str(viajes),
            "--features-path",
            str(tmp_path / "no_existe.parquet"),
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert corrida.returncode == 0, corrida.stderr

    texto = (tmp_path / "resumen.json").read_text(encoding="utf-8")
    resumen = json.loads(texto)

    assert resumen["tipos"]["ok"]
    assert resumen["duplicados"]["filas_exactas"] == 1
    assert "duracion_min" in resumen["drift_viajes"]["columnas"]
    assert "usuario_id" not in resumen["drift_viajes"]["columnas"]
    # el resumen no debe traer valores individuales de usuarios
    assert str(USUARIO_BASE) not in texto
