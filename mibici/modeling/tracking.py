"""Metadatos de trazabilidad comunes a todos los runs de MLflow (#22).

Cada run del experimento `mibici-equipo01` debe poder reproducirse: qué
código (commit), qué datos (hash del manifiesto y del Parquet unificado) y
qué ventanas temporales se usaron. Todo se lee de `config.py` y de
`data/manifest.json`; nada va escrito a mano en el código de entrenamiento.
"""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import subprocess

from mibici.config import DATA_END, DATA_START, MANIFEST_PATH, PROJ_ROOT, TRAIN_END, VAL_END

EXPERIMENT_NAME = "mibici-equipo01"


def git_commit() -> str:
    """SHA del commit actual; 'desconocido' si no hay repositorio git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def split_params() -> dict[str, str]:
    """Ventanas train/val/test de `config.py` como params de MLflow."""
    return {
        "train_start": DATA_START.isoformat(),
        "train_end": TRAIN_END.isoformat(),
        "val_start": (TRAIN_END + timedelta(days=1)).isoformat(),
        "val_end": VAL_END.isoformat(),
        "test_start": (VAL_END + timedelta(days=1)).isoformat(),
        "test_end": DATA_END.isoformat(),
    }


def data_params() -> dict[str, str]:
    """Versión de datos: hash del manifiesto completo y del Parquet unificado."""
    if not MANIFEST_PATH.exists():
        return {"data_manifest_sha256": "sin_manifiesto", "data_parquet_sha256": "sin_manifiesto"}
    raw = MANIFEST_PATH.read_bytes()
    manifest = json.loads(raw)
    return {
        "data_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "data_parquet_sha256": manifest["unified_parquet"]["sha256"],
    }


def run_tags(pregunta: str) -> dict[str, str]:
    """Tags obligatorios de la convención del README."""
    return {"equipo": "01", "pregunta": pregunta, "commit": git_commit()}
