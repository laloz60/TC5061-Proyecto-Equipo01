from datetime import timedelta
import hashlib
import json
import subprocess

from mibici.config import DATA_END, DATA_START, MANIFEST_PATH, PROJ_ROOT, TRAIN_END, VAL_END

EXPERIMENT_NAME = "mibici-equipo01"


def git_commit():
    try:
        salida = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT)
        return salida.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def split_params():
    """Fechas de train, val y test (de config.py)."""
    return {
        "train_start": str(DATA_START),
        "train_end": str(TRAIN_END),
        "val_start": str(TRAIN_END + timedelta(days=1)),
        "val_end": str(VAL_END),
        "test_start": str(VAL_END + timedelta(days=1)),
        "test_end": str(DATA_END),
    }


def data_params():
    """Hash del manifiesto y del parquet de viajes para saber con que datos se entreno."""
    if not MANIFEST_PATH.exists():
        return {"data_manifest_sha256": "sin_manifiesto", "data_parquet_sha256": "sin_manifiesto"}

    contenido = MANIFEST_PATH.read_bytes()
    manifest = json.loads(contenido)
    return {
        "data_manifest_sha256": hashlib.sha256(contenido).hexdigest(),
        "data_parquet_sha256": manifest["unified_parquet"]["sha256"],
    }


def run_tags(pregunta):
    return {"equipo": "01", "pregunta": pregunta, "commit": git_commit()}
