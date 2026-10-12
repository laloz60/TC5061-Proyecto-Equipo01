from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.debug(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Datos abiertos de MiBici
# ---------------------------------------------------------------------------
# Fuente: https://mibici.net/es/datos-abiertos
# Los archivos CSV mensuales y la nomenclatura de estaciones se descargan
# desde aquí; dataset.py unifica todo el histórico en un solo Parquet.

MIBICI_OPEN_DATA_URL = "https://mibici.net/es/datos-abiertos"

# Corte del equipo: solo datos POSTpandemia. La emergencia sanitaria por COVID-19
# terminó en mayo 2023 (OMS 2023-05-05; México mayo 2023), así que el dataset
# arranca en 2023-05: la operación de MiBici ya no está distorsionada por
# restricciones de movilidad y refleja el patrón actual del sistema.
DATA_START = date(2023, 5, 1)
DATA_END = date(2026, 9, 30)

# Archivos unificados que produce la ingesta (dataset.py)
UNIFIED_TRIPS_PATH = RAW_DATA_DIR / "mibici_viajes_2023_05_a_2026_09.parquet"
NOMENCLATURA_CSV_PATH = RAW_DATA_DIR / "nomenclatura_2026_09.csv"
NOMENCLATURA_PARQUET_PATH = RAW_DATA_DIR / "nomenclatura_estaciones.parquet"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Semilla y partición temporal (sin fuga de información del futuro)
# ---------------------------------------------------------------------------
# Regla de oro: toda partición es TEMPORAL. Nunca se mezclan meses futuros
# en el entrenamiento con el objetivo que se quiere predecir.
SEED = 42

# Ventana de entrenamiento: DATA_START .. TRAIN_END (inclusive)
TRAIN_END = date(2024, 12, 31)
# Ventana de validación: (TRAIN_END .. VAL_END]
VAL_END = date(2025, 6, 30)
# Ventana de prueba: (VAL_END .. DATA_END]
# Es decir: train 2023-05..2024-12 | val 2025-01..2025-06 | test 2025-07..2026-09

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
