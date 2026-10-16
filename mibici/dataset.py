"""Ingesta reproducible de los datos abiertos de MiBici.

Descarga el histórico postpandemia de viajes (mayo 2023 – septiembre 2026,
41 archivos mensuales; ver `DATA_START` en config.py) y la nomenclatura de
estaciones desde el portal de datos abiertos de MiBici, normaliza esquemas y
codificación, y escribe un único archivo Parquet unificado más un manifiesto
versionable (SHA-256 por archivo) que permite reconstruir exactamente los
mismos datos en cualquier computadora.

Uso:
    python -m mibici.dataset                  # descarga todo y unifica
    python -m mibici.dataset --skip-download  # solo re-unifica lo ya descargado
    python -m mibici.dataset --limit 2        # prueba rápida con 2 meses
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm
import typer

from mibici.config import (
    DATA_DIR,
    DATA_END,
    DATA_START,
    MANIFEST_PATH,
    MIBICI_OPEN_DATA_URL,
    NOMENCLATURA_CSV_PATH,
    NOMENCLATURA_PARQUET_PATH,
    RAW_DATA_DIR,
    UNIFIED_TRIPS_PATH,
)

app = typer.Typer()

# ---------------------------------------------------------------------------
# Catálogo de archivos (extraído de https://mibici.net/es/datos-abiertos)
# El asset ID de cada mes es distinto y no es secuencial, por eso se lista
# de forma explícita: es parte del contrato de reproducibilidad de la ingesta.
# ---------------------------------------------------------------------------

_BASE = "https://mibici.net/site/assets/files"

# (año, mes) -> id de asset en el CMS de MiBici
_ASSET_IDS: dict[tuple[int, int], int] = {
    (2014, 12): 1079,
    (2015, 1): 1084,
    (2015, 2): 1097,
    (2015, 3): 1098,
    (2015, 4): 1099,
    (2015, 5): 1100,
    (2015, 6): 1101,
    (2015, 7): 1102,
    (2015, 8): 1103,
    (2015, 9): 1104,
    (2015, 10): 1105,
    (2015, 11): 1106,
    (2015, 12): 1107,
    (2016, 1): 1020,
    (2016, 2): 1021,
    (2016, 3): 1023,
    (2016, 4): 1029,
    (2016, 5): 1058,
    (2016, 6): 1080,
    (2016, 7): 1081,
    (2016, 8): 1082,
    (2016, 9): 1083,
    (2016, 10): 1108,
    (2016, 11): 1109,
    (2016, 12): 1110,
    (2017, 1): 1022,
    (2017, 2): 1111,
    (2017, 3): 1112,
    (2017, 4): 1115,
    (2017, 5): 1116,
    (2017, 6): 1119,
    (2017, 7): 1120,
    (2017, 8): 1122,
    (2017, 9): 1123,
    (2017, 10): 1124,
    (2017, 11): 1197,
    (2017, 12): 1198,
    (2018, 1): 1200,
    (2018, 2): 1201,
    (2018, 3): 1202,
    (2018, 4): 1203,
    (2018, 5): 1204,
    (2018, 6): 1205,
    (2018, 7): 1206,
    (2018, 8): 1207,
    (2018, 9): 1208,
    (2018, 10): 1209,
    (2018, 11): 1210,
    (2018, 12): 1211,
    (2019, 1): 1214,
    (2019, 2): 1217,
    (2019, 3): 1218,
    (2019, 4): 1219,
    (2019, 5): 1220,
    (2019, 6): 1221,
    (2019, 7): 1222,
    (2019, 8): 1223,
    (2019, 9): 1224,
    (2019, 10): 1225,
    (2019, 11): 1227,
    (2019, 12): 1228,
    (2020, 1): 1230,
    (2020, 2): 1231,
    (2020, 3): 1232,
    (2020, 4): 1235,
    (2020, 5): 1236,
    (2020, 6): 1237,
    (2020, 7): 1238,
    (2020, 8): 1239,
    (2020, 9): 1240,
    (2020, 10): 1241,
    (2020, 11): 1317,
    (2020, 12): 1318,
    (2021, 1): 1320,
    (2021, 2): 1323,
    (2021, 3): 1322,
    (2021, 4): 2129,
    (2021, 5): 3292,
    (2021, 6): 4572,
    (2021, 7): 5728,
    (2021, 8): 7073,
    (2021, 9): 8461,
    (2021, 10): 10088,
    (2021, 11): 11538,
    (2021, 12): 12780,
    (2022, 1): 14797,
    (2022, 2): 16831,
    (2022, 3): 19034,
    (2022, 4): 20338,
    (2022, 5): 21842,
    (2022, 6): 23473,
    (2022, 7): 25235,
    (2022, 8): 27484,
    (2022, 9): 29663,
    (2022, 10): 31507,
    (2022, 11): 33115,
    (2022, 12): 34432,
    (2023, 1): 36762,
    (2023, 2): 40197,
    (2023, 3): 43035,
    (2023, 4): 46758,
    (2023, 5): 46759,
    (2023, 6): 48116,
    (2023, 7): 49448,
    (2023, 8): 51295,
    (2023, 9): 53344,
    (2023, 10): 54935,
    (2023, 11): 57344,
    (2023, 12): 58715,
    (2024, 1): 61450,
    (2024, 2): 77251,
    (2024, 3): 77252,
    (2024, 4): 80028,
    (2024, 5): 81663,
    (2024, 6): 83901,
    (2024, 7): 85500,
    (2024, 8): 87665,
    (2024, 9): 89736,
    (2024, 10): 91758,
    (2024, 11): 93516,
    (2024, 12): 94962,
    (2025, 1): 97653,
    (2025, 2): 99879,
    (2025, 3): 101986,
    (2025, 4): 103813,
    (2025, 5): 105513,
    (2025, 6): 107088,
    (2025, 7): 108898,
    (2025, 8): 111021,
    (2025, 9): 113216,
    (2025, 10): 115104,
    (2025, 11): 116773,
    (2025, 12): 118153,
    (2026, 1): 120163,
    (2026, 2): 121761,
    (2026, 3): 123236,
    (2026, 4): 124927,
    (2026, 5): 125943,
    (2026, 6): 127734,
    (2026, 7): 129006,
    (2026, 8): 130362,
    (2026, 9): 131825,
}

# Nombres de archivo que se apartan del patrón datos_abiertos_AAAA_MM.csv
_NON_STANDARD_NAMES: dict[tuple[int, int], str] = {
    (2016, 1): "datos_abiertos_2016_01-1.csv",
    (2017, 4): "datos_abiertos_2017_04-1.csv",
}

# Nomenclatura de estaciones (archivo vigente a septiembre de 2026)
_NOMENCLATURA_URL = f"{_BASE}/1118/nomenclatura_2026_09.csv"

# Esquema canónico de columnas (verificado en archivos de 2014 a 2026).
# Entre años cambian comillas, encoding (UTF-8 vs latin-1) e incluso el
# header de 'Año_de_nacimiento' llega dañado en 4 archivos (p. ej. 2023-04
# tiene "A}äe_nacimiento" por una doble conversión de charset en la
# exportación). Por eso la identificación es por fragmento estable del
# nombre, no por igualdad exacta.
_CANONICAL_COLUMNS = [
    ("viaje_id", "viaje_id"),
    ("usuario_id", "usuario_id"),
    ("genero", "genero"),
    ("nacimiento", "anio_nacimiento"),  # "Año_de_nacimiento" y variantes dañadas
    ("inicio_del_viaje", "inicio_viaje"),
    ("fin_del_viaje", "fin_viaje"),
    ("origen", "origen_id"),
    ("destino", "destino_id"),
]


def normalize_columns(columns) -> list[str]:
    """Mapea los nombres de columna crudos al esquema canónico del proyecto."""
    normalized = []
    for raw in columns:
        name = raw.strip().strip('"').lower()
        mapped = None
        for fragment, canonical in _CANONICAL_COLUMNS:
            if fragment in name:
                mapped = canonical
                break
        if mapped is None:
            raise ValueError(f"Columna no reconocida: {raw!r}")
        normalized.append(mapped)
    return normalized


@dataclass(frozen=True)
class MonthFile:
    """Un archivo mensual de viajes publicado por MiBici."""

    year: int
    month: int
    url: str
    filename: str

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def catalog(
    start: tuple[int, int] | None = None, end: tuple[int, int] | None = None
) -> list[MonthFile]:
    """Regresa el catálogo de archivos mensuales, opcionalmente recortado."""
    entries = []
    for (year, month), asset_id in sorted(_ASSET_IDS.items()):
        if start and (year, month) < start:
            continue
        if end and (year, month) > end:
            continue
        filename = _NON_STANDARD_NAMES.get((year, month), f"datos_abiertos_{year}_{month:02d}.csv")
        entries.append(MonthFile(year, month, f"{_BASE}/{asset_id}/{filename}", filename))
    return entries


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash SHA-256 de un archivo, en bloques para no cargarlo en memoria."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


_HEADERS = {
    # El CMS de MiBici responde 406 al user-agent por defecto de requests
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
}


def download(url: str, dest: Path, timeout: int = 120) -> Path:
    """Descarga un archivo de forma reanudable: si ya existe, no lo repite."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout, headers=_HEADERS) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            fh.writelines(response.iter_content(chunk_size=1 << 20))
    tmp.rename(dest)
    return dest


def download_all(entries: list[MonthFile], workers: int = 6) -> None:
    """Descarga los CSV mensuales y la nomenclatura en paralelo."""
    NOMENCLATURA_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(e.url, RAW_DATA_DIR / "csv" / e.filename) for e in entries]
    jobs.append((_NOMENCLATURA_URL, NOMENCLATURA_CSV_PATH))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download, url, dest): dest for url, dest in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Descargando"):
            future.result()
    logger.info(f"Descarga completa: {len(entries)} meses + nomenclatura")


def parse_trip_datetime(values: pd.Series) -> pd.Series:
    """Convierte las fechas de un CSV mensual a datetime."""
    # 2023-11, 2024-09 y 2024-10 vienen como dd/mm/yyyy, los demas meses en ISO
    text = values.astype("string").str.strip()
    text = text.replace("", pd.NA)

    first = text.dropna()
    if len(first) == 0:
        return pd.to_datetime(text)

    if "/" in first.iloc[0]:
        parsed = pd.to_datetime(text, format="%d/%m/%Y %H:%M", errors="coerce")
    else:
        parsed = pd.to_datetime(text, format="ISO8601", errors="coerce")

    bad = parsed.isna() & text.notna()
    if bad.sum() > 0:
        raise ValueError(f"Hay {bad.sum()} fechas con formato desconocido")
    return parsed


def read_csv_normalized(path: Path) -> pd.DataFrame:
    """Lee un CSV mensual y lo deja con el esquema canónico del proyecto.

    Maneja las diferencias entre años: archivos con o sin comillas, encoding
    latin-1 vs UTF-8 (el header de 'Año_de_nacimiento' delata el encoding) y
    literales `NULL` sin comillas que pandas no detecta como NA en algunos
    años. Los tipos se fijan aquí para que el Parquet unificado sea estable.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    # Literales "NULL" sin comillas (p. ej. Genero en 2024-2026)
    df = df.replace("NULL", pd.NA)

    df.columns = normalize_columns(df.columns)

    expected = {canonical for _, canonical in _CANONICAL_COLUMNS}
    if set(df.columns) != expected or len(df.columns) != len(expected):
        raise ValueError(f"{path.name}: columnas inesperadas tras normalizar: {list(df.columns)}")

    df = df[[canonical for _, canonical in _CANONICAL_COLUMNS]]

    df["viaje_id"] = pd.to_numeric(df["viaje_id"], errors="coerce").astype("Int64")
    df["usuario_id"] = pd.to_numeric(df["usuario_id"], errors="coerce").astype("Int64")
    df["genero"] = df["genero"].astype("string").str.strip().str.upper()
    df["anio_nacimiento"] = pd.to_numeric(df["anio_nacimiento"], errors="coerce").astype("Int16")
    df["inicio_viaje"] = parse_trip_datetime(df["inicio_viaje"])
    df["fin_viaje"] = parse_trip_datetime(df["fin_viaje"])
    df["origen_id"] = pd.to_numeric(df["origen_id"], errors="coerce").astype("Int32")
    df["destino_id"] = pd.to_numeric(df["destino_id"], errors="coerce").astype("Int32")

    return df


def build_unified_parquet(entries: list[MonthFile], output: Path = UNIFIED_TRIPS_PATH) -> int:
    """Unifica todos los meses en un solo Parquet, mes a mes (streaming).

    Se procesa un mes a la vez con pyarrow.ParquetWriter para no necesitar
    decenas de GB de RAM: el histórico completo no cabe en un DataFrame de
    pandas. Regresa el total de filas escritas. `output` es la ruta destino:
    con --limit se manda a un archivo aparte.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = None
    total_rows = 0
    writer = None
    try:
        for entry in tqdm(entries, desc="Unificando"):
            csv_path = RAW_DATA_DIR / "csv" / entry.filename
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"{csv_path.name} no existe en {csv_path.parent}. "
                    "Corre la ingesta sin --skip-download para descargarlo: "
                    "python -m mibici.dataset"
                )
            df = read_csv_normalized(csv_path)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output, schema, compression="snappy")
            else:
                table = table.cast(schema)
            writer.write_table(table)
            total_rows += table.num_rows
        if writer is None:
            raise RuntimeError("No se escribió ningún mes")
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def build_nomenclatura() -> int:
    """Normaliza la nomenclatura de estaciones a Parquet. Regresa las filas."""
    if not NOMENCLATURA_CSV_PATH.exists():
        logger.info("Nomenclatura de estaciones no encontrada; descargándola…")
        download(_NOMENCLATURA_URL, NOMENCLATURA_CSV_PATH)
    try:
        nom = pd.read_csv(NOMENCLATURA_CSV_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        nom = pd.read_csv(NOMENCLATURA_CSV_PATH, encoding="latin-1")
    nom.columns = [c.strip().lower() for c in nom.columns]
    rename = {
        "id": "estacion_id",
        "name": "nombre",
        "latitude": "latitud",
        "longitude": "longitud",
        "dpcapacity": "capacidad",
    }
    nom = nom.rename(columns=rename)
    nom["estacion_id"] = pd.to_numeric(nom["estacion_id"], errors="coerce").astype("Int32")
    nom["latitud"] = pd.to_numeric(nom["latitud"], errors="coerce")
    nom["longitud"] = pd.to_numeric(nom["longitud"], errors="coerce")
    nom["capacidad"] = pd.to_numeric(nom["capacidad"], errors="coerce").astype("Int16")
    nom["nombre"] = nom["nombre"].astype("string")
    NOMENCLATURA_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    nom.to_parquet(NOMENCLATURA_PARQUET_PATH, index=False)
    return len(nom)


def output_paths(limit: int | None) -> tuple[Path, Path]:
    """Rutas de salida de la ingesta: (Parquet unificado, manifiesto).

    Con --limit (modo prueba) se escribe todo aparte para no pisar ni el
    Parquet unificado ni data/manifest.json, que es lo que se versiona.
    """
    if limit:
        return RAW_DATA_DIR / "mibici_viajes_limit.parquet", DATA_DIR / "manifest_limit.json"
    return UNIFIED_TRIPS_PATH, MANIFEST_PATH


def write_manifest(
    entries: list[MonthFile],
    total_rows: int,
    n_stations: int,
    unified_path: Path = UNIFIED_TRIPS_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    """Escribe el manifiesto de versionado de datos.

    Es el artefacto que se versiona en Git (data/manifest.json): contiene el
    hash SHA-256 de cada archivo descargado, los archivos unificados y el
    total de registros. Su propio hash se puede registrar en MLflow como
    parámetro `data_version` para ligar cada run a la versión exacta de datos.
    Con --limit se escribe en `manifest_path` (data/manifest_limit.json), que
    no se versiona.
    """
    files = []
    for entry in tqdm(entries, desc="Hasheando"):
        csv_path = RAW_DATA_DIR / "csv" / entry.filename
        files.append(
            {
                "month": entry.label,
                "url": entry.url,
                "filename": entry.filename,
                "size_bytes": csv_path.stat().st_size,
                "sha256": sha256_of(csv_path),
            }
        )

    manifest = {
        "dataset": "mibici_viajes",
        "source_url": MIBICI_OPEN_DATA_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "n_months": len(entries),
        "first_month": entries[0].label,
        "last_month": entries[-1].label,
        "total_rows": total_rows,
        "n_stations": n_stations,
        "unified_parquet": {
            "path": str(unified_path.relative_to(DATA_DIR.parent)),
            "size_bytes": unified_path.stat().st_size,
            "sha256": sha256_of(unified_path),
            "rows": total_rows,
        },
        "nomenclatura": {
            "url": _NOMENCLATURA_URL,
            "path": str(NOMENCLATURA_PARQUET_PATH.relative_to(DATA_DIR.parent)),
            "sha256": sha256_of(NOMENCLATURA_PARQUET_PATH),
            "rows": n_stations,
        },
        "files": files,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Hash del manifiesto completo: el "data_version" que se registra en MLflow
    manifest_hash = sha256_of(manifest_path)
    logger.success(f"Manifiesto: {manifest_path} (hash={manifest_hash[:12]}…)")


@app.command()
def main(
    start: str = typer.Option(None, help=f"Primer mes YYYY-MM (default: {DATA_START:%Y-%m})"),
    end: str = typer.Option(None, help=f"Último mes YYYY-MM (default: {DATA_END:%Y-%m})"),
    workers: int = typer.Option(6, help="Hilos de descarga en paralelo"),
    skip_download: bool = typer.Option(
        False, help="No descargar; usa los CSV ya presentes en data/raw/csv"
    ),
    limit: int = typer.Option(
        None, help="Solo los primeros N meses (para pruebas); escribe todo aparte"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Sobrescribe el Parquet unificado existente sin preguntar",
    ),
):
    """Descarga, normaliza y unifica el histórico de viajes de MiBici."""
    t0 = time.time()

    def parse(ym: str | None) -> tuple[int, int] | None:
        if ym is None:
            return None
        year, month = ym.split("-")
        return int(year), int(month)

    entries = catalog(
        start=parse(start) or (DATA_START.year, DATA_START.month),
        end=parse(end) or (DATA_END.year, DATA_END.month),
    )
    if limit:
        entries = entries[:limit]
    if not entries:
        raise typer.BadParameter("El rango seleccionado no contiene meses")

    unified_path, manifest_path = output_paths(limit)

    logger.info(f"Catálogo: {len(entries)} meses ({entries[0].label} .. {entries[-1].label})")

    if unified_path.exists() and not force:
        existing_rows = pq.ParquetFile(unified_path).metadata.num_rows
        size_mb = unified_path.stat().st_size / 1e6
        logger.warning(
            f"El Parquet unificado ya existe ({size_mb:,.0f} MB, "
            f"{existing_rows:,} filas): {unified_path.name}"
        )
        logger.warning(
            f"Regenerarlo vuelve a descargar los {len(entries)} CSVs y sobrescribe el archivo actual."
        )
        if not typer.confirm("¿Continuar?", default=False):
            logger.info(
                "Abortado: nada fue modificado. "
                "Para regenerar a la fuerza: python -m mibici.dataset --force"
            )
            raise typer.Exit(code=1)

    if not skip_download:
        download_all(entries, workers=workers)

    n_stations = build_nomenclatura()
    total_rows = build_unified_parquet(entries, unified_path)
    write_manifest(entries, total_rows, n_stations, unified_path, manifest_path)

    elapsed = time.time() - t0
    logger.success(
        f"Ingesta completa en {elapsed / 60:.1f} min: {total_rows:,} viajes, "
        f"{n_stations} estaciones -> {unified_path.name}"
    )
    if limit:
        logger.info(
            f"Modo --limit: se escribieron {unified_path.name} y {manifest_path.name} aparte; "
            "el Parquet unificado y data/manifest.json quedaron como estaban."
        )


if __name__ == "__main__":
    app()
