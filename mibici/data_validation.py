"""Validación de datos con DeepChecks: tipos, faltantes, duplicados y drift.

Corre en su propio entorno (requirements-validation.txt).

Faltantes y duplicados se cuentan sobre todas las filas. Los chequeos de DeepChecks
y el drift usan una muestra con SEED, porque son más de 15 millones de viajes.

El resumen solo guarda conteos y puntajes agregados: ningún valor individual de
`usuario_id`, `genero` o `anio_nacimiento` llega a reports/.
"""

import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SEED,
    TRAIN_END,
    UNIFIED_TRIPS_PATH,
    VAL_END,
)

app = typer.Typer(help="Validación de datos con DeepChecks — MiBici Equipo 01")

FEATURES_PATH = PROCESSED_DATA_DIR / "features_estacion_hora.parquet"
OUTPUT_DIR = REPORTS_DIR / "data_validation"

# Puntaje de drift a partir del cual una columna se marca (default de DeepChecks)
MAX_DRIFT = 0.2

TIPOS_VIAJES = {
    "viaje_id": "integer",
    "usuario_id": "integer",
    "genero": "string",
    "anio_nacimiento": "integer",
    "inicio_viaje": "datetime",
    "fin_viaje": "datetime",
    "origen_id": "integer",
    "destino_id": "integer",
}

# Se dejan fuera los identificadores de viaje y usuario: compararlos entre periodos
# no dice nada del comportamiento de los datos.
DRIFT_VIAJES_CAT = ["genero", "origen_id", "destino_id", "hora_inicio", "dia_semana"]
DRIFT_VIAJES_NUM = ["anio_nacimiento", "duracion_min"]

DRIFT_FEATURES_CAT = ["hora_del_dia", "dia_semana", "es_fin_de_semana", "mes"]
DRIFT_FEATURES_NUM = [
    "salidas",
    "llegadas",
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "salidas_media_7d",
    "salidas_prom_4sem",
    "neto_lag_1h",
    "neto_lag_24h",
    "neto_media_7d",
    "capacidad",
]


def tipo_de_columna(serie: pd.Series) -> str:
    """Clasifica el dtype de una columna en entero, texto, fecha u otro."""
    if pd.api.types.is_integer_dtype(serie):
        return "integer"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "datetime"
    if pd.api.types.is_string_dtype(serie):
        return "string"
    return str(serie.dtype)


def revisar_tipos(df: pd.DataFrame, esperados: dict[str, str]) -> dict:
    """Compara las columnas y sus tipos contra el esquema esperado."""
    encontrados = {col: tipo_de_columna(df[col]) for col in df.columns}
    diferencias = {
        col: {"esperado": esperado, "encontrado": encontrados.get(col, "ausente")}
        for col, esperado in esperados.items()
        if encontrados.get(col) != esperado
    }
    extra = sorted(set(encontrados) - set(esperados))
    return {"diferencias": diferencias, "columnas_extra": extra, "ok": not diferencias}


def contar_faltantes(df: pd.DataFrame) -> dict:
    """Faltantes por columna, en conteo y porcentaje."""
    return {
        col: {"nulos": int(nulos), "porcentaje": round(100 * nulos / len(df), 6)}
        for col, nulos in df.isna().sum().items()
    }


def contar_duplicados(df: pd.DataFrame) -> dict:
    """Duplicados exactos, de `viaje_id` y de un mismo viaje con distinto id."""
    sin_id = [col for col in df.columns if col != "viaje_id"]
    return {
        "filas_exactas": int(df.duplicated().sum()),
        "viaje_id": int(df["viaje_id"].duplicated().sum()),
        "mismo_viaje_con_otro_id": int(df.duplicated(subset=sin_id).sum()),
    }


def separar_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte los viajes con las ventanas de config.py; validación queda fuera."""
    fecha = df["inicio_viaje"].dt.date
    return df[fecha <= TRAIN_END], df[fecha > VAL_END]


def preparar_drift_viajes(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas para el drift de viajes, con tipos que DeepChecks puede comparar."""
    duracion = (df["fin_viaje"] - df["inicio_viaje"]).dt.total_seconds() / 60
    return pd.DataFrame(
        {
            "genero": df["genero"].astype(object).where(df["genero"].notna(), None),
            "origen_id": df["origen_id"].astype("int64"),
            "destino_id": df["destino_id"].astype("int64"),
            "hora_inicio": df["inicio_viaje"].dt.hour,
            "dia_semana": df["inicio_viaje"].dt.dayofweek,
            "anio_nacimiento": df["anio_nacimiento"].astype("float64"),
            "duracion_min": duracion,
        }
    )


def muestrear(df: pd.DataFrame, n_filas: int) -> pd.DataFrame:
    """Muestra reproducible; si hay menos filas que `n_filas`, las usa todas."""
    return df.sample(n=min(n_filas, len(df)), random_state=SEED)


def correr_drift(train, test, cat_features, num_features, html_path: Path | None = None) -> dict:
    """Drift por columna entre train y test con FeatureDrift de DeepChecks."""
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import FeatureDrift

    columnas = cat_features + num_features
    ds_train = Dataset(train[columnas], cat_features=cat_features, features=columnas)
    ds_test = Dataset(test[columnas], cat_features=cat_features, features=columnas)

    resultado = FeatureDrift(n_samples=len(train) + len(test)).run(ds_train, ds_test)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        resultado.save_as_html(str(html_path))

    columnas_drift = {
        col: {"puntaje": round(float(datos["Drift score"]), 4), "metodo": datos["Method"]}
        for col, datos in resultado.value.items()
    }
    marcadas = sorted(c for c, d in columnas_drift.items() if d["puntaje"] >= MAX_DRIFT)
    return {"umbral": MAX_DRIFT, "columnas": columnas_drift, "marcadas": marcadas}


def correr_chequeos_muestra(muestra: pd.DataFrame) -> dict:
    """Tipos mezclados, nulos disfrazados y duplicados con DeepChecks sobre la muestra."""
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import DataDuplicates, MixedDataTypes, MixedNulls

    # sin los ids: DataDuplicates nunca encontraría una fila repetida con viaje_id
    columnas = [c for c in muestra.columns if c not in ("viaje_id", "usuario_id")]
    datos = muestra[columnas].copy()
    datos["genero"] = datos["genero"].astype(object).where(datos["genero"].notna(), None)
    for col in ("anio_nacimiento", "origen_id", "destino_id"):
        datos[col] = datos[col].astype("float64")
    ds = Dataset(datos, cat_features=["genero"])

    mezclados = MixedDataTypes().run(ds).value
    nulos = MixedNulls().run(ds).value["columns"]
    return {
        "filas": len(datos),
        "columnas_con_tipos_mezclados": sorted(c for c, v in mezclados.items() if v),
        "nulos_por_columna": {
            col: {str(tipo): int(info["count"]) for tipo, info in tipos.items()}
            for col, tipos in nulos.items()
            if tipos
        },
        "porcentaje_filas_duplicadas": round(100 * float(DataDuplicates().run(ds).value), 4),
    }


def validar(sample_rows: int, viajes_path: Path, features_path: Path, html_dir: Path | None):
    """Corre los cuatro chequeos y devuelve el resumen."""
    logger.info(f"Leyendo {viajes_path.name}...")
    viajes = pd.read_parquet(viajes_path)
    resumen = {
        "filas": len(viajes),
        "semilla": SEED,
        "filas_por_muestra": sample_rows,
        "tipos": revisar_tipos(viajes, TIPOS_VIAJES),
        "faltantes": contar_faltantes(viajes),
        "duplicados": contar_duplicados(viajes),
    }

    logger.info("Chequeos de DeepChecks sobre la muestra...")
    resumen["deepchecks_muestra"] = correr_chequeos_muestra(muestrear(viajes, sample_rows))

    logger.info("Drift de viajes entre train y test...")
    train, test = separar_train_test(viajes)
    resumen["drift_viajes"] = correr_drift(
        preparar_drift_viajes(muestrear(train, sample_rows)),
        preparar_drift_viajes(muestrear(test, sample_rows)),
        DRIFT_VIAJES_CAT,
        DRIFT_VIAJES_NUM,
        html_dir / "drift_viajes.html" if html_dir else None,
    )
    del viajes, train, test

    if features_path.exists():
        logger.info("Drift de features entre train y test...")
        columnas = ["split"] + DRIFT_FEATURES_CAT + DRIFT_FEATURES_NUM
        features = pd.read_parquet(features_path, columns=columnas)
        resumen["drift_features"] = correr_drift(
            muestrear(features[features["split"] == "train"], sample_rows),
            muestrear(features[features["split"] == "test"], sample_rows),
            DRIFT_FEATURES_CAT,
            DRIFT_FEATURES_NUM,
            html_dir / "drift_features.html" if html_dir else None,
        )
    else:
        logger.warning("Sin features: corre `python -m mibici.features` para validarlas.")
    return resumen


@app.command()
def main(
    sample_rows: int = typer.Option(200_000, help="Filas por muestra y por ventana."),
    viajes_path: Path = typer.Option(UNIFIED_TRIPS_PATH, help="Parquet unificado de viajes."),
    features_path: Path = typer.Option(FEATURES_PATH, help="Parquet de features."),
    output_dir: Path = typer.Option(OUTPUT_DIR, help="Carpeta del resumen."),
    html: bool = typer.Option(False, help="Guarda también el HTML de drift (no se versiona)."),
) -> None:
    """Valida los datos y guarda el resumen en reports/data_validation/resumen.json."""
    if not viajes_path.exists():
        logger.error(f"No existe {viajes_path}: corre `python -m mibici.dataset`.")
        raise typer.Exit(code=1)

    resumen = validar(sample_rows, viajes_path, features_path, output_dir if html else None)

    output_dir.mkdir(parents=True, exist_ok=True)
    salida = output_dir / "resumen.json"
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Resumen en {salida}")

    if not resumen["tipos"]["ok"]:
        logger.error(f"Esquema inesperado: {resumen['tipos']['diferencias']}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
