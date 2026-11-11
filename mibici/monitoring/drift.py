"""Módulo para el monitoreo de data drift y calidad de datos con Evidently."""

import json
from pathlib import Path

from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from loguru import logger
import pandas as pd

from mibici import config


def generate_drift_report():
    logger.info("Cargando features procesadas para el análisis de Drift y Calidad...")

    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "features_estacion_hora.parquet")
    # Definición explícita de los periodos de monitoreo.
    # Referencia: 2023-05-01 a 2024-12-31.
    # Actual: 2025-01-01 a 2026-09-30.
    reference_start = pd.Timestamp("2023-05-01")
    train_end = pd.Timestamp(config.TRAIN_END)
    current_start = pd.Timestamp("2025-01-01")
    current_end = pd.Timestamp("2026-09-30 23:59:59")

    logger.info(f"Periodo de referencia: {reference_start.date()} a {train_end.date()}")
    logger.info(f"Periodo actual: {current_start.date()} a {current_end.date()}")

    # Validar la existencia de la columna temporal.
    if "hora" not in df.columns:
        raise ValueError("No se encontró la columna temporal 'hora' en el dataset.")

    df["hora"] = pd.to_datetime(df["hora"])

    # Separación por fechas reales, no por posición.
    reference_df = df[(df["hora"] >= reference_start) & (df["hora"] <= train_end)].copy()

    current_df = df[(df["hora"] >= current_start) & (df["hora"] <= current_end)].copy()

    logger.info(f"Registros de referencia: {len(reference_df)}")
    logger.info(f"Registros actuales: {len(current_df)}")

    # Validar que ambos periodos tengan registros.
    if reference_df.empty:
        raise ValueError(
            "El conjunto de referencia está vacío. Verifica el periodo 2023-05 a 2024-12."
        )

    if current_df.empty:
        raise ValueError("El conjunto actual está vacío. Verifica el periodo 2025-01 a 2026-09.")

    # Eliminar columnas temporales y de partición.
    # No son variables que deban utilizarse en el análisis estadístico.
    columns_to_drop = ["hora", "split"]

    reference_df = reference_df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    current_df = current_df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    # Garantizar que ambos conjuntos tengan las mismas variables.
    if set(reference_df.columns) != set(current_df.columns):
        raise ValueError("Los conjuntos de referencia y actual tienen diferentes columnas.")

    current_df = current_df[reference_df.columns]

    # Muestreo reproducible para controlar el consumo de memoria.
    sample_size = min(
        30000,
        len(reference_df),
        len(current_df),
    )

    reference_data = reference_df.sample(
        n=sample_size,
        random_state=config.SEED,
    )

    current_data = current_df.sample(
        n=sample_size,
        random_state=config.SEED,
    )

    logger.info(f"Muestras seleccionadas: {sample_size} registros por conjunto.")

    logger.info("Calculando Data Drift y Data Quality con Evidently...")

    # Reporte completo de Drift y Calidad de Datos.
    drift_report = Report(
        metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
        ]
    )

    drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    # Guardar reportes HTML y JSON.
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    html_path = reports_dir / "drift_report.html"
    json_path = reports_dir / "drift_report.json"

    drift_report.save_html(str(html_path))
    drift_report.save_json(str(json_path))

    # Leer resultados del reporte para identificar Data Drift.
    with open(json_path, "r", encoding="utf-8") as file:
        drift_result = json.load(file)

    dataset_drift = False
    share_of_drifted_columns = 0.0

    for metric in drift_result.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            result = metric["result"]

            dataset_drift = result["dataset_drift"]
            share_of_drifted_columns = result["share_of_drifted_columns"]
            break

    logger.success(f"Reportes HTML y JSON guardados exitosamente en {reports_dir}.")

    # Resultado del monitoreo.
    if dataset_drift:
        logger.warning(
            f"¡ALERTA DE DRIFT! El {share_of_drifted_columns * 100:.1f}% de las variables cambió."
        )

        logger.warning(
            "Recomendación de Sistema: Se requiere disparar el pipeline de Continuous Training."
        )
    else:
        logger.info(
            f"Sistema estable. Solo el {share_of_drifted_columns * 100:.1f}% de las variables cambió."
        )


if __name__ == "__main__":
    generate_drift_report()
