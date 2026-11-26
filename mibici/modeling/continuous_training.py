"""Ciclo de Continuous Training disparado por data drift (Issue #199).

Flujo:
1. Lee el resultado de ``reports/drift_report.json`` (generado por ``-e drift``).
2. Si no hay drift a nivel dataset (y ``force_retrain`` es False), termina sin reentrenar.
3. Si hay drift (o ``force_retrain`` es True para prueba controlada), reentrena P2
   con la ventana extendida y registra una nueva versión en el Model Registry.
4. Compara la nueva versión contra ``@champion`` con la misma métrica de validación
   y mueve el alias ``champion`` únicamente si la candidata gana.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici import config

# Estados alineados con mibici/modeling/register.py (PR #208 y PR #211)
STAGE_PRODUCTION = "production"
STAGE_EVALUATION = "evaluation"
STAGE_ARCHIVED = "archived"

# Ventana extendida sin fuga temporal (deja el split test 2025-07..2026-10 intacto)
CT_TRAIN_END = "2025-03-31"
CT_VAL_START = "2025-04-01"
CT_VAL_END = "2025-06-30"

FEATURE_COLS_P2 = [
    "hora_del_dia",
    "dia_semana",
    "mes",
    "es_fin_de_semana",
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "salidas_media_7d",
    "salidas_prom_4sem",
    "neto_lag_1h",
    "neto_lag_24h",
    "neto_media_7d",
    "capacidad",
    "latitud",
    "longitud",
]

_CANDIDATAS_FECHA = ("hora", "fecha_hora", "timestamp", "fecha", "hora_inicio", "datetime")
_CANDIDATAS_TARGET = (
    "riesgo",
    "en_riesgo",
    "target_riesgo",
    "riesgo_desabasto_o_saturacion",
    "y_riesgo",
    "label_riesgo",
)


@dataclass(frozen=True)
class DriftStatus:
    dataset_drift: bool
    drifted_columns: int
    total_columns: int
    drift_share: float


@dataclass(frozen=True)
class PromotionDecision:
    retrained: bool
    promoted: bool
    candidate_version: str | None
    champion_version: str | None
    candidate_metric: float | None
    champion_metric: float | None
    reason: str


def _inferir_pregunta(model_name: str) -> str:
    """Devuelve P1, P2 o P3 a partir del nombre del modelo registrado."""
    nombre = model_name.lower()
    if "-p1-" in nombre:
        return "P1"
    if "-p3-" in nombre:
        return "P3"
    return "P2"


def _es_mayor_mejor(metric_name: str) -> bool:
    """Indica si valores más altos de la métrica representan un mejor modelo."""
    nombre = metric_name.lower()
    return not any(token in nombre for token in ("mae", "rmse", "mse", "loss", "error"))


def _leer_git_commit(proj_root: Path) -> str:
    """Obtiene el SHA actual leyendo .git/HEAD sin invocar subprocess (limpio en Bandit)."""
    head_path = proj_root / ".git" / "HEAD"
    if not head_path.exists():
        return "desconocido"

    contenido = head_path.read_text(encoding="utf-8").strip()
    if not contenido.startswith("ref: "):
        return contenido

    ref = contenido.removeprefix("ref: ").strip()
    ref_path = proj_root / ".git" / ref
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()

    packed_refs = proj_root / ".git" / "packed-refs"
    if packed_refs.exists():
        for linea in packed_refs.read_text(encoding="utf-8").splitlines():
            if not linea.startswith("#") and linea.endswith(f" {ref}"):
                return linea.split(" ", 1)[0].strip()

    return "desconocido"


def _sha256_corto(path: Path) -> str:
    """Calcula los primeros 16 caracteres del SHA-256 de un archivo."""
    if not path.exists():
        return "?"
    hasher = hashlib.sha256()
    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            hasher.update(bloque)
    return hasher.hexdigest()[:16]


def _detectar_columna(df: pd.DataFrame, candidatas: tuple[str, ...], tipo: str) -> str:
    """Encuentra el nombre de columna esperado dentro del DataFrame de features."""
    for col in candidatas:
        if col in df.columns:
            return col
    if tipo == "fecha":
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
    disponibles = ", ".join(df.columns)
    raise KeyError(
        f"No se encontró la columna de {tipo} en features_estacion_hora.parquet. "
        f"Esperadas: {candidatas}. Disponibles: {disponibles}"
    )


def leer_reporte_drift(report_path: Path) -> DriftStatus:
    """Extrae el estado de drift a nivel dataset desde reports/drift_report.json."""
    if not report_path.exists():
        raise FileNotFoundError(
            f"No existe {report_path}. Ejecuta primero: mlflow run . -e drift --env-manager=uv"
        )

    data: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))

    # Soporta formato resumido propio o estructura estándar de Evidently.
    if "dataset_drift" in data:
        return DriftStatus(
            dataset_drift=bool(data["dataset_drift"]),
            drifted_columns=int(data.get("number_of_drifted_columns", 0)),
            total_columns=int(data.get("number_of_columns", 0)),
            drift_share=float(data.get("share_of_drifted_columns", 0.0)),
        )

    for metric in data.get("metrics", []):
        result = metric.get("result", {})
        if "dataset_drift" in result:
            return DriftStatus(
                dataset_drift=bool(result["dataset_drift"]),
                drifted_columns=int(result.get("number_of_drifted_columns", 0)),
                total_columns=int(result.get("number_of_columns", 0)),
                drift_share=float(result.get("share_of_drifted_columns", 0.0)),
            )

    raise KeyError(f"No se encontró 'dataset_drift' en {report_path}")


def obtener_champion_actual(
    client: MlflowClient,
    model_name: str,
    metric_name: str,
) -> tuple[str | None, float | None]:
    """Obtiene la versión actual @champion y su métrica de validación desde el Registry."""
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return None, None

    tags = mv.tags or {}

    # 1. Formato oficial del Registry (Issue #191): metrica_seleccion + valor_metrica
    if tags.get("metrica_seleccion") == metric_name and "valor_metrica" in tags:
        return str(mv.version), float(tags["valor_metrica"])

    # 2. Compatibilidad si la métrica se guardó directamente como llave de tag
    if metric_name in tags:
        return str(mv.version), float(tags[metric_name])

    # 3. Respaldo: lee la métrica del run de origen
    run = client.get_run(mv.run_id)
    metric_val = run.data.metrics.get(metric_name)
    return str(mv.version), (float(metric_val) if metric_val is not None else None)


def reentrenar_y_registrar_candidata(
    client: MlflowClient,
    model_name: str,
    metric_name: str,
) -> tuple[str, float]:
    """Reentrena P2 con ventana extendida y registra nueva versión sin mover alias."""
    proj_root = getattr(config, "PROJ_ROOT", Path.cwd())
    data_dir = getattr(config, "DATA_DIR", proj_root / "data")
    processed_dir = getattr(config, "PROCESSED_DATA_DIR", data_dir / "processed")
    seed = int(getattr(config, "SEED", 42))

    features_path = processed_dir / "features_estacion_hora.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"No existe {features_path}. Ejecuta primero: mlflow run . -e features"
        )

    df = pd.read_parquet(features_path)
    fecha_col = _detectar_columna(df, _CANDIDATAS_FECHA, "fecha")
    target_col = _detectar_columna(df, _CANDIDATAS_TARGET, "target")
    feature_cols = [c for c in FEATURE_COLS_P2 if c in df.columns]
    if len(feature_cols) != len(FEATURE_COLS_P2):
        faltantes = sorted(set(FEATURE_COLS_P2) - set(feature_cols))
        raise KeyError(f"Faltan columnas de features de P2: {faltantes}")

    fechas = pd.to_datetime(df[fecha_col])
    cols_necesarias = [*feature_cols, target_col]
    df_train = df.loc[fechas <= CT_TRAIN_END, cols_necesarias].dropna()
    df_val = df.loc[(fechas >= CT_VAL_START) & (fechas <= CT_VAL_END), cols_necesarias].dropna()

    if df_train.empty or df_val.empty:
        raise ValueError(
            f"Split vacío para Continuous Training: train={len(df_train)}, val={len(df_val)}"
        )

    X_train, y_train = df_train[feature_cols], df_train[target_col].astype(int)
    X_val, y_val = df_val[feature_cols], df_val[target_col].astype(int)

    modelo = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    modelo.fit(X_train, y_train)
    probs_val = modelo.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, probs_val))

    commit = _leer_git_commit(proj_root)
    manifest_sha = _sha256_corto(data_dir / "manifest.json")
    parquet_sha = _sha256_corto(features_path)

    with mlflow.start_run(run_name="P2-continuous-training-candidate", nested=True) as run:
        mlflow.log_params(
            {
                "model_family": "LogisticRegression",
                "C": 0.1,
                "train_end": CT_TRAIN_END,
                "val_start": CT_VAL_START,
                "val_end": CT_VAL_END,
                "train_rows": len(df_train),
                "val_rows": len(df_val),
                "seed": seed,
                "data_manifest_sha256": manifest_sha,
                "data_parquet_sha256": parquet_sha,
            }
        )
        mlflow.set_tags(
            {
                "equipo": "01",
                "pregunta": _inferir_pregunta(model_name),
                "stage_of_work": "continuous_training",
                "commit": commit,
            }
        )
        mlflow.log_metric(metric_name, val_auc)

        muestra = X_val.head(5)
        signature = infer_signature(muestra, modelo.predict(muestra))
        model_info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=signature,
            input_example=muestra,
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    # Registra nueva versión sin mover @champion ni @ab-candidate
    mv = mlflow.register_model(model_uri, model_name)
    descripcion = (
        f"Candidato de Continuous Training ({_inferir_pregunta(model_name)}) "
        f"entrenado hasta {CT_TRAIN_END} y validado en {CT_VAL_START}..{CT_VAL_END}. "
        f"Run de origen: P2-continuous-training-candidate ({run_id}). "
        f"Commit: {commit}. Datos: manifest {manifest_sha} / features {parquet_sha}. "
        f"{metric_name}={val_auc:.4f}. Registrado con mibici/modeling/continuous_training.py."
    )
    client.update_model_version(model_name, mv.version, description=descripcion)

    return str(mv.version), val_auc


def decidir_promocion(
    champion_metric: float | None,
    candidate_metric: float,
    higher_is_better: bool = True,
) -> bool:
    """Devuelve True solo si la candidata supera estrictamente al @champion vigente."""
    if champion_metric is None:
        return True
    if higher_is_better:
        return candidate_metric > champion_metric
    return candidate_metric < champion_metric


def ejecutar_continuous_training(
    drift_report: Path,
    model_name: str,
    metric_name: str = "val_roc_auc",
    force_retrain: bool = False,
) -> PromotionDecision:
    """Orquesta el ciclo completo de Continuous Training."""
    drift = leer_reporte_drift(drift_report)
    print(
        f"[CT] Drift reportado: dataset_drift={drift.dataset_drift} "
        f"({drift.drifted_columns}/{drift.total_columns} columnas, "
        f"{drift.drift_share:.1%})"
    )

    client = MlflowClient()
    champion_ver, champion_val = obtener_champion_actual(client, model_name, metric_name)

    if not drift.dataset_drift and not force_retrain:
        return PromotionDecision(
            retrained=False,
            promoted=False,
            candidate_version=None,
            champion_version=champion_ver,
            candidate_metric=None,
            champion_metric=champion_val,
            reason="Sin drift a nivel dataset: se conserva @champion actual.",
        )

    cand_ver, cand_val = reentrenar_y_registrar_candidata(client, model_name, metric_name)
    gana = decidir_promocion(
        champion_val,
        cand_val,
        higher_is_better=_es_mayor_mejor(metric_name),
    )
    pregunta = _inferir_pregunta(model_name)

    client.set_model_version_tag(model_name, cand_ver, "pregunta", pregunta)
    client.set_model_version_tag(model_name, cand_ver, "metrica_seleccion", metric_name)
    client.set_model_version_tag(model_name, cand_ver, "valor_metrica", f"{cand_val:.4f}")

    champ_str = f"{champion_val:.4f}" if champion_val is not None else "N/A"
    if gana:
        client.set_registered_model_alias(model_name, "champion", cand_ver)
        client.set_model_version_tag(model_name, cand_ver, "stage", STAGE_PRODUCTION)
        if champion_ver is not None:
            client.set_model_version_tag(model_name, champion_ver, "stage", STAGE_ARCHIVED)
        reason = (
            f"Candidata v{cand_ver} ({metric_name}={cand_val:.4f}) superó a "
            f"@champion v{champion_ver} ({champ_str}): alias promovido."
        )
    else:
        client.set_model_version_tag(model_name, cand_ver, "stage", STAGE_EVALUATION)
        reason = (
            f"Candidata v{cand_ver} ({metric_name}={cand_val:.4f}) no superó a "
            f"@champion v{champion_ver} ({champ_str}): se mantiene @champion."
        )

    return PromotionDecision(
        retrained=True,
        promoted=gana,
        candidate_version=cand_ver,
        champion_version=cand_ver if gana else champion_ver,
        candidate_metric=cand_val,
        champion_metric=champion_val,
        reason=reason,
    )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "si", "sí"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous Training disparado por drift.")
    parser.add_argument("--drift-report", type=Path, default=Path("reports/drift_report.json"))
    parser.add_argument("--model-name", type=str, default="mibici-equipo01-p2-riesgo")
    parser.add_argument("--metric", type=str, default="val_roc_auc")
    parser.add_argument("--force-retrain", type=str, default="false")
    args = parser.parse_args()

    force = _parse_bool(args.force_retrain)
    run_kwargs: dict[str, Any] = {}
    if not os.environ.get("MLFLOW_RUN_ID"):
        run_kwargs["run_name"] = "P2-continuous-training-check"

    with mlflow.start_run(**run_kwargs):
        decision = ejecutar_continuous_training(
            drift_report=args.drift_report,
            model_name=args.model_name,
            metric_name=args.metric,
            force_retrain=force,
        )
        mlflow.log_params(
            {
                "model_name": args.model_name,
                "metric": args.metric,
                "force_retrain": force,
            }
        )
        mlflow.set_tags(
            {
                "ct_retrained": str(decision.retrained),
                "ct_promoted": str(decision.promoted),
                "ct_reason": decision.reason,
            }
        )

    print(f"[CT] Resultado: {decision.reason}")
    print(f"[CT] @champion vigente en {args.model_name}: v{decision.champion_version}")


if __name__ == "__main__":
    main()
