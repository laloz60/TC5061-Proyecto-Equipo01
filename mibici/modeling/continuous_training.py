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
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

STAGE_PRODUCTION = "production"
STAGE_EVALUACION = "evaluacion"
STAGE_ARCHIVED = "archived"


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
    """Reentrena P2 con ventana extendida y registra nueva versión sin mover @champion.

    TODO(#193): Conectar aquí la función real de entrenamiento con ventana
    extendida de ``mibici/modeling/ab_testing.py`` (o ``train.py``) y el helper de
    linaje de ``mibici/modeling/register.py`` para escribir descripción y tags.
    """
    raise NotImplementedError(
        "Conectar con el entrenamiento de ventana extendida de ab_testing.py "
        "y devolver (candidate_version, candidate_val_metric)."
    )


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
    gana = decidir_promocion(champion_val, cand_val, higher_is_better=True)
    pregunta = _inferir_pregunta(model_name)

    client.set_model_version_tag(model_name, cand_ver, "pregunta", pregunta)
    client.set_model_version_tag(model_name, cand_ver, "metrica_seleccion", metric_name)
    client.set_model_version_tag(model_name, cand_ver, "valor_metrica", f"{cand_val:.4f}")

    if gana:
        client.set_registered_model_alias(model_name, "champion", cand_ver)
        client.set_model_version_tag(model_name, cand_ver, "stage", STAGE_PRODUCTION)
        if champion_ver is not None:
            client.set_model_version_tag(model_name, champion_ver, "stage", STAGE_ARCHIVED)
        reason = (
            f"Candidata v{cand_ver} ({metric_name}={cand_val:.4f}) superó a "
            f"@champion v{champion_ver} ({champion_val}): alias promovido."
        )
    else:
        client.set_model_version_tag(model_name, cand_ver, "stage", STAGE_EVALUACION)
        reason = (
            f"Candidata v{cand_ver} ({metric_name}={cand_val:.4f}) no superó a "
            f"@champion v{champion_ver} ({champion_val:.4f}): se mantiene @champion."
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

    decision = ejecutar_continuous_training(
        drift_report=args.drift_report,
        model_name=args.model_name,
        metric_name=args.metric,
        force_retrain=_parse_bool(args.force_retrain),
    )

    with mlflow.start_run(run_name="P2-continuous-training-check"):
        mlflow.log_params(
            {
                "model_name": args.model_name,
                "metric": args.metric,
                "force_retrain": _parse_bool(args.force_retrain),
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
