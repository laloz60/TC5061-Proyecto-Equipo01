"""A/B testing del modelo P2 (Riesgo): V1 (@champion) vs V2 (issue #138).

Compara dos versiones del clasificador de riesgo sobre EXACTAMENTE el mismo
conjunto de evaluacion: el split ``test`` (2025-07-01..2026-10-31), con las
mismas features, la misma etiqueta y la misma semilla (``config.SEED``). La
unica diferencia entre los dos brazos es el modelo y su umbral, de modo que la
diferencia de metricas es atribuible a la V2 y no a los datos.

Pasos:
    train-v2  Entrena la V2 en la ventana extendida (train + val) y la
              registra en el Model Registry con el alias ``ab-candidate``
              (nunca toca ``champion``: esa decision la da este reporte).
    compare   Evalua V1 y V2 en el set de prueba y escribe la tabla
              comparativa, el test de McNemar y el reporte en ``reports/``.
    run       Los dos pasos en secuencia.

Uso:
    python -m mibici.modeling.ab_testing run
"""

from datetime import date
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import (
    DATA_START,
    FIGURES_DIR,
    MANIFEST_PATH,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    RISK_ALPHA,
    RISK_HORIZON_HOURS,
    SEED,
    VAL_END,
)
from mibici.features import add_risk_label
from mibici.modeling.regression import FEATURES
from mibici.modeling.tracking import EXPERIMENT_NAME, data_params, run_tags, split_params

app = typer.Typer(help="A/B testing P2: V1 (@champion) vs V2 sobre el mismo set de evaluacion.")

# modelo en el registry: la V2 entra como NUEVA VERSION de este mismo nombre
NOMBRE_MODELO = "mibici-equipo01-p2-riesgo"
# alias de la V2: distinto de 'candidate' para que register.py no la promueva
# por accidente al registrar P2 (alla solo busca stage_of_work='candidate')
ALIAS_V2 = "ab-candidate"
ALIAS_V1 = "champion"

# misma malla de seleccion que train.py (p2 logreg), para que la V2 solo
# cambie por la ventana de datos y no por una busqueda de hiperparametros distinta
C_GRID = (0.1, 1.0, 10.0)
UMBRAL_GRID = np.linspace(0.3, 0.7, 41)

# holdout interno de seleccion: el ultimo trimestre de la ventana V2. Ahi se
# eligen C y el umbral; el set de prueba NO se toca hasta compare.
HOLDOUT_INICIO = pd.Timestamp("2025-04-01")

REPORT_PATH = REPORTS_DIR / "ab_testing_p2.md"
CSV_PATH = REPORTS_DIR / "ab_testing_p2.csv"
FIGURES_PATH = FIGURES_DIR / "ab_testing_p2"


# ---------------------------------------------------------------------------
# utilidades puras (sin MLflow: son las que cubren los tests)
# ---------------------------------------------------------------------------


def metricas_clasificacion(y_true, y_prob, umbral) -> dict[str, float]:
    """Metricas de P2 con un umbral concreto (mismas que train.py)."""
    y_pred = (np.asarray(y_prob) >= umbral).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def seleccionar_umbral(y_true, y_prob, grid=UMBRAL_GRID) -> tuple[float, float]:
    """Umbral con mejor F1 macro en el holdout (mismo criterio que train.py)."""
    mejor_umbral, mejor_f1 = 0.5, -1.0
    for umbral in grid:
        y_pred = (np.asarray(y_prob) >= umbral).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > mejor_f1:
            mejor_umbral, mejor_f1 = float(umbral), float(f1)
    return mejor_umbral, mejor_f1


def mcnemar_exacto(y_true, pred_v1, pred_v2) -> dict[str, float]:
    """Test exacto de McNemar (binomtest bilateral) sobre pares discordantes.

    Solo importa en que difieren las predicciones: ``solo_v1`` son filas donde
    V1 acierta y V2 falla, ``solo_v2`` al reves. Con p < 0.05 la diferencia de
    exactitud entre versiones es estadisticamente significativa.
    """
    y_true = np.asarray(y_true)
    acierta_v1 = np.asarray(pred_v1) == y_true
    acierta_v2 = np.asarray(pred_v2) == y_true
    solo_v1 = int((acierta_v1 & ~acierta_v2).sum())
    solo_v2 = int((~acierta_v1 & acierta_v2).sum())
    discordantes = solo_v1 + solo_v2
    pvalue = 1.0 if discordantes == 0 else float(binomtest(solo_v1, discordantes, 0.5).pvalue)
    return {
        "solo_v1": solo_v1,
        "solo_v2": solo_v2,
        "discordantes": discordantes,
        "pvalue": pvalue,
    }


def tabla_comparativa(metricas_v1: dict, metricas_v2: dict) -> pd.DataFrame:
    """Tabla metrica -> V1, V2 y diferencia (V2 - V1)."""
    filas = [
        {
            "metrica": nombre,
            "v1": metricas_v1[nombre],
            "v2": metricas_v2[nombre],
            "delta": metricas_v2[nombre] - metricas_v1[nombre],
        }
        for nombre in metricas_v1
    ]
    return pd.DataFrame(filas)


def _texto_pvalue(pvalue: float) -> str:
    """p-value legible: con cientos de miles de discordantes puede sublimar a 0.0."""
    if pvalue < 1e-4:
        return "p < 0.0001"
    return f"p = {pvalue:.4f}"


def veredicto(tabla: pd.DataFrame, mcnemar: dict, sensibilidad: pd.DataFrame) -> str:
    """Conclusion en una oracion, cruzando la comparacion operativa con la
    sensibilidad al umbral para no atribuirle al modelo una mejora del corte.

    Con 97% de negativos, McNemar compara el despliegue completo (modelo +
    umbral): por eso el empate a igual umbral es lo que decide si la V2 es
    mejor MODELO o solo una configuracion mejor.
    """
    delta_f1 = float(tabla.loc[tabla["metrica"] == "f1_macro", "delta"].iloc[0])
    delta_auc = float(tabla.loc[tabla["metrica"] == "roc_auc", "delta"].iloc[0])
    f1 = dict(zip(sensibilidad["configuracion"], sensibilidad["f1_macro"]))
    delta_igual = f1["V2 @ umbral de la V1"] - f1["V1 @ su umbral"]
    p = _texto_pvalue(mcnemar["pvalue"])

    if mcnemar["pvalue"] >= 0.05:
        return (
            f"Sin diferencia significativa (McNemar {p} >= 0.05): los cambios "
            "no justifican mover el alias champion."
        )

    if abs(delta_igual) < 0.005:
        return (
            f"La V2 gana en el despliegue (F1 macro {delta_f1:+.4f}, McNemar {p}), "
            f"PERO a igual umbral los modelos empatan ({delta_igual:+.4f}): la "
            "mejora la explica el umbral (0.70 vs 0.50), no el reentrenamiento. "
            "Promover la V2 es inocuo y deja el corte afinado; el ROC AUC, que no "
            f"depende del umbral, apenas se mueve ({delta_auc:+.4f})."
        )

    if delta_igual > 0 and delta_auc > 0:
        return (
            f"V2 supera a V1 también a igual umbral (F1 macro {delta_igual:+.4f}) y "
            f"en el despliegue ({delta_f1:+.4f}, McNemar {p}): el reentrenamiento "
            "sí mejoró el modelo. Conviene promover la V2."
        )

    return (
        f"V1 se mantiene como modelo: a igual umbral la V2 pierde "
        f"({delta_igual:+.4f}) y en el despliegue tampoco gana en F1 macro "
        f"({delta_f1:+.4f}) pese a McNemar ({p})."
    )


def lectura_de_la_diferencia(tabla: pd.DataFrame, umbral_v1: float, umbral_v2: float) -> str:
    """Explica DE DONDE sale la diferencia: del modelo o del umbral.

    Es la parte que convierte la tabla en un argumento: si el AUC apenas se
    mueve, el ranking de riesgo es practicamente el mismo y la mejora del F1
    macro viene del umbral re-optimizado en la ventana extendida.
    """
    delta = {f.metrica: f.delta for f in tabla.itertuples(index=False)}
    frases = []

    if abs(delta["roc_auc"]) < 0.01:
        frases.append(
            f"El ROC AUC prácticamente no se mueve ({delta['roc_auc']:+.4f}): las dos "
            "versiones ordenan casi igual las horas de riesgo, así que el modelo "
            "aprendió un patrón muy parecido."
        )
    else:
        frases.append(
            f"El ROC AUC cambia {delta['roc_auc']:+.4f}: la V2 ordena mejor las "
            "horas de riesgo que la V1."
        )

    if delta["recall"] < 0 < delta["precision"]:
        frases.append(
            f"La V2 opera con umbral {umbral_v2:.2f} frente a {umbral_v1:.2f} de la V1: "
            f"gana precisión ({delta['precision']:+.4f}) a costa de recall "
            f"({delta['recall']:+.4f}), y de ahí sale el F1 macro "
            f"({delta['f1_macro']:+.4f})."
        )
    elif delta["f1_macro"] > 0:
        frases.append(
            f"La V2 gana en precisión ({delta['precision']:+.4f}), recall "
            f"({delta['recall']:+.4f}) y F1 macro ({delta['f1_macro']:+.4f}) con su "
            f"umbral {umbral_v2:.2f} frente al {umbral_v1:.2f} de la V1."
        )
    return " ".join(frases)


def tabla_sensibilidad(y_true, prob_v1, prob_v2, umbral_v1, umbral_v2) -> pd.DataFrame:
    """Cruza modelo x umbral para separar el efecto del modelo del del umbral.

    Si la V2 sigue ganando con el MISMO umbral de la V1, la mejora es del
    modelo; si no, la mejora viene de haber re-optimizado el umbral en la
    ventana extendida. Es la comprobacion que evita atribuirle a la V2 una
    mejora que en realidad dio el corte de probabilidad.
    """
    filas = []
    for configuracion, prob, umbral in (
        ("V1 @ su umbral", prob_v1, umbral_v1),
        ("V2 @ su umbral", prob_v2, umbral_v2),
        ("V1 @ umbral de la V2", prob_v1, umbral_v2),
        ("V2 @ umbral de la V1", prob_v2, umbral_v1),
    ):
        metricas = metricas_clasificacion(y_true, prob, umbral)
        filas.append({"configuracion": configuracion, "umbral": umbral, **metricas})
    return pd.DataFrame(filas)


def _conclusion_sensibilidad(sensibilidad: pd.DataFrame) -> str:
    """En una frase, si la V2 gana tambien a igual umbral que la V1.

    Usa el mismo margen de empate (0.005) que veredicto, para que las dos
    secciones del reporte digan lo mismo.
    """
    f1 = dict(zip(sensibilidad["configuracion"], sensibilidad["f1_macro"]))
    umbral_v1 = float(
        sensibilidad.loc[sensibilidad["configuracion"] == "V1 @ su umbral", "umbral"].iloc[0]
    )
    v1, v2 = f1["V1 @ su umbral"], f1["V2 @ umbral de la V1"]
    delta = v2 - v1

    if abs(delta) < 0.005:
        cuerpo = f"los modelos empatan en F1 macro ({v1:.4f} vs {v2:.4f})"
        motivo = (
            "la mejora de la V2 al operar con su propio umbral viene de la "
            "calibración del corte, no de un mejor ranking"
        )
    elif delta > 0:
        cuerpo = f"la V2 gana en F1 macro ({v2:.4f} vs {v1:.4f})"
        motivo = "la V2 es mejor modelo aun sin ventaja de umbral"
    else:
        cuerpo = f"la V1 gana en F1 macro ({v1:.4f} vs {v2:.4f})"
        motivo = "el modelo de la V1 sigue siendo mejor"
    return f"A igual umbral ({umbral_v1:.2f}) {cuerpo}: {motivo}."


def renderizar_reporte(
    tabla: pd.DataFrame, mcnemar: dict, contexto: dict, sensibilidad: pd.DataFrame
) -> str:
    """Arma el Markdown del A/B: metodologia, tabla, McNemar y veredicto."""
    lineas = [
        "# A/B testing P2 (Riesgo): V1 champion vs V2 ventana extendida",
        "",
        f"- Fecha: {contexto['fecha']} | commit `{contexto['commit']}`",
        f"- Semilla: `{contexto['seed']}` (misma para entrenar y evaluar)",
        (
            f"- **Set único de evaluación**: split `test` "
            f"({contexto['eval_inicio']} a {contexto['eval_fin']}), "
            f"{contexto['eval_filas']:,} filas, {contexto['eval_pct_riesgo']:.2%} en riesgo"
        ),
        (
            f"- V1: `{contexto['v1_alias']}` (versión {contexto['v1_version']}, "
            f"run `{contexto['v1_run'][:8]}`, C={contexto['v1_C']}, "
            f"umbral={contexto['v1_umbral']:.3f})"
        ),
        (
            f"- V2: `{contexto['v2_alias']}` (versión {contexto['v2_version']}, "
            f"run `{contexto['v2_run'][:8]}`, C={contexto['v2_C']}, "
            f"umbral={contexto['v2_umbral']:.3f})"
        ),
        "",
        "## Metodología",
        "",
        (
            f"1. La V2 se entrenó con la ventana extendida `{DATA_START}..{VAL_END}` "
            f"(splits train + val); C y el umbral se eligieron en un holdout interno "
            f"(`{HOLDOUT_INICIO.date()}..{VAL_END}`) y el modelo final se reajustó "
            "sobre toda la ventana."
        ),
        (
            f"2. V1 y V2 se evaluaron sobre las MISMAS {contexto['eval_filas']:,} "
            "filas del split `test`: mismas features, misma etiqueta, misma semilla. "
            "La única variable del experimento es el modelo (y su umbral propio)."
        ),
        (
            "3. ROC AUC es umbral-independiente; precisión, recall y F1 macro usan "
            "el umbral que le corresponde a cada versión (el suyo, elegido en "
            "validación, no en el set de prueba)."
        ),
        (
            "4. La diferencia de aciertos se contrasta con el **test exacto de "
            "McNemar** sobre los pares discordantes (binomtest bilateral)."
        ),
        "",
        "## Tabla comparativa",
        "",
        "| métrica | V1 (champion) | V2 (ab-candidate) | delta (V2 - V1) |",
        "|---|---:|---:|---:|",
    ]
    for fila in tabla.itertuples(index=False):
        lineas.append(f"| {fila.metrica} | {fila.v1:.4f} | {fila.v2:.4f} | {fila.delta:+.4f} |")

    lineas += [
        "",
        "## Lectura de la diferencia",
        "",
        lectura_de_la_diferencia(tabla, contexto["v1_umbral"], contexto["v2_umbral"]),
        "",
        "## Sensibilidad al umbral (modelo × umbral)",
        "",
        "| configuración | umbral | roc_auc | precision | recall | f1_macro |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fila in sensibilidad.itertuples(index=False):
        lineas.append(
            f"| {fila.configuracion} | {fila.umbral:.2f} | {fila.roc_auc:.4f} | "
            f"{fila.precision:.4f} | {fila.recall:.4f} | {fila.f1_macro:.4f} |"
        )

    lineas += [
        "",
        _conclusion_sensibilidad(sensibilidad),
        "",
        "## Significancia (McNemar exacto)",
        "",
        f"- Pares donde solo acierta V1: **{mcnemar['solo_v1']:,}**",
        f"- Pares donde solo acierta V2: **{mcnemar['solo_v2']:,}**",
        f"- Pares discordantes: {mcnemar['discordantes']:,}",
        (
            f"- Test bilateral de McNemar: **{_texto_pvalue(mcnemar['pvalue'])}** "
            "(umbral de decisión: 0.05)"
        ),
        "",
        "## Veredicto",
        "",
        veredicto(tabla, mcnemar, sensibilidad),
        "",
    ]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# datos
# ---------------------------------------------------------------------------


def cargar_con_etiqueta(features_path: Path) -> pd.DataFrame:
    """Lee las features, agrega la etiqueta de riesgo y descarta filas incompletas.

    Se etiqueta TODO el frame antes de cortar por split (igual que train.py):
    la etiqueta necesita las 3 horas siguientes de cada estacion.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Falta {features_path}. Corre antes: python -m mibici.features")
    columnas = ["estacion_id", "salidas", "llegadas", "split", "hora"] + FEATURES
    df = pd.read_parquet(features_path, columns=columnas)
    df = add_risk_label(df)
    df = df.dropna(subset=FEATURES + ["riesgo"])
    # la etiqueta ya esta calculada: estas 3 columnas solo pesan memoria
    return df.drop(columns=["estacion_id", "salidas", "llegadas"])


def _construir_modelo(c_val: float):
    """Pipeline de la V2: el mismo esqueleto que train.py (logreg + scaler)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_val, class_weight="balanced", random_state=SEED, max_iter=1000),
    )


# ---------------------------------------------------------------------------
# paso 1: entrenar y registrar la V2
# ---------------------------------------------------------------------------


def _registrar_v2(cliente: MlflowClient, run_id: str, umbral: float, c_val: float):
    """Version nueva del MISMO modelo, con alias ``ab-candidate`` (no champion)."""
    version = None
    for candidata in cliente.search_model_versions(f"name='{NOMBRE_MODELO}'"):
        if candidata.run_id == run_id:
            version = candidata
            break
    if version is None:
        version = mlflow.register_model(f"runs:/{run_id}/model", NOMBRE_MODELO)

    descripcion = (
        f"Candidata al A/B testing de P2 (issue #138). Run de origen: {run_id}. "
        f"Ventana extendida {DATA_START}..{VAL_END} (train+val), C={c_val}, "
        f"umbral={umbral:.3f} elegidos en holdout interno "
        f"{HOLDOUT_INICIO.date()}..{VAL_END}. "
        f"Commit: {run_tags('P2')['commit']}. "
        "Evaluada contra la V1 en reports/ab_testing_p2.md. "
        "Alias: ab-candidate (el alias champion no se mueve aqui)."
    )
    cliente.update_model_version(NOMBRE_MODELO, str(version.version), description=descripcion)
    cliente.set_registered_model_alias(NOMBRE_MODELO, ALIAS_V2, str(version.version))
    return version


def entrenar_v2(features_path: Path) -> str:
    """Entrena la V2 en la ventana extendida y la registra. Devuelve el run_id."""
    logger.info("A/B testing: entrenamiento de la V2 de P2")
    df = cargar_con_etiqueta(features_path)

    ventana = df.loc[df["split"].isin(["train", "val"])]
    holdout = ventana.loc[ventana["hora"] >= HOLDOUT_INICIO]
    ajuste = ventana.loc[ventana["hora"] < HOLDOUT_INICIO]
    del df

    logger.info(
        f"V2: ajuste={len(ajuste):,} filas ({ajuste['hora'].min().date()}.."
        f"{(ajuste['hora'].max()).date()}) | holdout={len(holdout):,} filas "
        f"({holdout['hora'].min().date()}..{holdout['hora'].max().date()})"
    )

    X_ajuste, y_ajuste = ajuste[FEATURES], ajuste["riesgo"].astype(int)
    X_holdout, y_holdout = holdout[FEATURES], holdout["riesgo"].astype(int)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="P2-v2-ventana-extendida") as run:
        tags = run_tags("P2")
        tags.update(
            {
                "model_family": "logreg",
                "stage_of_work": "ab_candidate",
                "ab_role": "v2",
            }
        )
        mlflow.set_tags(tags)
        mlflow.log_params(split_params())
        mlflow.log_params(data_params())
        mlflow.log_params(
            {
                "modelo": "logreg",
                "random_state": SEED,
                "risk_horizon_hours": RISK_HORIZON_HOURS,
                "risk_alpha": RISK_ALPHA,
                "ventana_v2": f"{DATA_START}..{VAL_END}",
                "holdout_seleccion": f"{HOLDOUT_INICIO.date()}..{VAL_END}",
                "C_grid": ",".join(str(c) for c in C_GRID),
            }
        )

        # seleccion de C y umbral en el holdout interno (NUNCA en el test)
        mejor_f1, mejor_C, mejor_umbral = -1.0, C_GRID[0], 0.5
        for c_val in C_GRID:
            modelo = _construir_modelo(c_val)
            modelo.fit(X_ajuste, y_ajuste)
            prob = modelo.predict_proba(X_holdout)[:, 1]
            umbral, f1 = seleccionar_umbral(y_holdout, prob)
            metricas = metricas_clasificacion(y_holdout, prob, umbral)
            mlflow.log_metrics({f"holdout_{k}_C{c_val}": v for k, v in metricas.items()})
            mlflow.log_metric(f"holdout_umbral_C{c_val}", umbral)
            logger.info(f"C={c_val}: holdout F1 macro={f1:.4f} con umbral={umbral:.3f}")
            if f1 > mejor_f1:
                mejor_f1, mejor_C, mejor_umbral = f1, c_val, umbral

        # reajuste sobre TODA la ventana con el C elegido en el holdout
        X_ventana, y_ventana = ventana[FEATURES], ventana["riesgo"].astype(int)
        modelo_v2 = _construir_modelo(mejor_C)
        modelo_v2.fit(X_ventana, y_ventana)

        prob_final = modelo_v2.predict_proba(X_holdout)[:, 1]
        metricas_finales = metricas_clasificacion(y_holdout, prob_final, mejor_umbral)
        mlflow.log_metrics({f"holdout_final_{k}": v for k, v in metricas_finales.items()})
        mlflow.log_param("best_C", mejor_C)
        mlflow.log_param("umbral_final", round(mejor_umbral, 3))

        muestra = X_ventana.head(1000)
        signature = infer_signature(muestra, modelo_v2.predict(muestra))
        mlflow.sklearn.log_model(
            modelo_v2,
            name="model",
            signature=signature,
            input_example=muestra.head(5),
        )
        if MANIFEST_PATH.exists():
            from mlflow.data import from_pandas

            ds = from_pandas(X_ventana, source=str(MANIFEST_PATH), name="mibici_p2_v2")
            mlflow.log_input(ds, context="training")

        run_id = run.info.run_id
        del X_ajuste, y_ajuste, X_holdout, y_holdout, X_ventana, y_ventana

    version = _registrar_v2(MlflowClient(), run_id, mejor_umbral, mejor_C)
    logger.success(
        f"V2 registrada: {NOMBRE_MODELO} v{version.version} -> alias {ALIAS_V2} "
        f"(C={mejor_C}, umbral={mejor_umbral:.3f}, F1 holdout={mejor_f1:.4f})"
    )
    return run_id


# ---------------------------------------------------------------------------
# paso 2: comparar V1 vs V2 en el MISMO set de evaluacion
# ---------------------------------------------------------------------------


def _version_y_umbral(cliente: MlflowClient, alias: str):
    """Carga la version del alias y su umbral (parametro del run que la entreno)."""
    version = cliente.get_model_version_by_alias(NOMBRE_MODELO, alias)
    params = cliente.get_run(version.run_id).data.params
    umbral = float(params.get("umbral_final", 0.5))
    return version, umbral, params


def _graficar_roc(y_true, prob_v1, prob_v2, salida: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, prob in (("V1 champion", prob_v1), ("V2 ab-candidate", prob_v2)):
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_true, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="aleatorio")
    ax.set_xlabel("Falsos positivos")
    ax.set_ylabel("Verdaderos positivos")
    ax.set_title("Curva ROC - A/B testing P2 (mismo set de evaluacion)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def _graficar_metricas(tabla: pd.DataFrame, salida: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tabla))
    ax.bar(x - 0.2, tabla["v1"], width=0.4, label="V1 champion")
    ax.bar(x + 0.2, tabla["v2"], width=0.4, label="V2 ab-candidate")
    ax.set_xticks(x, tabla["metrica"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Metricas P2: V1 vs V2 sobre el mismo set de prueba")
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def comparar(features_path: Path) -> pd.DataFrame:
    """Evalua V1 y V2 en el split test y escribe reporte, CSV, figuras y run."""
    logger.info("A/B testing: comparacion V1 (@champion) vs V2 (@ab-candidate)")
    cliente = MlflowClient()
    v1, umbral_v1, params_v1 = _version_y_umbral(cliente, ALIAS_V1)
    v2, umbral_v2, params_v2 = _version_y_umbral(cliente, ALIAS_V2)

    modelo_v1 = mlflow.sklearn.load_model(f"runs:/{v1.run_id}/model")
    modelo_v2 = mlflow.sklearn.load_model(f"runs:/{v2.run_id}/model")

    # UN solo set de evaluacion para los dos brazos: el split test
    df = cargar_con_etiqueta(features_path)
    test = df.loc[df["split"] == "test"]
    del df
    X_test, y_test = test[FEATURES], test["riesgo"].astype(int)
    logger.info(
        f"Set de evaluacion: test {test['hora'].min()}..{test['hora'].max()} "
        f"({len(test):,} filas, {y_test.mean():.2%} en riesgo)"
    )

    prob_v1 = modelo_v1.predict_proba(X_test)[:, 1]
    prob_v2 = modelo_v2.predict_proba(X_test)[:, 1]
    pred_v1 = (prob_v1 >= umbral_v1).astype(int)
    pred_v2 = (prob_v2 >= umbral_v2).astype(int)

    metricas_v1 = metricas_clasificacion(y_test, prob_v1, umbral_v1)
    metricas_v2 = metricas_clasificacion(y_test, prob_v2, umbral_v2)
    tabla = tabla_comparativa(metricas_v1, metricas_v2)
    sensibilidad = tabla_sensibilidad(y_test, prob_v1, prob_v2, umbral_v1, umbral_v2)
    mcnemar = mcnemar_exacto(y_test, pred_v1, pred_v2)
    logger.info(
        "McNemar: solo_V1="
        f"{mcnemar['solo_v1']:,} solo_V2={mcnemar['solo_v2']:,} "
        f"p={mcnemar['pvalue']:.6g}"
    )

    contexto = {
        "fecha": date.today().isoformat(),
        "commit": run_tags("P2")["commit"],
        "seed": SEED,
        "eval_inicio": str(test["hora"].min().date()),
        "eval_fin": str(test["hora"].max().date()),
        "eval_filas": len(test),
        "eval_pct_riesgo": float(y_test.mean()),
        "v1_alias": f"models:/{NOMBRE_MODELO}@{ALIAS_V1}",
        "v1_version": v1.version,
        "v1_run": v1.run_id,
        "v1_C": params_v1.get("best_C", "?"),
        "v1_umbral": umbral_v1,
        "v2_alias": f"models:/{NOMBRE_MODELO}@{ALIAS_V2}",
        "v2_version": v2.version,
        "v2_run": v2.run_id,
        "v2_C": params_v2.get("best_C", "?"),
        "v2_umbral": umbral_v2,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    roc_path = FIGURES_PATH / "roc_v1_vs_v2.png"
    barras_path = FIGURES_PATH / "metricas_v1_vs_v2.png"
    _graficar_roc(y_test, prob_v1, prob_v2, roc_path)
    _graficar_metricas(tabla, barras_path)

    reporte = renderizar_reporte(tabla, mcnemar, contexto, sensibilidad)
    REPORT_PATH.write_text(reporte, encoding="utf-8")
    tabla.to_csv(CSV_PATH, index=False)
    logger.success(f"Reporte A/B en {REPORT_PATH}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="P2-ab-test-v1-vs-v2"):
        tags = run_tags("P2")
        tags.update({"model_family": "logreg", "stage_of_work": "ab_test"})
        mlflow.set_tags(tags)
        mlflow.log_params(
            {
                "eval_split": "test",
                "eval_inicio": contexto["eval_inicio"],
                "eval_fin": contexto["eval_fin"],
                "eval_filas": len(test),
                "random_state": SEED,
                "v1_alias": contexto["v1_alias"],
                "v1_run": v1.run_id,
                "v1_version": str(v1.version),
                "v1_umbral": umbral_v1,
                "v2_alias": contexto["v2_alias"],
                "v2_run": v2.run_id,
                "v2_version": str(v2.version),
                "v2_umbral": umbral_v2,
            }
        )
        metricas = {}
        for nombre in metricas_v1:
            metricas[f"v1_{nombre}"] = metricas_v1[nombre]
            metricas[f"v2_{nombre}"] = metricas_v2[nombre]
            metricas[f"delta_{nombre}"] = metricas_v2[nombre] - metricas_v1[nombre]
        # efecto cruzado modelo x umbral: la V2 con el umbral de la V1
        for fila in sensibilidad.itertuples(index=False):
            clave = fila.configuracion.replace(" @ ", "_a_").replace(" ", "_")
            metricas[f"sens_{clave}_f1_macro"] = fila.f1_macro
            metricas[f"sens_{clave}_roc_auc"] = fila.roc_auc
        metricas["mcnemar_pvalue"] = mcnemar["pvalue"]
        metricas["mcnemar_discordantes"] = float(mcnemar["discordantes"])
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(str(REPORT_PATH))
        mlflow.log_artifact(str(CSV_PATH))
        for figura in sorted(FIGURES_PATH.glob("*.png")):
            mlflow.log_artifact(str(figura), artifact_path="ab_figures")

    logger.success(veredicto(tabla, mcnemar, sensibilidad))
    return tabla


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command("train-v2")
def train_v2(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features_estacion_hora.parquet",
        "--features-path",
        help="Parquet de features estacion x hora.",
    ),
) -> None:
    """Entrena la V2 de P2 y la registra con el alias ab-candidate."""
    entrenar_v2(features_path)
    raise typer.Exit()


@app.command()
def compare(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features_estacion_hora.parquet",
        "--features-path",
        help="Parquet de features estacion x hora.",
    ),
) -> None:
    """Compara V1 (@champion) vs V2 (@ab-candidate) en el mismo set de prueba."""
    comparar(features_path)
    raise typer.Exit()


@app.command()
def run(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features_estacion_hora.parquet",
        "--features-path",
        help="Parquet de features estacion x hora.",
    ),
) -> None:
    """Entrena la V2 y despues compara las dos versiones (flujo completo)."""
    entrenar_v2(features_path)
    comparar(features_path)
    raise typer.Exit()


if __name__ == "__main__":
    app()
