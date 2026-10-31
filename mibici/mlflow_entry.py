"""Puntos de entrada del MLflow Project: ingest, features, train y search.

Traduce los parámetros del `MLproject` a los CLIs del proyecto. No repite
rutas ni fechas: esos defaults viven en `mibici/config.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys

import typer

app = typer.Typer(help="Entry points del MLflow Project — MiBici Equipo 01")

EXPERIMENT_NAME = "mibici-equipo01"

# pregunta -> comando de búsqueda con nested runs en train.py y opción de su malla
_BUSQUEDAS = {
    "p1": ("p1-search", "--n-estimators-list"),  # Random Forest: n_estimators
    "p3": ("p3-search", "--n-clusters-list"),    # K-Means++: K
}

# pregunta -> (modelo baseline, modelos admitidos)
_MODELOS = {
    "p1": ("regla", ("regla", "ridge", "rf")),
    "p2": ("dummy", ("dummy", "logreg")),
    "p3": ("random", ("random", "k-means++")),
}


def _asbool(valor: str) -> bool:
    """Interpreta los strings que envía MLflow ('true'/'false')."""
    return str(valor).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _ejecutar(argv: list[str]) -> int:
    """Corre `python -m <argv>` con el intérprete y entorno actuales."""
    return subprocess.call([sys.executable, "-m", *argv])


def _reconciliar_experimento() -> None:
    """Evita el choque de experimento entre `mlflow run` y `train.py`.

    `mlflow run` crea un run antes del entry point (en "Default"). Si quedó
    en otro experimento que `mibici-equipo01`, soltamos MLFLOW_RUN_ID para
    que train.py cree su run dentro del experimento del equipo.
    """
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return

    import mlflow

    cliente = mlflow.MlflowClient()
    run = cliente.get_run(run_id)
    exp = cliente.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is not None and run.info.experiment_id == exp.experiment_id:
        return  # ya coincide: se reutiliza el run
    os.environ.pop("MLFLOW_RUN_ID", None)


@app.command()
def ingest(
    force: str = typer.Option(
        "false", "--force", help="Sobrescribe el Parquet unificado sin preguntar."
    ),
    skip_download: str = typer.Option(
        "false", "--skip-download", help="No descargar; usa los CSV ya presentes."
    ),
    limit: str = typer.Option("", "--limit", help="Solo los primeros N meses (modo prueba)."),
) -> None:
    """Ejecuta la ingesta (mibici/dataset.py)."""
    argv = ["mibici.dataset"]
    if _asbool(force):
        argv.append("--force")
    if _asbool(skip_download):
        argv.append("--skip-download")
    if limit.strip() and limit.strip() != "0":
        argv += ["--limit", limit.strip()]
    raise typer.Exit(code=_ejecutar(argv))


@app.command()
def features() -> None:
    """Genera las features estación x hora (mibici/features.py)."""
    raise typer.Exit(code=_ejecutar(["mibici.features"]))


@app.command()
def train(
    question: str = typer.Option("p1", "--question", help="Pregunta: p1 | p2 | p3."),
    model: str = typer.Option("", "--model", help="Variante del modelo (vacío = baseline)."),
    n_estimators: int = typer.Option(100, "--n-estimators", help="P1 (rf): árboles del bosque."),
    rf_max_samples: int = typer.Option(500_000, "--rf-max-samples", help="P1 (rf): filas por árbol."),
    n_clusters: int = typer.Option(5, "--n-clusters", help="P3: número de clústeres K."),
    n_init: int = typer.Option(1, "--n-init", help="P3: inicios de k-means."),
) -> None:
    """Entrena y registra la pregunta indicada (mibici/modeling/train.py)."""
    q = question.strip().lower()
    if q not in _MODELOS:
        raise typer.BadParameter(f"Pregunta inválida: {question!r}. Usa p1, p2 o p3.")

    baseline, permitidos = _MODELOS[q]
    modelo = model.strip() or baseline
    if modelo not in permitidos:
        opciones = ", ".join(permitidos)
        raise typer.BadParameter(f"Modelo inválido para {q}: {modelo!r}. Opciones: {opciones}.")

    _reconciliar_experimento()

    if q == "p1":
        argv = [
            "mibici.modeling.train", "p1",
            "--model", modelo,
            "--n-estimators", str(n_estimators),
            "--rf-max-samples", str(rf_max_samples),
        ]
    elif q == "p2":
        argv = ["mibici.modeling.train", "p2", "--model-name", modelo]
    else:
        argv = [
            "mibici.modeling.train", "p3",
            "--init-method", modelo,
            "--n-clusters", str(n_clusters),
            "--n-init", str(n_init),
        ]

    raise typer.Exit(code=_ejecutar(argv))


@app.command()
def search(
    question: str = typer.Option("p1", "--question", help="Pregunta: p1 | p2 | p3."),
    grid: str = typer.Option(
        "auto", "--grid",
        help="Valores separados por comas (p1: n_estimators, p3: K). 'auto' = malla de train.py.",
    ),
) -> None:
    """Búsqueda de hiperparámetros con nested runs: un run padre y un run hijo por valor.

    p1 -> n_estimators (default 50,100,200); p3 -> K (default 3,4,5,6).
    p2 -> búsqueda de C de la regresión logística (0.1, 1, 10), la misma que
    `train -P question=p2 -P model=logreg`; `grid` no aplica.
    """
    q = question.strip().lower()
    if q == "p2":
        argv = ["mibici.modeling.train", "p2", "--model-name", "logreg"]
    elif q in _BUSQUEDAS:
        comando, opcion = _BUSQUEDAS[q]
        argv = ["mibici.modeling.train", comando]
        malla = grid.strip()
        # 'auto' o vacío (en Windows MLflow puede pasar '""') -> malla por defecto de train.py
        if malla and malla.lower() != "auto" and malla != '""':
            argv += [opcion, malla]
    else:
        raise typer.BadParameter(f"Pregunta inválida: {question!r}. Usa p1, p2 o p3.")

    _reconciliar_experimento()
    raise typer.Exit(code=_ejecutar(argv))


if __name__ == "__main__":
    app()
