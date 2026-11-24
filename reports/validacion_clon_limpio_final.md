# Validación de reproducibilidad desde clon limpio (issue #190)

Ejecución del pipeline completo del proyecto desde un clon limpio, usando únicamente los entry points del `MLproject`. Se generó con [`scripts/validar_clon_limpio.sh`](../scripts/validar_clon_limpio.sh), que:

1. Clona la rama en una carpeta vacía (sin datos, sin `mlflow.db` y sin `mlruns/`).
2. Levanta un servidor MLflow propio del clon en el puerto `5002`, separado del Model Registry del equipo.
3. Corre `ingest` → `features` → baselines → modelos elegidos → `register` con `mlflow run --env-manager uv`.
4. Crea un entorno con `requirements.txt`, corre `pytest` y verifica que los tres `@champion` carguen.

Para repetirla:

```bash
source .venv/bin/activate          # requiere mlflow y uv en el PATH
bash scripts/validar_clon_limpio.sh main
```

## Entorno

| Dato | Valor |
|---|---|
| Fecha | 2026-11-23 22:32 |
| Rama / commit | `fix/issue-190-reproducibilidad` / `44143ae` |
| Sistema | macOS (Darwin arm64) |
| Python del MLflow Project | 3.12.9 (creado por `uv` desde `python_env.yaml`) |
| MLflow del MLflow Project | 3.17.2 (alineado con `requirements.txt` en este issue) |
| uv | 0.13.5 |
| Tracking URI | `http://127.0.0.1:5002` (servidor propio del clon) |

## Resultados por paso

| # | Paso | Comando | Resultado | Duración |
|---|---|---|---|---|
| 1 | ingest | `mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e ingest` | OK | 1m 27s |
| 2 | features | `mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e features` | OK | 0m 12s |
| 3 | train P1 baseline | `… -e train -P question=p1 -P model=regla` | OK | 0m 31s |
| 4 | train P2 baseline | `… -e train -P question=p2 -P model=dummy` | OK | 0m 51s |
| 5 | train P3 baseline | `… -e train -P question=p3 -P model=random` | OK | 0m 10s |
| 6 | train P1 elegido | `… -e train -P question=p1 -P model=rf -P n_estimators=200` | OK | 1m 22s |
| 7 | train P2 elegido | `… -e train -P question=p2 -P model=logreg` | OK | 1m 04s |
| 8 | train P3 elegido | `… -e train -P question=p3 -P model=k-means++ -P n_init=10 -P n_clusters=3` | OK | 0m 10s |
| 9 | register | `… -e register` | OK | 0m 03s |
| 10 | entorno de pruebas | `uv venv --python 3.12 .venv && uv pip install -r requirements.txt` | OK | 0m 04s |
| 11 | pytest | `.venv/bin/python -m pytest -q` | OK | 1m 03s |
| 12 | verificar registry | `.venv/bin/python scripts/verificar_registry.py` | OK | 0m 03s |

**Los 12 pasos terminaron correctamente en unos 7 minutos.**

## Qué produjo cada etapa

**Datos y features**
- Ingesta: 42 meses + nomenclatura descargados; **16,011,685 viajes** y **470 estaciones** en el Parquet unificado.
- Features: **11,277,935 filas** estación × hora (475 estaciones con viajes en el histórico).

**Métricas registradas en MLflow Tracking (baseline vs elegido)**

| Pregunta | Métrica | Baseline (val / test) | Elegido (val / test) |
|---|---|---|---|
| P1 Regresión | MAE (menor es mejor) | 0.8517 / 0.8581 (`regla`) | **0.8118 / 0.8089** (`rf`, 200 árboles) |
| P2 Clasificación | ROC AUC (mayor es mejor) | 0.5000 / 0.5000 (`dummy`) | **0.8278 / 0.8223** (`logreg`) |
| P3 Clustering | Silhouette (mayor es mejor) | 0.2984 / 0.1031 (k-means aleatorio, k=5) | **0.3872 / 0.1893** (k-means++, k=3) |

**Model Registry del clon** (`register` elige por métricas de validación):

| Modelo | Versión | Run de origen | Métrica de selección |
|---|---|---|---|
| `mibici-equipo01-p1-demanda` | v1 `@champion` | `P1-candidato-rf` | `val_mae = 0.8118` |
| `mibici-equipo01-p2-riesgo` | v1 `@champion` | `P2-parent-search-logreg` | `val_roc_auc = 0.8278` |
| `mibici-equipo01-p3-grupos` | v1 `@champion` | `p3-candidate-k3` | `val_silhouette_score = 0.3872` |

`verificar_registry.py` cargó los tres modelos por alias (`models:/<nombre>@champion`).

**Pruebas:** `83 passed, 1 skipped`. La omitida es `tests/test_data_validation.py::test_validacion_completa_con_deepchecks`, que requiere el entorno aparte de DeepChecks (`make validation_env`), como documenta el README.

## Observaciones

1. **`python_env.yaml` y `requirements.txt` ya fijan la misma versión de MLflow (3.17.2).** Antes el MLflow Project se creaba con 3.17.0, la versión con CVE-2026-53995.
2. **La fuente de datos sigue cambiando.** La descarga trajo 16,011,685 viajes, 26 más que los 16,011,659 del manifiesto versionado, así que los hashes de `data/manifest.json` difieren. El pipeline y las pruebas funcionan con los datos descargados; el manifiesto versionado sigue siendo la referencia del equipo y no se debe commitear el regenerado (ver README, "Descarga de datos").
3. **Parámetros de los modelos elegidos.** Para reproducir los `@champion` actuales, P1 usa `n_estimators=200` y P3 `n_clusters=3`. El README todavía muestra `n_estimators=300` y no fija `n_clusters` (default 5); conviene alinearlo en el README general (#189).
4. **`register` en un registry nuevo ya elige por validación** y deja las métricas `val_*` en el linaje. Esto sirve de referencia para el issue #191, donde la v2 del registry del equipo no tiene esas métricas.
5. La validación no toca el Model Registry del equipo: usa su propio `mlflow.db` en el clon.
