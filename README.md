# TC5061 Proyecto MiBici Equipo 01

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Regresión, clasificación y clustering sobre los viajes del sistema de bicicleta pública de Guadalajara (MiBici), con ingesta reproducible, partición temporal sin fuga y experimentos registrados en MLflow.

## Preguntas de investigación

|    | Pregunta | Tarea | Responsable |
|----|----------|-------|-------------|
| P1 | ¿Cuántos viajes se iniciarán en una estación en un periodo futuro? | Regresión | [@laloz60](https://github.com/laloz60) |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicis/anclajes en un periodo futuro? | Clasificación | [@GemmaMartinez](https://github.com/GemmaMartinez) + [@smuciton](https://github.com/smuciton) |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | [@JesusTafolla](https://github.com/JesusTafolla) |

> **Reparto de P2:** es la tarea más pesada (diseño del proxy de riesgo + baseline de clasificación + evidencia MLflow), así que se divide en dos: **@smuciton** diseña y justifica el proxy de riesgo (#6) y **@GemmaMartinez** implementa el baseline de clasificación con su run en MLflow (#7).

## Datos

- **Fuente:** [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos)  CSVs mensuales de viajes + nomenclatura de estaciones.
- **Cobertura:** 2023-05 a 2026-10 (**42 meses**, corte postpandemia decidido por el equipo: fin de la emergencia sanitaria COVID-19, mayo 2023)  **16,011,659 viajes** y **470 estaciones**.
- **Lo que se versiona en Git:** solo `data/manifest.json` (hash SHA-256 de cada CSV, del Parquet unificado y de la nomenclatura). Los CSVs (~1.2 GB) y el Parquet (~330 MB) **no se suben al repositorio**; se regeneran con la ingesta y el manifiesto garantiza que sean idénticos.
- **Partición temporal sin fuga** (definida en `mibici/config.py`, nunca se mezclan meses futuros con el entrenamiento):

| Split  | Rango |
|--------|-------|
| train  | 2023-05 .. 2024-12 |
| val    | 2025-01 .. 2025-06 |
| test   | 2025-07 .. 2026-10 |

## Requisitos

- Python 3.12
- ~2 GB de espacio libre (CSVs + Parquet)
- [uv](https://docs.astral.sh/uv/getting-started/installation/): `mlflow run` lo usa para crear el entorno del `python_env.yaml` (Python 3.12.9 con versiones fijadas). Sin uv, MLflow necesita pyenv y falla con `'None' install 3.12.9`.
- **Windows:** activar el *Modo de programador* (Configuración → Actualización y seguridad → Para programadores). `mlflow run` crea symlinks al instalar dependencias; sin ese permiso falla con `WinError 1314`.

## Instalación

```bash
git clone https://github.com/laloz60/TC5061-Proyecto-Equipo01.git
cd TC5061-Proyecto-Equipo01
python3.12 -m venv .venv         # PowerShell: python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # instala también el paquete mibici (-e .)
```

## Descarga de datos

**Clon limpio (desde cero):** solo hace falta este comando. Descarga los 41 CSVs mensuales y la nomenclatura de estaciones (~1.2 GB) y los unifica en un Parquet:

```bash
python -m mibici.dataset
```

> Si vas a reproducir la entrega con `mlflow run` (ver [Reproducir con `mlflow run`](#reproducir-con-mlflow-run)), el entry point `ingest` ya ejecuta este mismo paso; no hace falta correrlo aparte.

**Opcionales** (no se necesitan en un clon limpio):

| Comando | Cuándo usarlo |
|---------|---------------|
| `python -m mibici.dataset --skip-download` | Ya tienes los CSVs en `data/raw/csv/` y solo quieres volver a unificarlos (p. ej. después de un cambio en la normalización). Falla si falta algún CSV. |
| `python -m mibici.dataset --force` | El Parquet unificado ya existe y quieres regenerarlo sin que se detenga a preguntar `¿Continuar?`. Los CSVs que ya están descargados se reutilizan. |
| `python -m mibici.dataset --limit 2` | Prueba rápida con solo 2 meses, para revisar que la ingesta funciona sin descargar todo. Escribe sus salidas aparte y no toca el Parquet ni el manifiesto completos. |

Salidas:

```
data/raw/mibici_viajes_2023_05_a_2026_10.parquet   # Parquet unificado (no versionado)
data/raw/nomenclatura_estaciones.parquet           # 470 estaciones: capacidad, lat/lon
data/manifest.json                                 # versionado: totales + SHA-256
```

Con `--limit` los archivos de prueba salen en `data/raw/mibici_viajes_limit.parquet` y `data/manifest_limit.json`: no se versionan y no reemplazan a los de arriba, así que el manifiesto completo solo lo escribe la ingesta completa.

Verificación de que tus datos son idénticos a los del equipo:

```bash
pytest            # pruebas de ingesta, manifiesto y features
```

Si `pytest` falla con `assert 15376153 == 15376127` en `test_unified_parquet_exists_and_is_non_empty`, no es un error del código: MiBici actualizó el CSV de 2023-05 después de que se generó el manifiesto versionado, así que tu descarga ya no coincide con él. Para validar tu entorno con los datos que descargaste:

```bash
python -m mibici.dataset --skip-download --force   # regenera el manifiesto local con tus datos (~1.5 min, sin volver a descargar)
pytest                                             # ahora deben pasar las 67 pruebas
git restore data/manifest.json                     # no commitees el manifiesto regenerado
```

En ese caso `pytest` confirma que la ingesta es consistente (Parquet, manifiesto y hashes cuadran) y que features y modelos funcionan, pero ya no que tus datos sean idénticos a los del equipo.

## Features (P1 y P2)

```bash
python -m mibici.features
```

Genera `data/processed/features_estacion_hora.parquet` con una fila por estación y hora: salidas, llegadas, variables de calendario, lags, promedios y capacidad de la estación. Los lags y promedios solo usan horas anteriores.

## P2 (Clasificación)

Para ejecutar el entrenamiento de los modelos de clasificación y registrar los experimentos y artefactos en MLflow, utiliza los siguientes comandos:

```bash
python -m mibici.modeling.train p2 --model-name dummy
python -m mibici.modeling.train p2 --model-name logreg
```
Esta etapa evalúa y predice si una estación se encuentra en riesgo de desabastecimiento de bicicletas. Mediante 2 modelos, depende de cual seleccionemos.
1. **Modelo Baseline (`dummy`):** Clasificador por defecto que establece el rendimiento mínimo de referencia utilizando clases mayoritarias.
2. **Modelo Candidato (`logreg`):** Modelo basado en Regresión Logística, diseñado para capturar relaciones lineales.

## Experimentos (MLflow)

### Servidor de tracking

El Model Registry necesita un almacenamiento respaldado por base de datos, así que el tracking se levanta con un servidor local sobre SQLite. Lánzalo **desde la raíz del repo** (ahí viven `mlflow.db` y `mlruns/`):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns --port 5000
# UI en http://127.0.0.1:5000
```

`--artifacts-destination ./mlruns` hace que el servidor sirva los artefactos por HTTP (`mlflow-artifacts:/`). Para correr en el host no hace falta, pero el contenedor sí lo necesita para descargar los modelos (ver [Conexión del contenedor con MLflow](#conexión-del-contenedor-con-mlflow)).


En otra terminal (o antes de lanzar cualquier corrida), apunta la variable de entorno al servidor para que todos los runs se registren ahí:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"      # bash / macOS / Linux / Git Bash
```
```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"     # PowerShell
```
En CMD, sin espacios ni comillas (CMD guarda en la variable todo lo que va después del `=`, incluido un espacio al final):

```cmd
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
echo [%MLFLOW_TRACKING_URI%]
```

El `echo` debe mostrar `[http://127.0.0.1:5000]`, con el `]` pegado al `5000`. Si aparece `[http://127.0.0.1:5000 ]`, la variable quedó con un espacio y `mlflow run` falla con `Invalid url: http://127.0.0.1:5000 /api/...`: vuelve a escribir el `set`.

La variable solo vive en la terminal donde se definió: los `mlflow run` deben lanzarse desde esa misma terminal. Puedes persistirla en `.env` (no se versiona) para no repetirla en cada sesión. Todas las corridas de esta entrega se lanzan con `mlflow run`, no desde un notebook.

> `mlflow.db` y `mlruns/` **no se versionan** (están en `.gitignore`, junto con `mlartifacts/`, los datos MiBici y los modelos serializados). En un clon nuevo el tracking arranca vacío, así que los runs se regeneran con los comandos de esta sección (las capturas de evidencia están en `reports/figures/mlflow_evidencia/`).

### Reproducir con `mlflow run`

Con el servidor arriba y `MLFLOW_TRACKING_URI` definida, desde la raíz del repo y en este orden. `--experiment-name` va siempre, para que los runs no caigan en `Default`; `--env-manager uv` crea el entorno aislado del `python_env.yaml` (la primera vez tarda ~1 min, después se reutiliza).

```bash
# 1. Datos y features
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e ingest              # macOS / Linux
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e ingest -P limit=0   # Windows
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e features

# 2. Baselines
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p1 -P model=regla
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p2 -P model=dummy
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p3 -P model=random

# 3. Modelo elegido de cada pregunta
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p1 -P model=rf -P n_estimators=300
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p2 -P model=logreg
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p3 -P model=k-means++ -P n_init=10

# 4. Model Registry
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e register
```

Valores de `model` por pregunta (vacío = baseline): p1 `regla` | `ridge` | `rf` · p2 `dummy` | `logreg` · p3 `random` | `k-means++`. Los parámetros completos de cada entry point están en el `MLproject`; los defaults (rutas, fechas, semilla) viven en `mibici/config.py`.

Notas:

- **`-P limit=0` en la ingesta (solo Windows)** significa "sin límite". En Windows, el default vacío llega como `""` y la ingesta falla con `Invalid value for '--limit'`; en macOS/Linux no hace falta.
- **Re-ejecutar la ingesta** cuando el Parquet ya existe: agrega `-P force=true`, o se detiene a preguntar `¿Continuar?`. Para no volver a descargar los CSV: `-P skip_download=true`.
- **`features` es obligatorio** antes de entrenar P1 y P2. Sus runs (y los de `ingest`) no registran métricas ni parámetros: solo dejan trazabilidad de commit y entry point.
- **Tiempos de referencia** (Windows, conexión doméstica): ingesta completa ~7.5 min (descarga ~6 min), features ~1 min, P2 `logreg` ~4.5 min, P3 <1 min.
- **Datos de la fuente:** MiBici puede actualizar CSV ya publicados. Al 22 oct 2026, el de 2023-05 trae 26 viajes más que el registrado en `data/manifest.json` (15,376,153 vs 15,376,127 en total), así que una descarga nueva produce otros hashes y `pytest` marca esa diferencia (ver [Descarga de datos](#descarga-de-datos) para validar con tus datos). El manifiesto versionado es la referencia del equipo: **no lo commitees al re-ejecutar la ingesta** (`git restore data/manifest.json`).

La salida completa de estos comandos en un clon limpio (comando, respuesta del log y resultado de cada paso) está en [`reports/validacion_clon_limpio_issue62.md`](reports/validacion_clon_limpio_issue62.md).

La comparación de baseline contra candidato de cada pregunta está en [`reports/baseline_vs_candidato.md`](reports/baseline_vs_candidato.md).

### Model Registry

`mlflow run . -e register` (`mibici/modeling/register.py`) registra el mejor run de cada pregunta según su métrica de **validación** (la selección no mira test; esa ventana queda solo para el reporte final):

| Pregunta | Modelo registrado | Métrica de selección |
|---|---|---|
| P1 | `mibici-equipo01-p1-demanda` | `val_mae` (menor es mejor) |
| P2 | `mibici-equipo01-p2-riesgo` | `val_roc_auc` (mayor es mejor) |
| P3 | `mibici-equipo01-p3-grupos` | `val_silhouette_score` (mayor es mejor) |

- Busca primero entre las corridas con tag `stage_of_work=candidate`; si ninguna tiene modelo logueado, usa todas las corridas de la pregunta.
- Cada versión lleva en su descripción el linaje: run de origen, commit y hashes del manifiesto y del Parquet.
- No registra dos veces el mismo run: re-ejecutarlo es seguro.
- **Ciclo de vida:** la versión en producción se marca con el alias `@champion` y se carga por nombre:

  ```python
  import mlflow
  modelo = mlflow.pyfunc.load_model("models:/mibici-equipo01-p2-riesgo@champion")
  ```

#### Ubicación del registry y modelos desplegados

El Model Registry vive en la **máquina anfitriona**: los metadatos en `mlflow.db` (SQLite) y los artefactos en `mlruns/`, servidos por el servidor MLflow del puerto 5000. La API **no lleva modelos dentro de la imagen**: al arrancar, se conecta a ese servidor con `MLFLOW_TRACKING_URI` y carga los tres modelos `@champion`. Para publicar una versión nueva basta con mover el alias en el registry y reiniciar el contenedor, sin reconstruir la imagen.

Modelos que sirve la API:

| Pregunta | Modelo registrado | Versión `@champion` | Run de origen | Endpoint |
|---|---|---|---|---|
| P1 | `mibici-equipo01-p1-demanda` | 2 | `p1-candidate-rf-n200` (`41e42807…`) | `POST /predict/p1-demanda` |
| P2 | `mibici-equipo01-p2-riesgo` | 2 | `P2-parent-search-logreg`, C=1.0 (`3ad75953…`) | `POST /predict/p2-riesgo` |
| P3 | `mibici-equipo01-p3-grupos` | 2 | `p3-candidate-k3` (`9fa2a199…`) | `POST /predict/p3-clustering` |

La carga la hace `mibici/api/registry.py` (`load_champion`) en dos pasos:

1. **En el host**, `mlflow.pyfunc.load_model("models:/<nombre>@champion")` carga directo, porque la ruta local de los artefactos existe.
2. **En el contenedor**, ese paso falla con `No such artifact: ''`, porque el registry guarda una ruta local del host. Entonces resuelve el alias con `MlflowClient` y descarga los artefactos por el proxy `mlflow-artifacts:/` del servidor. Por eso el servidor necesita `--artifacts-destination ./mlruns` (ver [Conexión del contenedor con MLflow](#conexión-del-contenedor-con-mlflow)).

Para comprobar qué versión resuelve cada alias y que los tres modelos cargan:

```bash
python scripts/verificar_registry.py
```

> La versión 2 de los tres modelos se registró antes de que `register` eligiera por métricas de validación, así que sus runs solo tienen métricas `test_*`. Volver a correr `-e register` puede crear una versión 3 y mover `@champion`: después de hacerlo, revisa esta tabla con `scripts/verificar_registry.py`.

### Ejecución directa (sin MLflow Project)

Útil para desarrollo. Para reproducir la entrega usa los comandos `mlflow run` de arriba. Las búsquedas con nested runs de P1 y P3 hoy solo se lanzan así: `python -m mibici.modeling.train p1-search` y `python -m mibici.modeling.train p3-search`.

- Experimento: `mibici-equipo01`
- Convención de runs: nombre `P1-baseline`, `P2-baseline-logreg`, `P3-kmeans`; tags `equipo=01`, `pregunta=P1|P2|P3`, `commit=<sha>`; params incluyen rango de fechas y hash del manifiesto; al menos 1 artefacto por run.
- Entrenamiento: `python -m mibici.modeling.train`. Antes de P1 y P2 correr `python -m mibici.features` para generar las features.
- **P1 (Regresión)** — baseline `regla` (no entrena) y candidatos `ridge` y `rf`:
  ```bash
  python -m mibici.modeling.train p1 --model regla
  python -m mibici.modeling.train p1 --model ridge
  python -m mibici.modeling.train p1 --model rf
  ```
- **P2 (Clasificación)** — baseline `dummy` y candidato `logreg`:
  ```bash
  python -m mibici.modeling.train p2 --model-name dummy
  python -m mibici.modeling.train p2 --model-name logreg
  ```
- **P3 (Clustering)** — baseline (K-Means con inicialización aleatoria). El pipeline completo es de tres pasos:
  ```bash
  python -m mibici.modeling.train p3    # entrena K-Means (k=5) y lo serializa en models/kmeans_p3.pkl
  python -m mibici.modeling.predict p3  # asigna clústeres y escribe reports/p3_evaluation_summary.txt
  python -m mibici.reportes_p3          # regenera el summary y las figuras de P3 desde los datos
  ```
  El tercer comando (`mibici/reportes_p3`) reconstruye `reports/p3_evaluation_summary.txt`,
  `reports/figures/p3_distribucion_clusters.png` y `reports/figures/p3_mapa_geografico.png`
  sin depender de binarios versionados, así el reporte es reproducible en un clon limpio.

Las capturas de pantalla de los *runs*, artefactos y comparación de modelos se encuentran en:
```text
reports/figures/mlflow_evidencia/
```

## Pruebas de calidad y seguridad
El proyecto incluye pruebas con `pytest`, validación de datos con `DeepChecks` y chequeos de seguridad (`bandit` + `pip-audit`).

**Estructura de `tests/`** (7 archivos, 67 pruebas)
- `tests/test_data.py` (12): datos de MiBici, validación de SHA-256 e integridad del parquet unificado, comportamiento del CLI y parsing de fechas. Incluye casos límite: header malformado de 2023-04, literales nulos y formato de fecha desconocido.
- `tests/test_features.py` (13): cálculo de features por estación y hora, lags, media móvil, calendario, split temporal y etiqueta de riesgo de P2. Cubre casos límite como horas sin viajes, estación que empieza en su primer viaje y llegadas fuera de la ventana. Incluye `test_sin_fuga_de_informacion`.
- `tests/test_modeling.py` (14): fechas de `config`, hash del parquet, tags de MLflow, métricas de regresión, selección por validación, silhouette de P3 y perfil de ventana. Casos límite: silhouette con una sola fila (`NaN`) y ventana de perfil vacía.
- `tests/test_reproducibilidad.py` (3): dos entrenamientos con la misma semilla dan las mismas predicciones y métricas; con semilla distinta cambian (control negativo); Ridge es estable entre corridas.
- `tests/test_register.py` (13): detección de modelo logueado, selección del mejor run por métrica de validación, fallback a corridas previas e idempotencia del registro con alias `@champion`.
- `tests/test_proxy_riesgo.py` (11): construcción de la tabla de riesgo de P2, sensibilidad al umbral, top de estaciones y plots. Casos límite: estación sin desbalance, estación inexistente en nomenclatura y corte `n` exacto.
- `tests/test_data_validation.py` (2): validación del esquema y de la corrida completa de DeepChecks (esta última corre solo con el entorno de validación creado).

**Comandos para ejecutar**
Ya estando en el entorno virtual, es necesario ejecutar

1. Pruebas unitarias `pytest`
  ```bash
pytest -v
  ```

Salida ejemplo
  ```bash
platform darwin -- Python 3.12.9, pytest-9.1.1, pluggy-1.7.0 -- .venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/gerardocarreno/Documents/MaestriaTecMonterrey/CuartoTrimestre/MLOps/Semana_5/TC5061-Proyecto-Equipo01
configfile: pyproject.toml
plugins: platformdirs-4.13.3, anyio-4.15.1
collecting ... collected 67 items

tests/test_data.py::test_catalog_cover_post_pandemic_history PASSED      [  1%]
tests/test_data.py::test_normalize_columns_canonical_schema PASSED       [  2%]
tests/test_data.py::test_normalize_columns_mangled_header_2023_04 PASSED [  4%]
tests/test_data.py::test_unified_parquet_exists_and_is_non_empty PASSED  [  5%]
tests/test_data.py::test_manifest_is_versioned_and_complete PASSED       [  7%]
tests/test_data.py::test_output_paths_limit_escribe_aparte PASSED        [  8%]
...
tests/test_reproducibilidad.py::test_misma_semilla_repite_predicciones_y_metricas PASSED [ 97%]
tests/test_reproducibilidad.py::test_semilla_distinta_cambia_el_entrenamiento PASSED [ 98%]
tests/test_reproducibilidad.py::test_ridge_tambien_es_reproducible PASSED [100%]

================== 67 passed, 1 warning in 20.54s ===================
  ```

> Para que las 67 corran hace falta el entorno de validación del punto 3 (`make validation_env`): sin él, `test_validacion_completa_con_deepchecks` se salta y el total queda en 66.

Captura de pantalla para verificar que se ejecute correctamente `reports\figures\evidencia_pruebas_calidad_seguridad\pytest`
![Pytest Success](reports/figures/evidencia_pruebas_calidad_seguridad/pytest.png)

2. Pruebas de reproducibilidad del entrenamiento
  ```bash
pytest tests/test_reproducibilidad.py -v
  ```

Verifica que dos entrenamientos con la misma semilla producen las mismas predicciones y métricas, que cambiar la semilla sí altera el resultado (control negativo) y que Ridge es estable entre corridas.

3. Validación de datos con `DeepChecks`

Revisa tipos, valores faltantes, duplicados y drift entre train y test. Corre en un entorno aparte porque DeepChecks 0.19.1 no funciona con las versiones de scikit-learn y NumPy del proyecto.

  ```bash
make validation_env   # solo la primera vez
make validate
  ```

Sin `make`, en Windows:

  ```bash
uv venv --python 3.12 .venv-validacion
uv pip install --python .venv-validacion\Scripts\python.exe -r requirements-validation.txt
.venv-validacion\Scripts\python.exe -W ignore -m mibici.data_validation
  ```

La salida queda en `reports/data_validation/resumen.json`. Los hallazgos y las acciones tomadas están en [`reports/validacion_datos.md`](reports/validacion_datos.md).

**Resumen del último chequeo** (15,376,127 viajes, semilla 42, muestras de 200,000 filas):

| Chequeo | Resultado |
|---|---|
| Tipos de datos | `ok = True`, sin diferencias contra el esquema esperado |
| Valores faltantes | `genero` 104,981 (0.68%) · `anio_nacimiento` 7,773 (0.05%) · resto 0 |
| Duplicados | 758 filas exactas · 1,065 mismos viajes con otro `viaje_id` |
| DeepChecks (muestra) | sin tipos mezclados · 0.0% de filas duplicadas |
| Drift viajes (train vs test) | ninguna columna marcada · máx. `anio_nacimiento` 0.0788 (K-S) |
| Drift features (train vs test) | ninguna columna marcada · máx. `mes` 0.1832 (Cramer's V) |

El umbral de drift es 0.2 y ninguna columna lo alcanza. Las 758 filas duplicadas exactas ya se eliminan al generar las features.

Captura de la ejecución de `make validate` con DeepChecks, `reports\figures\evidencia_pruebas_calidad_seguridad\DeepChecks.png`
![Validación DeepChecks](reports/figures/evidencia_pruebas_calidad_seguridad/DeepChecks.png)

4. Análisis de calidad
  ```bash
ruff check .
  ```

Salida ejemplo
  ```bash
All checks passed!
  ```

Captura de pantalla para verificar que se ejecute correctamente `reports\figures\evidencia_pruebas_calidad_seguridad\ruff.png`
![Lint Success](reports/figures/evidencia_pruebas_calidad_seguridad/ruff.png)

5. Chequeos de seguridad (`bandit` + `pip-audit`)
  ```bash
make security
  ```

Los reportes quedan en `reports/security/bandit_report.txt` y `reports/security/pip_audit_report.txt`.

Captura de la ejecución de `make security`, `reports\figures\evidencia_pruebas_calidad_seguridad\security.png`
![Security Success](reports/figures/evidencia_pruebas_calidad_seguridad/security.png)

### Comparación de Familias de Modelos

Explorarando múltiples algoritmos, el pipeline de entrenamiento ahora compara distintas familias de modelos utilizando *nested runs* en MLflow para cada pregunta de investigación:

- **P1 (Regresión):** Se compara el baseline (regla del promedio de la misma hora en las 4 semanas previas) contra `Ridge`, `RandomForestRegressor` y la nueva familia `GradientBoostingRegressor`.
- **P2 (Clasificación):** Se compara el baseline (`Dummy`) contra `LogisticRegression` y la nueva familia `RandomForestClassifier`.
- **P3 (Clustering):** Se compara el baseline (`KMeans` con inicialización aleatoria) contra `KMeans++` y la nueva familia `GaussianMixture`.

Para ejecutar las búsquedas de hiperparámetros y registrar los candidatos de estas nuevas familias en tu servidor local de MLflow, ejecuta:
```bash
python -m mibici.modeling.train p1-search
python -m mibici.modeling.train p2 --model-name rf
python -m mibici.modeling.train p3-search
```

## Monitoreo, Drift y A/B Testing (Actividad 9)

El proyecto incluye una capa de observabilidad para mantener vigentes los modelos: detecta cambios en los datos y evalúa nuevas versiones antes de promoverlas. El resultado integrado (drift, A/B y recomendación de Continuous Training) está en [`reports/reporte_actividad9.md`](reports/reporte_actividad9.md).

### 1. Reporte de Data Drift y Calidad (Evidently)

Compara la distribución y la calidad de los datos en dos periodos separados por fecha, para evitar fuga temporal y simular el entorno de producción:

- **Conjunto de Referencia (Train):** mayo 2023 a diciembre 2024. Es la ventana con la que se entrenaron los modelos vigentes (`config.TRAIN_END`), en un periodo postpandemia ya estabilizado.
- **Conjunto Actual (Test/Producción):** enero 2025 a octubre 2026. Periodo que el modelo no vio; muestra cómo cambian la movilidad y la operación de la red con el tiempo.

De cada periodo se toman 30,000 filas con `config.SEED` para que el reporte quepa en memoria y sea reproducible.

```bash
mlflow run . -e drift --env-manager=uv
```

Salida: `reports/drift_report.html` (reporte interactivo) y `reports/drift_report.json` (métricas por variable).

### 2. A/B Testing del modelo P2 (Riesgo)

Compara la versión vigente del modelo (`@champion`) contra una V2 reentrenada con la ventana extendida 2023-05 a 2025-06, registrada como nueva versión del mismo modelo con el alias `@ab-candidate`. El alias `champion` no se mueve.

Las dos versiones se evalúan sobre **el mismo conjunto**: el split `test` (2025-07 a 2026-09), con las mismas features, la misma etiqueta y la misma semilla. Se reportan ROC AUC, average precision, precisión, recall y F1 macro, y la diferencia se prueba con el test exacto de McNemar.

```bash
mlflow run . -e ab_test --env-manager=uv
```

El entry point entrena y registra la V2, y después compara las dos versiones (`python -m mibici.modeling.ab_testing run`). Requiere el servidor de MLflow levantado y la V1 registrada con el alias `@champion`. Salida: `reports/ab_testing_p2.md`, `reports/ab_testing_p2.csv` y las gráficas en `reports/figures/ab_testing_p2/`.

## Buenas prácticas

- **Métricas**: se nombran con el conjunto donde se calculan, `<split>_<métrica>` — `test_rmse`, `test_roc_auc`, `test_silhouette_score` (ver `mibici/modeling/train.py`).
- **Notebooks**: convención CCDS `notebooks/<número>-<descripción>.ipynb` (p. ej. `1.0-iniciales-descripcion.ipynb`). Solo exploración: lo que se reutiliza baja a los módulos (`mibici/features.py`, `mibici/modeling/`). Sin datos personales de usuarios en las gráficas exportadas.
- **Credenciales y datos personales**: nunca se suben a MLflow ni al repo. Viven en `.env`, que está en `.gitignore`; los runs solo registran params y tags públicos.
- **Config centralizada**: rutas, cortes de fechas y `SEED` se leen de `mibici/config.py`, sin duplicarlos en `mibici/dataset.py` ni en `mibici/modeling/`.
- **`.gitignore`**: excluye los datos MiBici (`/data/*`, salvo `data/manifest.json`), `mlruns/`, `mlartifacts/`, `.env` y los modelos serializados (`models/*.pkl`, `models/*.joblib`, `models/*.onnx`, `models/*.bin`).
- **Versionado de datos**: `data/manifest.json` guarda el SHA-256 del parquet unificado y de la nomenclatura, con el total de viajes y estaciones. Es lo único de `data/` que entra al repo.

## Seguridad y Calidad del Código

Para garantizar que el repositorio no exponga credenciales ni dependencias vulnerables, integramos herramientas de análisis estático y auditoría de seguridad:

- **Chequeo de credenciales (Issue #102):** El archivo `.gitignore` excluye explícitamente `.env`, bases de datos locales (`mlflow.db`) y datos crudos (`data/raw/`). Se verificó en el historial mediante `git ls-files` que no existen secretos expuestos. Asimismo, una búsqueda con `git grep` descartó credenciales hardcodeadas. La plantilla versionada es [`.env.example`](.env.example): se copia a `.env` (no versionado) y se edita.
- **Análisis estático (Bandit v1.9.4):** Se ejecuta sobre el directorio `mibici/`. Se detectaron 5 alertas de severidad baja (Low) asociadas al uso de `subprocess` en `tracking.py` y `mlflow_entry.py`. **Justificación (Issue #101):** Estas alertas se asumen como seguras ya que los subprocesos solo ejecutan comandos internos de infraestructura (`git rev-parse HEAD` y llamadas nativas a Python), sin procesar *input* externo no confiable. El reporte completo está en `reports/security/bandit_report.txt`.
- **Auditoría de dependencias (pip-audit v2.10.1):** El escaneo inicial detectó la vulnerabilidad *CVE-2026-53995* en la librería `mlflow` (v3.17.0). **Resolución:** Se mitigó el riesgo actualizando la dependencia a `mlflow==3.17.2` en el archivo `requirements.txt`, resultando en una auditoría limpia. El reporte está en `reports/security/pip_audit_report.txt`.

Para ejecutar automatizadamente los chequeos de seguridad (Issue #101), utiliza el siguiente comando de `make`:
```bash
make security
```

## Ejecución de Pruebas y Chequeos de Seguridad

Para garantizar la calidad y seguridad del código, este repositorio integra pruebas unitarias, validación de datos y análisis estático. Todo el flujo está orquestado mediante `Make`.

### Instrucciones de ejecución
Desde la raíz del repositorio, crea una vez el entorno de validación con `make validation_env` y ejecuta el siguiente comando unificado:
```bash
make check-all
```

- **Limpieza y privacidad:** consulta la [preparación y protección de datos](docs/clean_doc).

## Despliegue de Modelos (API REST con Docker)

El servicio expone los modelos de Regresión (P1), Clasificación (P2) y Clustering (P3) del proyecto mediante una API REST construida con **FastAPI** y validación de las entradas y las salidas con **Pydantic**. Los modelos **no se incluyen en la imagen**: a través de la variable `MLFLOW_TRACKING_URI` el contenedor se conecta al servidor local de MLflow, carga los modelos en memoria RAM al iniciar y los sirve con el alias `@champion`, de modo que siempre se entrega la última versión sin reconstruir la imagen OCI.

### Requisitos

- Docker instalado y en ejecución (el `Dockerfile` es estándar OCI: funciona igual con Podman).
- Servidor MLflow accesible desde el contenedor (ver [Conexión del contenedor con MLflow](#conexión-del-contenedor-con-mlflow)).
- Modelos registrados en MLflow con el alias `@champion`.

### Construir la imagen

Desde la raíz del repositorio:

```bash
docker build -t mibici-api:latest .
```

O, con el target del `Makefile` (mismo resultado, automatizado):

```bash
make docker-build   # Construye la imagen mibici-api:latest
```

### Levantar el contenedor

Primero levanta el servidor de MLflow (ver [Experimentos](#experimentos-mlflow) y [Conexión del contenedor con MLflow](#conexión-del-contenedor-con-mlflow)). Después:

```bash
docker run -d \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  --name mibici-service \
  mibici-api:latest
```

En Linux, agrega `--add-host` al comando (en Docker Desktop para macOS y Windows el hostname ya resuelve y el flag no estorba):

```bash
docker run -d \
  --add-host=host.docker.internal:host-gateway \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  --name mibici-service \
  mibici-api:latest
```

Equivalente con el `Makefile` (contenedor llamado `mibici-api`, en primer plano):

```bash
make docker-run
```

La documentación interactiva de la API queda en [http://localhost:8000/docs](http://localhost:8000/docs) y el chequeo de estado responde en `http://localhost:8000/health`.

### Conexión del contenedor con MLflow

El contenedor llega al servidor MLflow de la máquina anfitriona por el hostname `host.docker.internal`. Para que funcione, hacen falta tres cosas:

| Dónde | Qué | Para qué |
|---|---|---|
| Servidor (host) | `MLFLOW_SERVER_ALLOWED_HOSTS="localhost:5000,127.0.0.1:5000,host.docker.internal:5000"` | Sin ella MLflow rechaza el `Host` del contenedor (protección DNS rebinding) con `403 Invalid Host header` |
| Servidor (host) | `--artifacts-destination ./mlruns` | Sirve los artefactos por HTTP; el `artifact_uri` guardado es una ruta local del host que no existe dentro del contenedor |
| Contenedor | `MLFLOW_TRACKING_URI=http://host.docker.internal:5000` | Ya viene en el `Dockerfile`; se puede sobrescribir con `-e` |
| Contenedor (Linux) | `--add-host=host.docker.internal:host-gateway` | En Linux el hostname no existe por defecto; Docker Desktop (Windows/macOS) ya lo resuelve, y el flag no estorba |

Levantar el servidor:

```bash
# bash / macOS / Linux
MLFLOW_SERVER_ALLOWED_HOSTS="localhost:5000,127.0.0.1:5000,host.docker.internal:5000" \
  mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns --port 5000
```
```powershell
# PowerShell
$env:MLFLOW_SERVER_ALLOWED_HOSTS = "localhost:5000,127.0.0.1:5000,host.docker.internal:5000"
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns --port 5000
```
```cmd
:: CMD (sin comillas ni espacios)
set MLFLOW_SERVER_ALLOWED_HOSTS=localhost:5000,127.0.0.1:5000,host.docker.internal:5000
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns --port 5000
```

Verificar desde el contenedor que los tres modelos `@champion` se resuelven y cargan:

```bash
docker run --rm --add-host=host.docker.internal:host-gateway \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  --mount type=bind,source="$(pwd)/scripts",target=/app/scripts,readonly \
  mibici-api:latest python scripts/verificar_registry.py
```

Errores comunes:

| Síntoma | Causa | Solución |
|---|---|---|
| `403 Invalid Host header` | Falta `MLFLOW_SERVER_ALLOWED_HOSTS` | Reiniciar el servidor con la variable |
| `No such artifact: ''` o fallo al descargar | Servidor sin `--artifacts-destination` | Reiniciar el servidor con `--artifacts-destination ./mlruns` |
| `Connection refused` | Servidor apagado, o lanzado con `--host 127.0.0.1` en Linux | Levantarlo; en Linux usar `--host 0.0.0.0` |
| `403 Server: AirTunes` (macOS) | AirPlay ocupa el puerto 5000 | Apagar el receptor AirPlay o usar otro puerto |
| `"docker" no se reconoce` (Windows) | La terminal se abrió antes de instalar Docker Desktop | Reiniciar VS Code o la terminal |

**Alternativa evaluada: empaquetar el modelo en la imagen.** Consiste en descargar los artefactos del registry al construir la imagen y cargarlos desde disco, sin servidor MLflow en tiempo de ejecución.

- A favor: la imagen es autocontenida y arranca sin depender del host.
- En contra: cada versión nueva del modelo obliga a reconstruir la imagen, y se pierde la trazabilidad en vivo con el alias `@champion`.

Elegimos la carga dinámica porque promover un modelo en el registry basta para que el servicio lo tome con solo reiniciar el contenedor.

La evidencia de esta validación está en [`reports/api/evidencia_conexion_contenedor.md`](reports/api/evidencia_conexion_contenedor.md).

### Ejemplo de consumo de la API

Salida de `make docker-test` contra el contenedor en marcha:
```bash
== 1. Health check ==
{"status":"healthy","modelos_cargados":["p1","p2","p3"]}   HTTP 200

== 2. P1 ==
{"demanda_predicha_viajes":5.620645371371248}   HTTP 200

== 3. P2 ==
{"riesgo_desabasto_o_saturacion":1,"probabilidad_riesgo":0.7448901989148815}   HTTP 200

== 4. P3 ==
{"cluster_asignado":2}   HTTP 200

== 5. Payload invalido ==   HTTP 422
```

Consulta el [detalle de la prueba](reports/api/evidencia_actividad8_contenedor.md).

![Petición y respuesta de la API](reports/figures/evidencia_actividad8_contenedor/01_swagger_predict_p1.png)

### Estrategia de despliegue

Se eligió inferencia en tiempo real mediante una API REST síncrona. Cada petición se procesa al recibirse, en lugar de agrupar solicitudes para ejecutarlas por lotes.

### Ubicación del Model Registry

El servicio consulta el Model Registry de MLflow mediante `MLFLOW_TRACKING_URI`. Al iniciar, carga los modelos registrados con el alias `@champion`; por ello, actualizar el modelo en el registry y reiniciar el contenedor permite servir la nueva versión sin reconstruir la imagen.

## Estructura del proyecto

```
├── LICENSE            <- Licencia open-source (MIT)
├── Makefile           <- Comandos de conveniencia (make data, make test, ...)
├── README.md          <- Este archivo
├── data
│   ├── external       <- Datos de terceros
│   ├── interim        <- Datos intermedios transformados
│   ├── processed      <- datasets finales para modelado
│   ├── raw            <- Parquet unificado + CSVs descargados (NO versionados)
│   └── manifest.json  <- Versionado: totales + SHA-256 de cada archivo de datos
│
├── docs               <- Proyecto mkdocs
│   └── clean_doc.md   <- Documentación limpieza de datos
├── models             <- Modelos entrenados y serializados
├── notebooks          <- Notebooks Jupyter (iniciales: <iniciales>-<descripción>)
├── pyproject.toml     <- Metadatos del paquete mibici + config de ruff
├── references         <- Diccionarios de datos, ML Canvas, documentos de diseño
├── reports
│   └── figures        <- Gráficas para el reporte (evidencia MLflow, etc.)
├── reports
│   └── evidencia_pruebas_calidad_seguridad  <- Capturas de pytest, deepchecks y seguridad
│   └── pr_team       <- PR semanal│
├── requirements.txt   <- Dependencias fijadas (reproducibilidad)
│
├── mibici             <- Código fuente
│   ├── config.py          <- Rutas, semilla, corte de datos y partición temporal
│   ├── dataset.py         <- Ingesta: descarga → normalización → Parquet + manifiesto
│   ├── features.py        <- Feature engineering sin fuga temporal
│   ├── modeling
│   │   ├── train.py       <- Entrenamiento P1/P2/P3
│   │   └── predict.py     <- Inferencia
│   ├── reportes_p3.py     <- Regenera summary y figuras de P3 sin binarios
│   └── plots.py           <- Visualizaciones
│
└── tests              <- Pruebas (pytest)
```

## Flujo de trabajo Git

- **Todo cambio entra por PR**: `main` es una rama protegida (`Require pull request before merging`); ningún cambio se fusiona directamente, ni aunque sea un archivo completamente nuevo.
- **VoBo obligatorio**: todo Pull Request necesita la aprobación de **al menos un compañero** antes de mergeearse.
- **Ramas:** `feature/<nombre-corto>` o `fix/<issue>`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `docs:`, `test:`, `chore:`, `fix:`.
- **Requisito de rúbrica:** cada integrante abre **al menos 1 PR** y reviewa **al menos 1 PR** de otro integrante.
- Ciclo de reviews: Gerardo → Gemma → Jesús → smuciton → Gerardo.

## Integrantes y responsabilidades

| Integrante | GitHub | Responsabilidades |
|------------|--------|-------------------|
| Gerardo Lozano | [@laloz60](https://github.com/laloz60) | Infra e ingesta de datos, `features.py`, P1 baseline de regresión, README |
| Gemma Martínez | [@GemmaMartinez](https://github.com/GemmaMartinez) | P2 — baseline de clasificación (#7), evidencia MLflow (#11) |
| Jesús Tafolla | [@JesusTafolla](https://github.com/JesusTafolla) | Clustering P3 (#8), propuesta de modelos y métricas (#9) |
| smuciton | [@smuciton](https://github.com/smuciton) | Definición de datos y condiciones de uso (#2), diagrama de arquitectura (#3), P2 — proxy de riesgo (#6) |
