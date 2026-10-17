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

- **Fuente:** [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos) — CSVs mensuales de viajes + nomenclatura de estaciones.
- **Cobertura:** 2023-05 a 2026-09 (**41 meses**, corte postpandemia decidido por el equipo: fin de la emergencia sanitaria COVID-19, mayo 2023) → **15,376,127 viajes** y **484 estaciones**.
- **Lo que se versiona en Git:** solo `data/manifest.json` (hash SHA-256 de cada CSV, del Parquet unificado y de la nomenclatura). Los CSVs (~1.2 GB) y el Parquet (321 MB) **no se suben al repositorio**; se regeneran con la ingesta y el manifiesto garantiza que sean idénticos.
- **Partición temporal sin fuga** (definida en `mibici/config.py`, nunca se mezclan meses futuros con el entrenamiento):

| Split  | Rango |
|--------|-------|
| train  | 2023-05 .. 2024-12 |
| val    | 2025-01 .. 2025-06 |
| test   | 2025-07 .. 2026-09 |

## Requisitos

- Python 3.12
- ~2 GB de espacio libre (CSVs + Parquet)

## Instalación

```bash
git clone https://github.com/laloz60/TC5061-Proyecto-Equipo01.git
cd TC5061-Proyecto-Equipo01
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # instala también el paquete mibici (-e .)
```

## Descarga de datos

```bash
python -m mibici.dataset               # primera vez: descarga los 41 CSVs y unifica
python -m mibici.dataset --skip-download  # CSVs ya presentes: solo re-unifica
python -m mibici.dataset --force       # regenera aunque exista el Parquet
python -m mibici.dataset --limit 2     # prueba rápida con 2 meses (escribe todo aparte)
```

Salidas:

```
data/raw/mibici_viajes_2023_05_a_2026_09.parquet   # Parquet unificado (no versionado)
data/raw/nomenclatura_estaciones.parquet           # 484 estaciones: capacidad, lat/lon
data/manifest.json                                 # versionado: totales + SHA-256
```

Con `--limit` los archivos de prueba salen en `data/raw/mibici_viajes_limit.parquet` y `data/manifest_limit.json`: no se versionan y no reemplazan a los de arriba, así que el manifiesto completo solo lo escribe la ingesta completa.

Verificación de que tus datos son idénticos a los del equipo:

```bash
pytest            # pruebas de ingesta, manifiesto y features
```

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

```bash
mlflow ui                             # UI en http://127.0.0.1:5000
```

En un clon nuevo MLflow arranca vacío: `mlflow.db` y `mlruns/` no se versionan, así que los runs se regeneran con los comandos de esta sección (las capturas de evidencia están en `reports/figures/mlflow_evidencia/`).

- Experimento: `mibici-equipo01`
- Convención de runs: nombre `P1-baseline-ridge`, `P2-baseline-logreg`, `P3-kmeans`; tags `equipo=01`, `pregunta=P1|P2|P3`, `commit=<sha>`; params incluyen rango de fechas y hash del manifiesto; al menos 1 artefacto por run.
- Entrenamiento: `python -m mibici.modeling.train` (falta P2). Antes de P1 correr `python -m mibici.features` para generar las features.
- **P1 (Regresión)** — baseline `ridge` y candidato `rf`:
  ```bash
  python -m mibici.modeling.train p1 --model ridge
  python -m mibici.modeling.train p1 --model rf
  ```
- **P3 (Clustering)** — baseline (K-Means con inicialización aleatoria):
  ```bash
  python mibici/modeling/train.py p3
  ```

Las capturas de pantalla de los *runs*, artefactos y comparación de modelos se encuentran en:
```text
reports/figures/mlflow_evidencia/
```

## Buenas prácticas

- **Métricas**: se nombran con el conjunto donde se calculan, `<split>_<métrica>` — `test_rmse`, `test_roc_auc`, `test_silhouette_score` (ver `mibici/modeling/train.py`).
- **Notebooks**: convención CCDS `notebooks/<número>-<descripción>.ipynb` (p. ej. `1.0-iniciales-descripcion.ipynb`). Solo exploración: lo que se reutiliza baja a los módulos (`mibici/features.py`, `mibici/modeling/`). Sin datos personales de usuarios en las gráficas exportadas.
- **Credenciales y datos personales**: nunca se suben a MLflow ni al repo. Viven en `.env`, que está en `.gitignore`; los runs solo registran params y tags públicos.
- **Config centralizada**: rutas, cortes de fechas y `SEED` se leen de `mibici/config.py`, sin duplicarlos en `mibici/dataset.py` ni en `mibici/modeling/`.
- **`.gitignore`**: excluye los datos MiBici (`/data/*`, salvo `data/manifest.json`), `mlruns/`, `mlartifacts/`, `.env` y los modelos serializados (`models/*.pkl`, `models/*.joblib`, `models/*.onnx`, `models/*.bin`).
- **Versionado de datos**: `data/manifest.json` guarda el SHA-256 del parquet unificado y de la nomenclatura, con el total de viajes y estaciones. Es lo único de `data/` que entra al repo.

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
├── models             <- Modelos entrenados y serializados
├── notebooks          <- Notebooks Jupyter (iniciales: <iniciales>-<descripción>)
├── pyproject.toml     <- Metadatos del paquete mibici + config de ruff
├── references         <- Diccionarios de datos, ML Canvas, documentos de diseño
├── reports
│   └── figures        <- Gráficas para el reporte (evidencia MLflow, etc.)
│
├── requirements.txt   <- Dependencias fijadas (reproducibilidad)
│
├── mibici             <- Código fuente
│   ├── config.py          <- Rutas, semilla, corte de datos y partición temporal
│   ├── dataset.py         <- Ingesta: descarga → normalización → Parquet + manifiesto
│   ├── features.py        <- Feature engineering sin fuga temporal
│   ├── modeling
│   │   ├── train.py       <- Entrenamiento P1/P2/P3
│   │   └── predict.py     <- Inferencia
│   └── plots.py           <- Visualizaciones
│
└── tests              <- Pruebas (pytest)
```

## Flujo de trabajo Git

- **VoBo obligatorio**: todo cambio —directo o por rama— necesita la aprobación de **al menos un compañero** (aprobación del PR o comentario `LGTM`).
- **Rama + PR**: cambios sobre código que ya existe (modificar, refactorizar, corregir).
- **Push directo a `main`**: solo cuando el cambio es completamente nuevo (archivos que aún no existen). *Nota: `main` tiene activo "Require pull request before merging", así que GitHub rechaza el push directo para quien no es admin; en ese caso se abre un PR y se pide el VoBo igual.*
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
