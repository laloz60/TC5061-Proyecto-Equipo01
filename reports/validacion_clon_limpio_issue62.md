# Validación del README en clon limpio (issue #62)

Bitácora de la ejecución del README desde un clon nuevo: para cada paso, la tarea, el comando que se corrió y la respuesta del log.

- **Fecha:** 22 oct 2026
- **Responsable:** [@smuciton](https://github.com/smuciton)
- **Rama / commit:** `docs/issue-62-readme-mlflow-run` @ `87622a4`
- **Entorno:** Windows 10 Pro, terminal CMD en VS Code, Python 3.12.13 (`.venv`), uv, Modo de programador activado.
- **Tracking:** servidor local `http://127.0.0.1:5000`, experimento `mibici-equipo01` (id 1).

## Resumen

| # | Tarea | Resultado |
|---|-------|-----------|
| 1 | Clonar el repo y cambiar a la rama del PR | ✅ |
| 2 | Crear el entorno e instalar dependencias | ✅ |
| 3 | Levantar el servidor MLflow | ✅ |
| 4 | Definir y validar `MLFLOW_TRACKING_URI` | ✅ |
| 5 | Ingesta de datos | ✅ |
| 6 | Generar features | ✅ |
| 7 | Entrenar los 3 baselines | ✅ |
| 8 | Entrenar los 3 modelos elegidos | ✅ |
| 9 | Registrar modelos en el Model Registry | ✅ |
| 10 | Cargar un modelo registrado | ✅ |
| 11 | Correr `pytest` | ✅ 29 passed (ver nota) |

---

## 1. Clonar el repo y cambiar a la rama del PR

**Comando:**

```cmd
git clone https://github.com/laloz60/TC5061-Proyecto-Equipo01.git
cd TC5061-Proyecto-Equipo01
git switch docs/issue-62-readme-mlflow-run
git log -1 --oneline
```

**Respuesta:**

```text
87622a4 docs(readme): explicar falla de pytest por datos actualizados de MiBici y cómo validar (#62)
```

**Resultado:** ✅ Clon en la rama del PR.

## 2. Crear el entorno e instalar dependencias

**Comando:**

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Respuesta:**

```text
Building wheels for collected packages: mibici
  Building editable for mibici (pyproject.toml) ... done
Successfully built mibici
Successfully installed Flask-3.1.3 ... mlflow-3.17.0 mlflow-skinny-3.17.0 ... scikit-learn-1.9.1 ...
```

**Resultado:** ✅ Dependencias instaladas, incluido el paquete `mibici` en modo editable.

> En un primer intento `pip` falló con `OSError: [Errno 2] No such file or directory: '...\.venv\Lib\site-packages\jupyterlab\galata\...'`: la ruta medía 260 caracteres y Windows tenía desactivadas las rutas largas (`LongPathsEnabled = 0`). Es una configuración de esta máquina, no del proyecto; se resolvió activándolas.

## 3. Levantar el servidor MLflow

**Comando** (terminal 1, se deja abierta):

```cmd
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Resultado:** ✅ UI disponible en `http://127.0.0.1:5000`.

## 4. Definir y validar `MLFLOW_TRACKING_URI`

**Comando** (terminal 2):

```cmd
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
echo [%MLFLOW_TRACKING_URI%]
```

**Respuesta:**

```text
[http://127.0.0.1:5000]
```

**Resultado:** ✅ Variable sin espacios.

> En un intento previo la variable quedó con un espacio al final y `mlflow run` falló con:
>
> ```text
> mlflow.exceptions.InvalidUrlException: Invalid url: http://127.0.0.1:5000 /api/2.0/mlflow/experiments/get-by-name
> ```
>
> Por eso el README ahora incluye la validación con `echo`.

## 5. Ingesta de datos

**Comando:**

```cmd
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e ingest -P limit=0
```

**Respuesta:**

```text
INFO mlflow.utils.virtualenv: Environment C:\Users\Sebastian\.mlflow\envs\mlflow-46b9cb7dbb901e5d7c5da3ee6f13d0f3b7622379 already exists
INFO mlflow.projects.backend.local: === Running command '... python -m mibici.mlflow_entry ingest --force false --skip-download false --limit 0' in run with ID '268b19c83f6848d98125354dd75198cf' ===
INFO     | __main__:main:549 - Catálogo: 41 meses (2023-05 .. 2026-09)
WARNING  | __main__:main:554 - El Parquet unificado ya existe (339 MB, 15,376,153 filas): mibici_viajes_2023_05_a_2026_09.parquet
WARNING  | __main__:main:558 - Regenerarlo vuelve a descargar los 41 CSVs y sobrescribe el archivo actual.
¿Continuar? [y/N]: y
Descargando: 100%|██████████| 42/42 [00:00<00:00, 2055.43it/s]
SUCCESS  | __main__:write_manifest:509 - Manifiesto: ...\data\manifest.json (hash=3a0c89a...)
SUCCESS  | __main__:main:576 - Ingesta completa ...: 15,376,153 viajes, 484 estaciones -> mibici_viajes_2023_05_a_2026_09.parquet
INFO mlflow.projects: === Run (ID '268b19c83f6848d98125354dd75198cf') succeeded ===
```

**Resultado:** ✅ Run `268b19c8` (painted-sloth-797) `FINISHED`. 15,376,153 viajes y 484 estaciones.

> La ingesta ya se había corrido antes con `python -m mibici.dataset` (descarga de los 41 CSVs en ~5 min + unificación ~1.5 min), por eso preguntó `¿Continuar?`. Al responder `y` reutilizó los CSVs (`42/42 [00:00]`) y solo re-unificó (~1.5 min).

## 6. Generar features

**Comando:**

```cmd
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e features
```

**Respuesta:**

```text
INFO     | __main__:build_features:154 - ...
SUCCESS  | __main__:main:176 - Features guardadas ...\data\processed\features_estacion_hora.parquet
INFO mlflow.projects: === Run (ID '794f889515974...') succeeded ===
```

**Resultado:** ✅ Run `794f8895` (polite-boar-620) `FINISHED`, ~1 min. Como indica el README, `ingest` y `features` no registran métricas.

## 7. Entrenar los 3 baselines

### P1 — Ridge

**Comando:**

```cmd
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p1 -P model=ridge
```

**Respuesta:**

```text
... in run with ID 'bc7382678c79458199fb8fff518b6cf1' ===
SUCCESS  | mibici.modeling.regression:train... MAE=0.814 (baseline 0.842), RMSE=1.462, R2=0.662
🏃 View run P1-baseline-ridge at: http://127.0.0.1:5000/#/experiments/1/runs/bc7382678c79458199fb8fff518b6cf1
```

**Resultado:** ✅ ~3 min.

### P2 — Dummy

**Comando:**

```cmd
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p2 -P model=dummy
```

**Respuesta:**

```text
... in run with ID 'ee1113096d2a493a9ce0d6f37af0ba54' ===
INFO     | __main__:train_p2:580 - P2 train ... | val=1,604,720 filas (3.71% en riesgo) | test=4,231,300 filas (2.56% en riesgo)
🏃 View run P2-baseline-dummy at: http://127.0.0.1:5000/#/experiments/1/runs/ee1113096d2a493a9ce0d6f37af0ba54
SUCCESS  | __main__:train_p2:735 - P2 FIN...
```

**Resultado:** ✅ ~3.5 min. ROC AUC 0.5000, F1 macro 0.4935.

### P3 — K-Means aleatorio

**Comando:**

```cmd
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p3 -P model=random
```

**Respuesta:**

```text
... in run with ID '93caace1354f49d1ba85c6b1f3fa63a5' ===
INFO     | __main__:train_model:148 - Entrenando ... (n_init=1) con SEED=42...
SUCCESS  | __main__:evaluate_and_log:199 - ... Silhouette Score: 0.1591
🏃 View run p3-baseline-k5 at: http://127.0.0.1:5000/#/experiments/1/runs/93caace1354f49d1ba85c6b1f3fa63a5
```

**Resultado:** ✅ <1 min.

## 8. Entrenar los 3 modelos elegidos

### P1 — Random Forest (300 árboles)

**Comando:**

```cmd
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p1 -P model=rf -P n_estimators=300
```

**Respuesta:**

```text
... in run with ID 'dcb88cd8841e4a2ca1c9ad9bf6b5ac58' ===
SUCCESS  | mibici.modeling.regression:train... MAE=0.793 (baseline 0.842), RMSE=1.448, R2=0.669
🏃 View run P1-candidato-rf at: http://127.0.0.1:5000/#/experiments/1/runs/dcb88cd8841e4a2ca1c9ad9bf6b5ac58
```

**Resultado:** ✅ ~15 min. Mejora al baseline (MAE 0.793 vs 0.814).

### P2 — Regresión logística

**Comando:**

```cmd
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p2 -P model=logreg
```

**Respuesta:**

```text
... in run with ID 'eb0ce972821c4ee1881a98367d2ab93e' ===
🏃 View run p2-child-logreg-C0.1 at: ...
🏃 View run p2-child-logreg-C1.0 at: ...
🏃 View run p2-child-logreg-C10.0 at: ...
INFO     | __main__:train_p2:686 - Mejor C ...
SUCCESS  | __main__:train_p2:731 - Búsqueda ... F1 Macro en val: 0.526
🏃 View run P2-parent-search-logreg at: http://127.0.0.1:5000/#/experiments/1/runs/eb0ce972821c4ee1881a98367d2ab93e
```

**Resultado:** ✅ ~4.5 min. Run padre con 3 runs hijos (C = 0.1, 1.0, 10.0). Test: ROC AUC 0.8240, F1 macro 0.5133.

### P3 — K-Means++ (n_init=10)

**Comando:**

```cmd
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e train -P question=p3 -P model=k-means++ -P n_init=10
```

**Respuesta:**

```text
... in run with ID '26685f1dd4704d6faaa07fcb5750889a' ===
INFO     | __main__:train_model:148 - Entrenando ... n_init=10) con SEED=42...
SUCCESS  | __main__:evaluate_and_log:199 - ... Silhouette Score: 0.1461
🏃 View run p3-candidate-k5 at: http://127.0.0.1:5000/#/experiments/1/runs/26685f1dd4704d6faaa07fcb5750889a
```

**Resultado:** ✅ <1 min.

> En todos los entrenamientos aparecen advertencias de MLflow (`Inferred schema contains integer column(s)`, `artifact_path is deprecated`, `dataset source ... LocalArtifactDatasetSource`). No afectan el resultado.

## 9. Registrar modelos en el Model Registry

**Comando:**

```cmd
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e register
```

**Respuesta:**

```text
... in run with ID 'fee7012dd9954de3896ec8350b4e7bc1' ===
Successfully registered model 'mibici-equipo01-p1-demanda'.
Created version '1' of model 'mibici-equipo01-p1-demanda'.
SUCCESS  | __main__:_registrar:90 - mibici-equipo01-p1-demanda v1 <- P1-candidato-rf (test_mae=0.7931)
Successfully registered model 'mibici-equipo01-p2-riesgo'.
Created version '1' of model 'mibici-equipo01-p2-riesgo'.
SUCCESS  | __main__:_registrar:90 - mibici-equipo01-p2-riesgo v1 <- P2-parent-search-logreg (test_roc_auc=0.8240)
Successfully registered model 'mibici-equipo01-p3-grupos'.
Created version '1' of model 'mibici-equipo01-p3-grupos'.
```

**Resultado:** ✅ 3 modelos registrados en v1:

| Modelo | Run de origen | Métrica |
|--------|---------------|---------|
| `mibici-equipo01-p1-demanda` | `dcb88cd8` P1-candidato-rf | `test_mae` = 0.7931 |
| `mibici-equipo01-p2-riesgo` | `eb0ce972` P2-parent-search-logreg | `test_roc_auc` = 0.8240 |
| `mibici-equipo01-p3-grupos` | `93caace1` p3-baseline-k5 | `test_silhouette_score` = 0.1591 |

> El aviso `has no artifacts at artifact path 'model', registering model ... instead` es esperado en MLflow 3: el modelo se guarda como *logged model* y el registro lo resuelve solo.

## 10. Cargar un modelo registrado

**Comando:**

```cmd
python -c "import mlflow; m = mlflow.pyfunc.load_model('models:/mibici-equipo01-p2-riesgo/1'); print('OK, run:', m.metadata.run_id)"
```

**Respuesta:**

```text
OK, run: eb0ce972821c4ee1881a98367d2ab93e
```

**Resultado:** ✅ Carga por versión. Ningún modelo tiene alias: `register.py` no asigna `@champion`, como indica el README (pendiente #71).

## 11. Correr `pytest`

**Comando:**

```cmd
pytest
```

**Respuesta:**

```text
platform win32 -- Python 3.12.13, pytest-9.1.1, pluggy-1.7.0
collected 29 items

tests\test_data.py ............
tests\test_features.py ............
tests\test_modeling.py .....

29 passed
```

**Resultado:** ✅ con una aclaración: la ingesta regenera `data/manifest.json` con los datos descargados, así que aquí `pytest` compara el Parquet contra un manifiesto generado de él mismo. Confirma que la ingesta es consistente y que features y modelado funcionan, no que los datos sean idénticos a los del equipo. Contra el manifiesto versionado, la prueba falla con:

```text
FAILED tests/test_data.py::test_unified_parquet_exists_and_is_non_empty - assert 15376153 == 15376127
```

porque MiBici agregó 26 viajes al CSV de 2023-05 (documentado en el README, sección *Descarga de datos*).

---

## Observaciones para el equipo

1. **P3 registra el baseline, no el candidato.** Ningún run de P3 tiene el tag `stage_of_work=candidate`, así que `register.py` busca entre todas las corridas de P3 y elige la de mayor silhouette: el baseline `random` (0.1591) supera al candidato `k-means++` (0.1461). Revisar si es lo esperado o si al run de P3 le falta el tag (@JesusTafolla).
2. **P1 tampoco tiene el tag `stage_of_work`.** Aquí no cambia el resultado (`rf` es el mejor por MAE), pero la selección no pasa por el filtro de candidatos.
3. **Reproducibilidad de P2.** P2 también se entrenó por ejecución directa (`python -m mibici.modeling.train p2`, runs `029de1d1` y `5d75021b`) y las métricas de test son idénticas a las de `mlflow run` (ROC AUC 0.8240, F1 macro 0.5133). Esos runs quedaron en el mismo experimento; `register` eligió el de `mlflow run`.
4. **El entrenamiento modifica figuras versionadas.** Después del pipeline, `git status` muestra modificados `data/manifest.json` y 5 figuras de `reports/figures/` (`matriz_confusion_p2_dummy.png`, `matriz_confusion_p2_logreg.png`, `p1_rf_pred_vs_real.png`, `p1_ridge_pred_vs_real.png`, `p3_clusters_map.png`). No se commitean: `git restore data/manifest.json reports/figures/`.
5. **Mensaje engañoso en la ingesta.** El aviso `Regenerarlo vuelve a descargar los 41 CSVs` no es cierto cuando los CSVs ya existen (se reutilizan, ver paso 5). Se abrió un issue aparte.
