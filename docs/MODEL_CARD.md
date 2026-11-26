# Model Card - Sistema Predictivo MiBici (Equipo 01)

| Metadato | Valor |
|---|---|
| **Proyecto** | `TC5061-Proyecto-Equipo01` |
| **Versión del documento** | `v1.0` |
| **Fecha de última actualización** | `2026-11-25` |
| **Materia** | Operaciones de Aprendizaje Automático (TC5061.10) - Maestría en Inteligencia Artificial Aplicada, Tecnológico de Monterrey |
| **Repositorio** | https://github.com/laloz60/TC5061-Proyecto-Equipo01 |
| **Responsables** | Gerardo Carreño García (`@laloz60`, P1 e infra), Gemma Alejandra Martínez Santos (`@GemmaMartinez`, P2 y carga), Sebastian Yair Mucito Nájera (`@smuciton`, P2 proxy, Registry y FinOps), Jesús Roberto Tafolla Aranda (`@JesusTafolla`, P3, CI/CD y gobernanza) |

---

## 1. Resumen de Modelos y Linaje en Model Registry

El servicio API REST (`mibici/api/`) carga dinámicamente tres modelos registrados en **MLflow Model Registry** mediante el alias `@champion` (descarga vía proxy `mlflow-artifacts:/`), sin empaquetar artefactos binarios dentro de la imagen Docker. La verificación completa de versiones, tags y consumo de la API se documenta en `reports/registry_issue191.md`.

| Pregunta | Nombre en Model Registry | Versión / Alias | Estado (`stage`) | Familia / Run de origen | Métrica de selección (`val_*`) | Commit | Endpoint REST |
|---|---|---|---|---|---|---|---|
| **P1 (Regresión)** | `mibici-equipo01-p1-demanda` | v3 `@champion` | `production` | `RandomForestRegressor` — `p1-candidate-rf-n200` (`e2ee147a…`) | `val_mae`: `0.8079` | `97b7e22` | `POST /predict/p1-demanda` |
| **P2 (Clasificación)** | `mibici-equipo01-p2-riesgo` | v4 `@champion` | `production` | `LogisticRegression` (`C=0.1`) — `P2-parent-search-logreg` (`3d8fdf6a…`) | `val_roc_auc`: `0.8293` | `97b7e22` | `POST /predict/p2-riesgo` |
| **P2 (Candidato A/B)** | `mibici-equipo01-p2-riesgo` | v3 `@ab-candidate` | `evaluacion` *(no en producción)* | `LogisticRegression` — `P2-v2-ventana-extendida` (ventana `2023-05` a `2025-06`, umbral `0.70`) | Evaluado en A/B test sobre `test` (`reports/ab_testing_p2.md`) | — | No expuesto en API |
| **P3 (Clustering)** | `mibici-equipo01-p3-grupos` | v3 `@champion` | `production` | `KMeans++` (`k=5`) — `p3-candidate-k5` (`ab2c73d6…`) | `val_silhouette_score`: `0.3268` | `97b7e22` | `POST /predict/p3-clustering` |

**Linaje de datos y ciclo de vida en MLflow:**
* **Commit de entrenamiento (`@champion` P1, P2 y P3):** `97b7e22236b4a974e4a8d0b320026dca47b07822` (`97b7e22`).
* **Hash de `data/manifest.json`:** `7db3fa3702f6b57f` (registrado en P2 v4 y P3 v3; en P1 v3 el run original `e2ee147a…` no registró este parámetro en MLflow).
* **Hash del Parquet unificado:** `ed2f4783e3a3ec77` (15,376,153 viajes sin los 758 duplicados exactos de 2026-09; registrado en P2 v4 y P3 v3).
* **Tags obligatorios por versión:** Cada `@champion` lleva `stage=production`, `pregunta` (`P1`/`P2`/`P3`), `metrica_seleccion` y `valor_metrica`. Las versiones v2 anteriores (registradas antes de la selección por métricas `val_*`) se conservan con `stage=archived`, y P2 v3 (`@ab-candidate`) se mantiene con `stage=evaluacion` sin recibir tráfico.

---

## 2. Uso Previsto y Alcance

### Uso previsto
* **P1 (Demanda horaria):** Estimar el volumen de salidas por estación en la siguiente hora para apoyar la planeación de rebalanceo de bicicletas.
* **P2 (Riesgo operativo):** Generar alertas de apoyo sobre estaciones con probabilidad alta de desabasto de bicicletas o saturación de anclajes cuando el costo de revisar una falsa alerta sea bajo.
* **P3 (Segmentación de estaciones):** Explorar grupos de estaciones según su perfil horario normalizado para análisis estratégico y operación de red.

### Usos fuera de alcance (*Out-of-Scope*)
* **No apto para acciones automáticas costosas sin revisión humana en P2:** Con una precisión de ~11% en `test`, las alertas deben operar como señal de apoyo y no disparar despachos costosos de forma automática.
* **No apto para segmentación estática permanente en P3:** La caída del *silhouette* entre validación y prueba obliga a recalibrar periódicamente los grupos con datos recientes.
* **No apto para inferencia individual ni extrapolación geográfica:** Los modelos operan exclusivamente a nivel `(estacion_id, hora)` sobre el Área Metropolitana de Guadalajara y no modelan cierres masivos ni eventos climatológicos extremos.

---

## 3. Datos de Entrenamiento, Calidad y Partición Temporal

* **Fuente:** Datos abiertos oficiales de MiBici / Instituto de Movilidad de Jalisco (`https://mibici.net/es/datos-abiertos`).
* **Cobertura:** **42 meses** postpandemia (`2023-05` a `2026-10`), con **16,011,659 viajes** y **470 estaciones** en la nomenclatura vigente.
* **Partición temporal sin fuga (`mibici/config.py`):**
  * `train`: `2023-05` a `2024-12`
  * `val`: `2025-01` a `2025-06` (selección de hiperparámetros y promoción al Model Registry)
  * `test`: `2025-07` a `2026-10` (evaluación final y A/B testing; nunca usada para seleccionar modelos)
* **Calidad y prevención de fuga:** Los datos crudos no se versionan en Git; su integridad se controla con SHA-256 en `data/manifest.json`. Todos los rezagos y medias móviles usan exclusivamente información pasada (`tests/test_features.py::test_sin_fuga_de_informacion`), y los 758 duplicados exactos detectados por `DeepChecks` se eliminan durante la generación de características.

---

## 4. Evaluación de Modelos (P1, P2 y P3) y A/B Testing

### 4.1 Comparativa Baseline vs. Candidato (`reports/baseline_vs_candidato.md`)

**P1 — Demanda horaria por estación (Regresión)**  
Baseline: `P1-baseline` (promedio de la misma hora en las 4 semanas previas). Candidato (`@champion` v3): `p1-candidate-rf-n200` (menor `val_mae` entre candidatos con métricas de validación).

| Métrica | Baseline `val` | Candidato `val` | Delta `val` | Baseline `test` | Candidato `test` | Delta `test` |
|---|---:|---:|---:|---:|---:|---:|
| **MAE** | `0.8476` | `0.8079` | `-4.7%` | `0.8422` | `0.7860` | `-6.7%` |
| **RMSE** | `1.5011` | `1.3795` | `-8.1%` | `1.5882` | `1.4399` | `-9.3%` |
| **R²** | `0.6537` | `0.7075` | `+8.2%` | `0.6011` | `0.6681` | `+11.1%` |

*Lectura:* El candidato mejora las tres métricas en ambas ventanas; el mayor avance en RMSE indica menor frecuencia de errores grandes (en MAE la ganancia en `test` es de `0.056` viajes/hora). `GradientBoostingRegressor` obtuvo `test_mae = 0.7836`, pero no se eligió por no registrar métricas `val_*` (para evitar fuga de selección sobre `test`).

**P2 — Riesgo de vaciado o saturación (Clasificación)**  
Baseline: `P2-baseline-dummy` (nunca predice riesgo). Candidato (`@champion` v4): `P2-parent-search-logreg` (`C=0.1`, umbral `0.7` elegidos en validación).

| Métrica | Baseline `val` | Candidato `val` | Delta `val` | Baseline `test` | Candidato `test` | Delta `test` |
|---|---:|---:|---:|---:|---:|---:|
| **ROC AUC** | `0.5000` | `0.8293` | `+65.9%` | `0.5000` | `0.8240` | `+64.8%` |
| **F1 macro** | `0.4905` | `0.5832` | `+18.9%` | `0.4935` | `0.5641` | `+14.3%` |
| **Precisión** | `0.0000` | `0.1528` | `n/a` | `0.0000` | `0.1149` | `n/a` |
| **Recall** | `0.0000` | `0.4253` | `n/a` | `0.0000` | `0.3715` | `n/a` |

*Lectura:* El candidato discrimina eventos de riesgo (`ROC AUC = 0.8240` en `test`), detectando ~37% de los casos positivos con ~11.5% de precisión. El umbral `0.7` quedó en el límite superior de la malla evaluada (`0.3` a `0.7`), por lo que umbrales superiores podrían ofrecer otro balance precisión/recall.

**P3 — Agrupamiento de estaciones (Clustering)**  
Baseline: `p3-baseline-k5` (`KMeans` aleatorio, `n_init=1`). Candidato (`@champion` v3): `p3-candidate-k5` (`k-means++`, `n_init=10`). Se comparan con el mismo `K=5` porque el *silhouette* no es comparable entre distintos valores de `K`.

| Métrica | Baseline `val` | Candidato `val` | Delta `val` | Baseline `test` | Candidato `test` | Delta `test` |
|---|---:|---:|---:|---:|---:|---:|
| **Silhouette** | `0.3137` | `0.3268` | `+4.2%` | `0.1076` | `0.1366` | `+27.0%` |

*Lectura:* `k-means++` supera al baseline en ambas ventanas (`+0.029` absoluto en `test`), pero la caída pronunciada del *silhouette* entre validación (`0.3268`) y `test` (`0.1366`) confirma que los perfiles de estación cambian en el tiempo y requieren revisión periódica.

### 4.2 A/B Testing en P2 (`mibici-equipo01-p2-riesgo`)

Evaluado sobre **4,231,300 filas** del split `test` (`2025-07-01` a `2026-09-30`), contrastando el modelo base del experimento (V1, entrenado hasta `2024-12` con umbral `0.50`, actualmente `v2` archivada tras promover `v4` por validación) frente al retador V2 (`v3` `@ab-candidate`, `P2-v2-ventana-extendida`, ventana `2023-05-01` a `2025-06-30`, umbral `0.70`):

| Métrica | V1 (umbral 0.50) | V2 (`@ab-candidate` v3, umbral 0.70) | Diferencia ($\Delta = V2 - V1$) |
|---|---:|---:|---:|
| **ROC AUC** | `0.8239` | `0.8248` | `+0.0008` |
| **Average Precision** | `0.1283` | `0.1287` | `+0.0005` |
| **Precisión** | `0.0805` | `0.1144` | `+0.0339` |
| **Recall** | `0.6819` | `0.3773` | `-0.3046` |
| **F1 macro** | `0.5129` | `0.5638` | `+0.0509` |

* **Sensibilidad al umbral:** V1 evaluada con umbral `0.70` alcanza `F1 macro = 0.5640` (vs. `0.5638` de V2), y V2 con umbral `0.50` obtiene `0.5123` (vs. `0.5129` de V1). Aunque el test exacto de McNemar es significativo ($p < 0.0001$; `32,983` aciertos exclusivos de V1 vs. `527,409` de V2), la diferencia proviene del umbral y no del reentrenamiento.
* **Decisión vigente:** Mantener `@ab-candidate` (`v3`) con `stage=evaluacion` sin promoverlo a producción y servir `@champion` (`v4`), acordando con operación cualquier ajuste de umbral según el costo relativo de falsas alertas frente a desabastos no detectados.

---

## 5. Limitaciones, Sesgos y Mitigaciones

1. **Cobertura temporal y cambios de demanda:** La serie comienza en mayo de 2023 (postpandemia), pero hacia finales de 2026 el volumen rompió la banda histórica (`315–420k` viajes/mes), alcanzando `490k` en septiembre (`+25%` vs. agosto) y `635,532` en octubre (`+51%` vs. octubre 2025). Además, la cobertura desigual entre referencia (20 meses) y periodo actual (22 meses) introduce artefactos de calendario en la variable `mes`.
2. **Estaciones dadas de baja y reconfiguración de la red:** La nomenclatura pasó de **484 a 470 estaciones** (**14 dadas de baja** y **11 con cambio de capacidad**), y 33 filas de la muestra actual carecen de registro en nomenclatura. Esto desplaza `estacion_id` (`Wasserstein = 0.255`) y las medias móviles semanales (`salidas_media_7d`, `neto_media_7d`), y explica la degradación del *silhouette* de P3 en `test`. *Mitigación:* actualizar el catálogo de estaciones y reevaluar P3 periódicamente.
3. **Desbalance severo de la etiqueta en P2:** Los eventos de riesgo son minoritarios, generando baja precisión (`0.0805` con umbral `0.50`; `0.1144–0.1149` con umbral `0.70`) y `Average Precision = 0.1283`. Subir el umbral mejora `F1 macro`, pero reduce el `Recall` en `-30.5` puntos porcentuales; además, `0.70` quedó en el extremo de la malla probada. *Mitigación:* evaluar con `ROC AUC`, `PR-AUC` y `F1 macro`, nunca con *accuracy*, y calibrar el umbral con costos operativos reales.
4. **Dependencia del proxy de riesgo en P2:** Los datos abiertos no incluyen inventario en tiempo real de bicicletas/anclajes; la etiqueta aproxima el riesgo a partir de flujos históricos (salidas, llegadas, saldo neto y capacidad). Intervenciones manuales de rebalanceo no visibles en los viajes pueden sesgar el proxy.
5. **Prevención de *training-serving skew* en P3:** El modelo de clustering se entrenó con perfiles horarios normalizados (proporciones que suman `1`). Para evitar entradas fuera de distribución en producción, `ClusterInput` (`mibici/api/schemas.py`) normaliza automáticamente conteos a proporciones y rechaza con HTTP `422` valores negativos o perfiles sin viajes (6 pruebas en `tests/test_api_schemas.py`).
6. **Actualizaciones retroactivas de la fuente de datos:** MiBici puede modificar CSVs históricos ya publicados (p. ej., 26 viajes adicionales en 2023-05 respecto al manifiesto original), provocando variaciones menores al regenerar el Parquet desde cero. *Mitigación:* versionar `data/manifest.json` como referencia oficial del equipo.
7. **Privacidad y sesgos demográficos:** Los campos `genero` y `anio_nacimiento` presentes en los CSV crudos se excluyen por completo del modelado y de la API; ninguna predicción utiliza datos personales (`PII`) ni atributos sensibles.

---

## 6. Monitoreo de Drift y Política de Continuous Training

* **Monitoreo de drift:** Ejecutado con **Evidently** (`mlflow run . -e drift --env-manager=uv`), comparando muestras reproducibles de `30,000` filas (`seed=42`) entre referencia (`2023-05-01` a `2024-12-31`) y actual (`2025-01-01` a `2026-10-31`).
* **Hallazgo vigente:** **4 de 21 variables (19%)** superaron el umbral individual (`0.10`): `mes` (`0.337`), `estacion_id` (`0.255`), `neto_media_7d` (`0.133`) y `salidas_media_7d` (`0.118`). Al no alcanzar el umbral global del `50%`, no existe drift a nivel dataset.
* **Política de Continuous Training (`mlflow run . -e continuous_training`):**
  1. **Disparador:** Lee `reports/drift_report.json` generado por Evidently; no activa reentrenamiento automático por *covariate shift* localizado. Solo dispara reentrenamiento cuando `dataset_drift == true` (`≥50%` de columnas con drift) o en ejecución controlada con `force_retrain=true`.
  2. **Reentrenamiento y versionado:** Reentrena P2 con ventana temporal extendida (`2023-05` a `2025-03`) y valida en `2025-04..2025-06` (manteniendo `test` intacto), registrando la nueva versión en el Model Registry con linaje completo y sin alterar `@ab-candidate`.
  3. **Regla de promoción:** Compara la candidata contra `@champion` usando la misma métrica de validación (`val_roc_auc`). El alias `@champion` y el tag `stage=production` solo se mueven si la candidata supera estrictamente al vigente (archivando la versión previa como `stage=archived`); de lo contrario, queda marcada como `stage=evaluation`.

---

## 7. Operación, Seguridad y FinOps

* **CI/CD y pruebas:** Pipeline en GitHub Actions (`.github/workflows/ci.yml`) con 5 *jobs*: `quality` (Ruff), `tests` (matriz `ubuntu-latest` y `macos-latest`), `docker-build` (Buildx con caché `gha` y publicación en GHCR), `api-smoke` (Docker Compose con `MIBICI_MOCK_REGISTRY=1`) y `notify`.
* **Seguridad:** `make security` ejecuta `Bandit` (5 alertas bajas justificadas por llamadas internas a `git`/Python) y `pip-audit` limpio tras mitigar `CVE-2026-53995` actualizando a `mlflow==3.17.2`. `.env`, `mlflow.db`, `mlruns/` y `data/raw/` están excluidos en `.gitignore`.
* **Desempeño y FinOps (AWS EC2 `us-east-1`, precios al 19/11/2026):**
  * **1 réplica:** `t3.medium` (escenarios B y C) o `t3.large` (A), `0.53–1.55 USD / 1M` inferencias *on-demand* (`0.20–0.59 USD` en *spot*), `0.00%` de errores. Verificación final del sistema integrado (Escenario A, `reports/load/operacion_final_issue195.md`): `2,434` peticiones, `20.34 RPS`, `p50 = 29 ms`, `p95 = 85 ms`, `0.00%` fallas.
  * **2 réplicas + Nginx (`docker-compose.scale.yml`):** Reduce `p95` un `89%` en A (`2,300 → 260 ms`) y `69%` en B (`1,300 → 400 ms`), requiriendo `t3.xlarge` por CPU en B y C (`1.28–2.00 USD / 1M` *on-demand*).
  * **Optimización prioritaria:** Desacoplar `mlflow` (`1.8–1.9 GiB` RAM en reposo, ~63–66% del host) del nodo de inferencia permite bajar a `t3.small`, reduciendo el costo un **50%** en B/C (`0.56 → 0.28 USD/1M` en B) y **75%** en A (`1.55 → 0.39 USD/1M`).