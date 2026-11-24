# Model Card - Sistema Predictivo MiBici (Equipo 01)

**Proyecto:** `TC5061-Proyecto-Equipo01`  
**Materia:** Operaciones de Aprendizaje Automático (TC5061.10) - Maestría en Inteligencia Artificial Aplicada, Tecnológico de Monterrey  
**Responsables:** Gerardo Carreño García (`@laloz60`), Gemma Alejandra Martínez Santos (`@GemmaMartinez`), Jesús Roberto Tafolla Aranda (`@JesusTafolla`), Sebastian Yair Mucito Nájera (`@smuciton`)  
**Última actualización:** Noviembre de 2026  

---

## 1. Resumen de Modelos en Producción (`@champion`)

El servicio API REST (`mibici/api/`) carga dinámicamente tres modelos registrados en **MLflow Model Registry** mediante el alias `@champion` (descarga vía proxy `mlflow-artifacts:/`), sin empaquetar artefactos binarios dentro de la imagen Docker.

| Pregunta | Nombre en Model Registry | Versión / Alias | Familia / Run de origen | Métrica de selección (`val`) | Endpoint REST |
|---|---|---|---|---|---|
| **P1 (Regresión)** | `mibici-equipo01-p1-demanda` | v2 `@champion` | `RandomForestRegressor` (`p1-candidate-rf-n200`) | `val_mae` (menor es mejor) | `POST /predict/p1-demanda` |
| **P2 (Clasificación)** | `mibici-equipo01-p2-riesgo` | v2 `@champion` | `LogisticRegression` (`C=1.0`, `P2-parent-search-logreg`) | `val_roc_auc` (mayor es mejor) | `POST /predict/p2-riesgo` |
| **P3 (Clustering)** | `mibici-equipo01-p3-grupos` | v2 `@champion` | `KMeans++` (`k=3`, `p3-candidate-k3`) | `val_silhouette_score` (mayor es mejor) | `POST /predict/p3-clustering` |

---

## 2. Uso Previsto y Alcance

### Uso previsto
* **P1 (Demanda horaria):** Estimar el volumen de salidas por estación y hora para apoyar la planeación de rebalanceo de bicicletas.
* **P2 (Riesgo operativo):** Generar alertas tempranas sobre estaciones con probabilidad alta de desabasto de bicicletas o saturación de anclajes.
* **P3 (Segmentación geográfica/uso):** Agrupar las 470 estaciones de la red en perfiles operativos para planeación estratégica y mantenimiento.

### Usos fuera de alcance (Out-of-Scope)
* **No apto para inferencia individual de usuarios:** Los modelos operan exclusivamente a nivel agregado `(estacion_id, hora)` y no deben usarse para perfilar ciclistas individuales.
* **No extrapolar a otras ciudades sin reentrenamiento:** Las distribuciones horarias, topología y capacidades corresponden únicamente al Área Metropolitana de Guadalajara.
* **Eventos extremos no históricos:** Cierres masivos de vialidades, eventos climatológicos atípicos o cambios bruscos en el número de estaciones requieren revisión operativa humana antes de automatizar decisiones.

---

## 3. Datos de Entrenamiento, Linaje y Partición Temporal

* **Fuente:** Datos abiertos oficiales de MiBici / Instituto de Movilidad de Jalisco (`https://mibici.net/es/datos-abiertos`).
* **Cobertura:** **42 meses** postpandemia (`2023-05` a `2026-10`), totalizando **16,011,659 viajes** y **470 estaciones**.
* **Reproducibilidad y linaje:** Los archivos crudos (~1.5 GB) no se versionan en Git; la integridad se verifica mediante hashes SHA-256 en `data/manifest.json` y se registra en la descripción de cada versión en MLflow junto con el `run_id` y el commit SHA.
* **Partición temporal sin fuga (`mibici/config.py`):**
  * `train`: `2023-05` a `2024-12`
  * `val`: `2025-01` a `2025-06` (usado para selección de hiperparámetros y registro en Model Registry)
  * `test`: `2025-07` a `2026-10` (reservado para evaluación final y A/B testing)
* **Validación de ingeniería de características:** Todos los rezagos (`lags`) y medias móviles (`salidas_media_7d`, `neto_media_7d`) usan exclusivamente información pasada, garantizado por la prueba automatizada `tests/test_features.py::test_sin_fuga_de_informacion`.

---

## 4. Evaluación, A/B Testing y Decisiones de Umbral

### Resultados de A/B Testing en P2 (`mibici-equipo01-p2-riesgo`)
Evaluado sobre **4,231,300 filas** del split `test` (`2025-07-01` a `2026-09-30`), contrastando V1 (`@champion`, entrenada hasta `2024-12`, umbral `0.50`) contra V2 (`@ab-candidate`, ventana extendida hasta `2025-06`, umbral `0.70`):

| Métrica | V1 (`@champion`, umbral 0.50) | V2 (`@ab-candidate`, umbral 0.70) | Diferencia ($\Delta$) |
|---|---:|---:|---:|
| **ROC AUC** | `0.8239` | `0.8248` | `+0.0008` |
| **Average Precision** | `0.1283` | `0.1287` | `+0.0005` |
| **F1 macro** | `0.5129` | `0.5638` | `+0.0509` |
| **Recall** | `0.6819` | `0.3773` | `-0.3046` |

* **Sensibilidad al umbral:** Al aplicar el umbral `0.70` a la V1 vigente se obtiene `F1 macro = 0.5640` (virtualmente idéntico a V2 con `0.5638`). El test exacto de McNemar ($p < 0.0001$) confirma diferencia en clasificaciones únicamente por el punto de corte, no por mejora intrínseca del reentrenamiento.
* **Política vigente:** Conservar V1 como `@champion` y evaluar cualquier ajuste de umbral (`0.50` vs `0.70`) junto con el equipo de operaciones en función del costo relativo entre falsas alertas y eventos de desabasto no detectados.

---

## 5. Monitoreo de Drift y Política de Continuous Training

* **Herramienta:** Evidently (`mlflow run . -e drift --env-manager=uv`), comparando muestras reproducibles de 30,000 filas (`seed=42`) entre referencia (`2023-05..2024-12`) y periodo actual (`2025-01..2026-10`).
* **Resultado más reciente:** Solo **4 de 21 variables (19%)** superaron el umbral individual de distancia Wasserstein (`0.10`): `mes` (`0.337`, artefacto de calendario), `estacion_id` (`0.255`, bajas y cambios de capacidad en la red), `neto_media_7d` (`0.133`) y `salidas_media_7d` (`0.118`). No existe drift a nivel dataset (umbral `50%`).
* **Criterio de reentrenamiento (CT):** No se dispara Continuous Training automático solo por *covariate shift* localizado. El reentrenamiento y promoción de `@champion` se activarán únicamente ante degradación sostenida de métricas de desempeño con etiquetas recientes o cambios estructurales mayores en la red de estaciones.

---

## 6. Operación, Seguridad, Privacidad y FinOps

* **Privacidad y ética:** Los datos de MiBici son públicos y anonimizados; no se incorporan identificadores personales (`PII`), género ni año de nacimiento como variables de decisión en las predicciones por estación, mitigando sesgos demográficos.
* **Seguridad de software:** Auditoría automatizada mediante `make security` (`Bandit` sin hallazgos críticos; `pip-audit` limpio tras mitigar `CVE-2026-53995` actualizando a `mlflow==3.17.2`) y validación estricta de entradas en tiempo de ejecución mediante Pydantic (HTTP `422` ante payloads inválidos).
* **Huella operativa y FinOps (AWS EC2 `us-east-1`):**
  * **Despliegue actual (1 réplica):** `$0.53 - $1.55 USD` por millón de inferencias (`t3.medium` / `t3.large`).
  * **Escalamiento (2 réplicas + Nginx):** Reduce `p95` un `89%` en Escenario A (`2,300 -> 260 ms`) y `69%` en B (`1,300 -> 400 ms`), elevando el costo `3.2×-3.6×`.
  * **Recomendación arquitectónica prioritaria:** Desacoplar el servidor MLflow (`~1.9 GB` RAM en reposo) del host de inferencia reduce el costo entre **50% y 75%** (bajando a `t3.small`: `$0.28 - $0.39 USD / 1M`).
