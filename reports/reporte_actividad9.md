# Actividad 9: Monitoreo de drift, A/B testing y Continuous Training

Equipo 01 · TC5061 MLOps · Proyecto MiBici

Este documento reúne los tres resultados de la actividad: el reporte de drift y calidad de datos, la comparación A/B entre dos versiones del modelo P2 (Riesgo) y la recomendación sobre Continuous Training. Se puede leer sin abrir el HTML de Evidently.

| Resultado | Cómo se reproduce | Archivos |
|---|---|---|
| Drift y calidad de datos | `mlflow run . -e drift --env-manager=uv` | `reports/drift_report.html`, `reports/drift_report.json` |
| A/B testing P2 | `mlflow run . -e ab_test --env-manager=uv` | `reports/ab_testing_p2.md`, `reports/ab_testing_p2.csv`, `reports/figures/ab_testing_p2/` |

---

## 1. Reporte de data drift y calidad de datos

### 1.1 Conjuntos comparados

Los dos conjuntos salen de `data/processed/features_estacion_hora.parquet` y se separan **por fecha**, no por posición, para que ninguna hora del periodo actual entre en la referencia:

| Conjunto | Periodo | Por qué |
|---|---|---|
| Referencia | 2023-05-01 a 2024-12-31 | Es la ventana con la que se entrenaron los modelos vigentes (`config.TRAIN_END`). Representa lo que el modelo "conoce". |
| Actual | 2025-01-01 a 2026-09-30 | Periodo posterior que el modelo no vio al entrenar. Simula los datos que llegan en producción. |

De cada conjunto se toma una muestra de 30,000 filas con `config.SEED = 42` para que el reporte quepa en memoria y sea reproducible. Evidently usa la distancia de Wasserstein normalizada en las variables numéricas y la distancia de Jensen-Shannon en `es_fin_de_semana`. El umbral es 0.1 por variable, y el dataset se marca con drift si al menos el 50 % de las columnas lo tiene.

### 1.2 Resultado general

- Variables analizadas: 21
- Variables con drift: **4 (19 %)**
- Drift a nivel dataset: **no** (el criterio es 50 % de las columnas)

![Vista general del reporte de Evidently](figures/evidencia_data_drift/Data-Drift-UI.png)

### 1.3 Variables con drift

| Variable | Distancia (Wasserstein normalizada) | Lectura |
|---|---:|---|
| `mes` | **0.4157** | Se debe sobre todo a las ventanas elegidas. La referencia cubre mayo a diciembre dos veces (2023 y 2024) y enero a abril una vez; la actual cubre enero a septiembre dos veces y octubre a diciembre una vez. Gran parte del drift es de calendario, no un cambio de comportamiento. |
| `estacion_id` | **0.2367** | Cambió la mezcla de estaciones que aparecen en la muestra: altas y bajas de estaciones y redistribución del uso dentro de la red. |
| `salidas_media_7d` | **0.1379** | El nivel de demanda semanal por estación se movió. Es la variable de mayor peso para predecir demanda (P1) y riesgo (P2). |
| `neto_media_7d` | **0.1288** | Cambió el balance semanal llegadas − salidas por estación, que es la base de la etiqueta de riesgo de P2. |

Las demás 17 variables quedan por debajo de 0.1. Las más altas son `salidas_prom_4sem` (0.078) y `salidas_lag_1h` (0.070). Las variables de hora y día (`hora_del_dia` 0.014, `dia_semana` 0.006, `es_fin_de_semana` 0.002) prácticamente no cambian: el patrón diario y semanal de uso es estable.

![Desplazamiento de la distribución en una variable con drift](figures/evidencia_data_drift/Dezplazamiento-de-la-distribucion.png)

### 1.4 Calidad de datos

| Indicador | Referencia | Actual |
|---|---:|---:|
| Filas (muestra) | 30,000 | 30,000 |
| Columnas | 21 | 21 |
| Valores faltantes | 4,776 | 1,189 |
| Filas duplicadas | 0 | 4 |
| Columnas constantes o vacías | 0 | 0 |

Los faltantes del periodo actual se concentran en los rezagos y promedios móviles (`salidas_prom_4sem`, `neto_prom_4sem`: 358; `*_lag_168h`, `*_media_7d`: 90), que no existen en las primeras horas de cada estación, y en `capacidad`, `latitud` y `longitud` (33), que vienen de estaciones sin registro en la nomenclatura. La referencia tiene más faltantes porque incluye el arranque de la serie en mayo de 2023, cuando todavía no hay historia para calcular rezagos. Los 4 duplicados del periodo actual no afectan al análisis.

---

## 2. A/B testing del modelo P2 (Riesgo)

### 2.1 Versiones comparadas

Las dos versiones están registradas bajo el **mismo nombre** en el Model Registry, `mibici-equipo01-p2-riesgo`:

| | Versión | Alias | Entrenamiento | Umbral |
|---|---|---|---|---:|
| V1 | 2 | `@champion` | Ventana original (train hasta 2024-12) | 0.50 |
| V2 | 4 | `@ab-candidate` | Ventana extendida 2023-05-01 a 2025-06-30 | 0.70 |

La V2 se registró con el alias `ab-candidate` y el tag `stage_of_work=ab_candidate`. Así el alias `champion` no se mueve solo y `register.py` no la promueve por accidente.

### 2.2 Metodología

1. **Un solo conjunto de evaluación.** Las dos versiones se evalúan sobre las mismas 4,231,300 filas del split `test` (2025-07-01 a 2026-09-30, 2.56 % de horas en riesgo), con las mismas features, la misma etiqueta y la misma semilla (42). La única variable del experimento es el modelo.
2. **Sin fuga de información.** La V2 se entrenó solo hasta 2025-06-30, y su `C` y su umbral se eligieron en un holdout interno (2025-04-01 a 2025-06-30). Ninguna fila del set de evaluación participó en el entrenamiento ni en la selección de hiperparámetros de ninguna de las dos versiones.
3. **Métricas.** ROC AUC y average precision miden la calidad del ranking y no dependen del umbral. Precisión, recall y F1 macro se calculan con el umbral propio de cada versión, como operaría en producción. F1 macro es la métrica principal porque la clase en riesgo es muy minoritaria.
4. **Significancia.** La diferencia de aciertos se prueba con el test exacto de McNemar sobre los pares discordantes. Es la prueba adecuada para dos clasificadores evaluados sobre las mismas observaciones.
5. **Control del umbral.** Además se evalúa cada modelo con el umbral del otro, para separar el efecto del reentrenamiento del efecto del corte.

> Nota: el set del A/B (desde 2025-07) es un subconjunto del periodo "actual" del drift (desde 2025-01). Los meses de enero a junio de 2025 quedan dentro del entrenamiento de la V2, que es justamente lo que se busca: un modelo que ya vio parte de la distribución nueva.

### 2.3 Tabla comparativa

| Métrica | V1 (champion) | V2 (ab-candidate) | Δ (V2 − V1) |
|---|---:|---:|---:|
| ROC AUC | 0.8239 | 0.8248 | +0.0008 |
| Average precision | 0.1283 | 0.1287 | +0.0005 |
| Precisión | 0.0805 | 0.1144 | +0.0339 |
| Recall | 0.6819 | 0.3773 | −0.3046 |
| **F1 macro** | 0.5129 | **0.5638** | **+0.0509** |

**McNemar:** 32,983 pares en los que solo acierta V1 frente a 527,409 en los que solo acierta V2; p < 0.0001.

![Métricas V1 vs V2](figures/ab_testing_p2/metricas_v1_vs_v2.png)

![Curvas ROC V1 vs V2](figures/ab_testing_p2/roc_v1_vs_v2.png)

### 2.4 Sensibilidad al umbral

| Configuración | Umbral | F1 macro |
|---|---:|---:|
| V1 con su umbral | 0.50 | 0.5129 |
| V2 con su umbral | 0.70 | 0.5638 |
| V1 con el umbral de la V2 | 0.70 | 0.5640 |
| V2 con el umbral de la V1 | 0.50 | 0.5123 |

### 2.5 Conclusión del A/B

La V2 gana como se desplegaría (F1 macro +0.051, diferencia significativa), pero **la mejora viene del umbral, no del reentrenamiento**. Con el mismo umbral los dos modelos empatan, y el ROC AUC casi no cambia (+0.0008). Reentrenar con la ventana extendida no mejoró la capacidad del modelo para ordenar las horas de riesgo.

Detalle completo en [`ab_testing_p2.md`](ab_testing_p2.md).

---

## 3. Recomendación de Continuous Training

<!-- PENDIENTE: integrar el texto del issue de interpretación (Gemma) cuando esté listo:
     - tabla variable | % drift / test | implicación para P1/P2
     - interpretación estadística
     - veredicto sí/no con su justificación -->

**Insumos que cruza la recomendación:**

- **Drift:** el 19 % de las variables tiene drift, sin drift a nivel dataset. El cambio de `mes` es en buena parte un efecto de las ventanas elegidas. Los cambios con efecto real están en `estacion_id`, `salidas_media_7d` y `neto_media_7d`.
- **A/B:** reentrenar con datos más recientes no mejora el ranking (ΔROC AUC +0.0008). La ganancia práctica sale de recalibrar el umbral.

**Veredicto:** _pendiente_

**Justificación:** _pendiente_
