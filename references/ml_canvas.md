# Machine Learning Canvas: Proyecto MiBici

## 1. Value Proposition (Propuesta de Valor)
- **Beneficiarios:** Operadores del sistema MiBici, equipo de rebalanceo operativo y planificadores urbanos del AMG.
- **Problema:** Mitigar estaciones vacías y saturación de anclajes, reduciendo la insatisfacción de los usuarios y optimizando costos logísticos de reubicación.
- **Integración:** Tableros operativos y alertas automatizadas que guían el trabajo diario de los operadores.

## 2. Data Sources (Fuentes de Datos)
- Portal de Datos Abiertos de MiBici (CSV mensuales de viajes del periodo post-pandemia **2023-05 a 2026-09**: 41 meses, 15,376,127 viajes, 484 estaciones; ver `mibici/config.py` y `data/manifest.json`).
- Archivos de nomenclatura de estaciones con capacidad de anclajes (`dpcapacity`).

## 3. Data Collection (Recolección y Actualización)
- Ingesta automatizada mediante `dataset.py`, normalizando y almacenando en formato **Parquet**.
- Versionado de datos mediante manifiesto con hashes **SHA-256** (`hashlib`).

## 4. Features (Características)
- Temporales (hora, día de la semana, fin de semana).
- Rezagos temporales (*lags* de 1h y 24h) y medias móviles de flujos netos.
- Geoespaciales y capacidad máxima por estación.

## 5. Building Models (Construcción de Modelos)
- Modelos de Regresión (P1), Clasificación (P2) y Clustering (P3).
- Versionado riguroso mediante **MLflow Tracking** bajo el experimento `mibici-equipo01`.

## 6. Making Predictions (Predicciones)
- Ejecución en modalidad **batch** de forma diaria o semanal.

## 7. Decisions (Decisiones Operativas)
- Enrutamiento dinámico de camionetas para rebalanceo preventivo y expansión de cicloestaciones.

## 8. Impact Simulation (Simulación de Impacto)
- **Hipótesis a validar (sin cuantificación previa):** las alertas de rebalanceo preventivo reducen la frecuencia de estaciones vacías/saturadas y los costos logísticos de reubicación.
- **Cómo se medirá:** contar eventos de estación vacía/saturada en el periodo de prueba (2025-07 a 2026-09) bajo el modelo vs. la línea base (*baseline* ingenuo) y reportar la reducción porcentual en `reports/`. La cifra se llena solo con resultados de los runs en MLflow.

## 9. Monitoring (Monitoreo)
- Seguimiento periódico de métricas (`test_rmse`, `test_roc_auc`, `test_silhouette_score`) y supervisión de deriva de datos.