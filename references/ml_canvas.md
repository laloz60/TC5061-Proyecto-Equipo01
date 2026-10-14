# Machine Learning Canvas: Proyecto MiBici

## 1. Value Proposition (Propuesta de Valor)
- **Beneficiarios:** Operadores del sistema MiBici, equipo de rebalanceo operativo y planificadores urbanos del AMG.
- **Problema:** Mitigar estaciones vacías y saturación de anclajes, reduciendo la insatisfacción de los usuarios y optimizando costos logísticos de reubicación.
- **Integración:** Tableros operativos y alertas automatizadas que guían el trabajo diario de los operadores.

## 2. Formulación del problema por pregunta

### P1 — ¿Cuántos viajes se iniciarán en una estación en un periodo futuro? (Regresión)
1. **Qué se predice:** número de viajes que se iniciarán (origen) en cada estación. **Granularidad:** estación × hora. **Horizonte:** la próxima hora (t+1) usando solo información disponible hasta t.
2. **Relevancia para la operación:** el volumen anticipado de salidas define cuántas bicicletas debe tener surtidas cada estación; subestimar deja usuarios sin bicis y sobreestimar deja bicis ociosas.
3. **Usuario y decisión:** el operador del sistema (planeación de flota) decide, cada hora, cuántas bicicletas enviar y a qué estaciones reponer antes del pico de demanda.

### P2 — ¿Una estación estará en riesgo de quedarse sin bicis/anclajes en un periodo futuro? (Clasificación)
1. **Qué se predice:** etiqueta binaria de riesgo por estación (1 = estación vacía o saturada). La etiqueta se construye como **proxy** a partir del saldo neto de viajes (salidas − llegadas) en la ventana futura comparado contra la capacidad de la estación (`dpcapacity`), porque MiBici no publica inventario en tiempo real (diseño del proxy: issue #6). **Granularidad:** estación × hora. **Horizonte:** ventana de 1 a 3 horas siguientes.
2. **Relevancia para la operación:** una estación vacía deja a los usuarios sin bicis y una saturada les impide aparcar; son las fallas más costosas en satisfacción y en costo de reubicación reactiva.
3. **Usuario y decisión:** el equipo de rebalanceo decide qué estación despachar primero y cuántas bicicletas/anclajes libres necesita, priorizando por severidad del riesgo.

### P3 — ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? (Clustering)
1. **Qué se agrupa:** estaciones agrupadas por su perfil horario de demanda de origen (proporción de viajes por hora del día, vector de 24 dimensiones normalizado por estación). **Granularidad:** una fila por estación (perfil agregado del histórico completo). **Horizonte:** no aplica; es un análisis descriptivo sobre datos históricos, sin predicción de futuro.
2. **Relevancia para la operación:** los patrones residenciales, corporativos y recreativos explican por qué ciertas estaciones se vacían a la misma hora y otras se saturan, y permiten diseñar estrategias de rebalanceo diferenciadas por zona.
3. **Usuario y decisión:** la planeación urbana y el equipo de expansión deciden dónde ubicar nuevas cicloestaciones y qué política de rebalanceo aplicar a cada grupo de estaciones.

## 3. Data Sources (Fuentes de Datos)
- Portal de Datos Abiertos de MiBici (CSV mensuales de viajes del periodo post-pandemia **2023-05 a 2026-09**: 41 meses, 15,376,127 viajes, 484 estaciones; ver `mibici/config.py` y `data/manifest.json`).
- Archivos de nomenclatura de estaciones con capacidad de anclajes (`dpcapacity`).

## 4. Data Collection (Recolección y Actualización)
- Ingesta automatizada mediante `dataset.py`, normalizando y almacenando en formato **Parquet**.
- Versionado de datos mediante manifiesto con hashes **SHA-256** (`hashlib`).

## 5. Features (Características)
- Temporales (hora, día de la semana, fin de semana).
- Rezagos temporales (*lags* de 1h y 24h) y medias móviles de flujos netos.
- Geoespaciales y capacidad máxima por estación.

## 6. Building Models (Construcción de Modelos)
- Modelos de Regresión (P1), Clasificación (P2) y Clustering (P3).
- Versionado riguroso mediante **MLflow Tracking** bajo el experimento `mibici-equipo01`.

## 7. Making Predictions (Predicciones)
- Ejecución en modalidad **batch** de forma diaria o semanal.

## 8. Decisions (Decisiones Operativas)
- Enrutamiento dinámico de camionetas para rebalanceo preventivo y expansión de cicloestaciones.

## 9. Impact Simulation (Simulación de Impacto)
- **Hipótesis a validar (sin cuantificación previa):** las alertas de rebalanceo preventivo reducen la frecuencia de estaciones vacías/saturadas y los costos logísticos de reubicación.
- **Cómo se medirá:** contar eventos de estación vacía/saturada en el periodo de prueba (2025-07 a 2026-09) bajo el modelo vs. la línea base (*baseline* ingenuo) y reportar la reducción porcentual en `reports/`. La cifra se llena solo con resultados de los runs en MLflow.

## 10. Monitoring (Monitoreo)
- Seguimiento periódico de métricas (`test_rmse`, `test_roc_auc`, `test_silhouette_score`) y supervisión de deriva de datos.
