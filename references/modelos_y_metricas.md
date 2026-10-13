# Propuesta de Modelos y Métricas Justificada en MiBici

Para cada una de las preguntas del proyecto integrador, se han definido los modelos base (*baselines*), los modelos candidatos y las métricas de evaluación alineadas con la guía de evaluación de modelos de scikit-learn y la toma de decisiones operativas.

---

### P1: Predicción de Demanda de Viajes (Regresión)
- **Baseline sencillo:** Promedio histórico de viajes por estación y hora de la semana (`DummyRegressor` basado en la media).
- **Modelo candidato:** `RandomForestRegressor` o `LightGBM Regressor`.
- **Métrica propuesta:** `test_neg_root_mean_squared_error` (`test_rmse`).
- **Justificación operativa y teórica:** El RMSE mantiene las unidades originales (número de viajes), lo que permite al operador del sistema interpretar directamente el error promedio de bicicletas faltantes o sobrantes por estación. Además, al elevar al cuadrado los errores antes de promediar, penaliza con mayor severidad los errores grandes, evitando fallas críticas en la planeación de la flota.

---

### P2: Riesgo de Desabastecimiento o Saturación (Clasificación)
- **Baseline sencillo:** Clasificador por clase mayoritaria (`DummyClassifier` con estrategia `most_frequent`).
- **Modelo candidato:** `LogisticRegression` (con balance de clases) y `GradientBoostingClassifier`.
- **Métrica propuesta:** `test_roc_auc` y `test_f1_macro`.
- **Justificación operativa y teórica:** Dado que las situaciones de riesgo crítico (estaciones vacías o saturadas) son minoritarias frente a los periodos de operación normal, las clases están desbalanceadas. El AUC-ROC evalúa la capacidad discriminativa del modelo en distintos umbrales, mientras que el F1-Macro pondera de forma equitativa el desempeño en la clase minoritaria de riesgo, evitando falsos negativos que dejen a los ciudadanos sin bicicletas.

---

### P3: Patrones de Uso y Agrupamiento de Estaciones (Clustering)
- **Baseline sencillo:** K-Means con inicialización aleatoria simple (`KMeans` con `init='random'`).
- **Modelo candidato:** K-Means++ con normalización previa de perfiles temporales de uso.
- **Métrica propuesta:** `silhouette_score`.
- **Justificación operativa y teórica:** El puntaje de silueta permite evaluar cuantitativamente qué tan compactos y bien separados están los clústeres formados, garantizando que los grupos de estaciones reflejen comportamientos diferenciados (zonas residenciales, corporativas o recreativas) útiles para la planeación urbana.