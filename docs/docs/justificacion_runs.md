## Justificación y Selección de los Mejores Modelos 

Justificación técnica, de negocio y operativa de los modelos seleccionados para cada pregunta del proyecto MiBici, respaldada por la experimentación sistemática en MLflow Tracking.

### 1. Pregunta 1: Pronóstico de Demanda de Viajes (`P1`)
* **Modelos Evaluados:** TODO
* **Modelo Seleccionado:** TODO
* **Justificación:** TODO
  
---

### 2. Pregunta 2: Clasificación de Riesgo de Saturación / Desabastecimiento (`P2`)
* **Modelos Evaluados:** Baseline trivial (`DummyClassifier`) vs. Candidatos de Regresión Logística optimizados mediante búsqueda con *nested runs* (`C` de 0.1 a 10.0 con clases balanceadas).
* **Modelo Seleccionado:** **Regresión Logística Regularizada ($C=10.0$)** *(Run ID: `350d144e8665446db9aed28219b864a2`)*.
* **Justificación:**
  * **Desempeño vs. Baseline:** El baseline obtuvo un `test_roc_auc` de **0.500** (azar puro), mientras que la Regresión Logística escaló el `test_roc_auc` hasta **0.824** y el `test_f1_macro` a **0.513**, demostrando capacidad real de aprendizaje predictivo.
  * **Decisión de Negocio:** Maximizar el ROC-AUC es crítico para minimizar los falsos negativos en escenarios de desabastecimiento (estaciones que se quedan sin bicicletas o anclajes). Al priorizar la sensibilidad y la precisión en clases minoritarias de riesgo, se protege el nivel de servicio al usuario.
  * **Estabilidad y Costo:** La búsqueda con *nested runs* demostró que la Regresión Logística es robusta y estable ante diferentes valores de regularización $C$. Su costo computacional de entrenamiento es menor a 7 segundos y su inferencia es prácticamente instantánea.

---

### 3. Pregunta 3: Agrupamiento Espacial y Perfiles Horarios de Estaciones (`P3`)
* **Modelos Evaluados:** Baseline K-Means con inicialización aleatoria (`init="random"`, `n_init=1`) vs. Candidato K-Means++ con inicialización robusta (`init="k-means++"`, `n_init=10`, $K=5$).
* **Modelo Seleccionado:** **K-Means++ ($K=5$)** *(Run ID: `ae57d281f2674a16a998e7801a95720a`)*.
* **Justificación:**
  * **Desempeño y Estabilidad:** K-Means++ arrojó un `test_silhouette_score` de **0.146**, con la ventaja crucial de eliminar la estocasticidad de la inicialización aleatoria. Garantiza reproducibilidad estricta y evita mínimos locales indeseados en los centroides de movilidad.
  * **Decisión de Negocio:** Segmentar el sistema en $K=5$ grupos espacio-temporales permite diseñar políticas de rebalanceo y expansiones de infraestructura adaptadas a la vocación real de las estaciones (ej. zonas residenciales, de oficinas o universitarias), facilitando la estrategia operativa urbana.
  * **Costo de Entrenamiento:** Con un tiempo de ejecución de apenas **4.4 segundos**, es altamente eficiente y escalable para actualizaciones periódicas del catálogo de estaciones.

