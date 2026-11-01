# Baseline vs candidato

Comparación del baseline contra el candidato de cada pregunta, en validación y en test. El candidato de P1 y P2 se elige con validación; en P3 se comparan ambos con K=5, porque el silhouette no es comparable entre distintos K. Test solo se reporta.

**Cómo leer el delta:** `(candidato - baseline) / baseline`. En MAE y RMSE, un delta negativo es mejor; en las demás métricas, uno positivo es mejor.

**Datos:** commit `97b7e22`, Parquet local `ed2f4783` (15,376,153 viajes) y features sin los 758 duplicados exactos de 2026-09. Este Parquet difiere del manifiesto versionado, así que otra descarga puede dar valores ligeramente distintos.

## P1: viajes por estación en la siguiente hora

Baseline: `P1-baseline`, promedio de la misma hora en las 4 semanas previas. Candidato: `p1-candidate-rf-n200`, el menor `val_mae` entre los candidatos con métricas de validación.

| Métrica | Baseline val | Candidato val | Delta val | Baseline test | Candidato test | Delta test |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 0.8476 | 0.8079 | -4.7% | 0.8422 | 0.7860 | -6.7% |
| RMSE | 1.5011 | 1.3795 | -8.1% | 1.5882 | 1.4399 | -9.3% |
| R² | 0.6537 | 0.7075 | +8.2% | 0.6011 | 0.6681 | +11.1% |

**Lectura:** el candidato gana en las tres métricas y en ambas ventanas. RMSE mejora más que MAE, lo que indica una reducción mayor en los errores grandes. Conviene cuando esos errores importan para planear la operación. Si basta una estimación simple y sin entrenamiento, la regla sigue siendo razonable, porque la mejora en MAE es de 0.056 viajes por hora en test.

**Límite:** `GradientBoostingRegressor` obtuvo `test_mae` de 0.7836, pero no registra métricas de validación, así que no se eligió. Usar ese resultado para elegirlo mezclaría la selección con test.

## P2: riesgo de vaciado o saturación

Baseline: `P2-baseline-dummy`, que nunca predice riesgo. Candidato: `P2-parent-search-logreg`, con `C=0.1` y umbral 0.7 elegidos en validación.

| Métrica | Baseline val | Candidato val | Delta val | Baseline test | Candidato test | Delta test |
|---|---:|---:|---:|---:|---:|---:|
| ROC AUC | 0.5000 | 0.8293 | +65.9% | 0.5000 | 0.8240 | +64.8% |
| F1 macro | 0.4905 | 0.5832 | +18.9% | 0.4935 | 0.5641 | +14.3% |
| Precisión | 0.0000 | 0.1528 | n/a | 0.0000 | 0.1149 | n/a |
| Recall | 0.0000 | 0.4253 | n/a | 0.0000 | 0.3715 | n/a |

El delta es n/a porque el baseline vale 0.

**Lectura:** el candidato sí distingue el riesgo, mientras que el baseline nunca genera alertas. En test detecta cerca de 37% de los casos, pero solo cerca de 11% de sus alertas son correctas. Conviene como apoyo cuando revisar una falsa alerta cuesta poco; no conviene para activar acciones costosas de forma automática.

**Límite:** el umbral 0.7 quedó en el extremo de la malla probada (0.3 a 0.7), así que un umbral mayor podría dar otro balance.

## P3: grupos de estaciones

Baseline: `p3-baseline-k5`, K-Means con inicialización aleatoria y `n_init=1`. Candidato: `p3-candidate-k5`, k-means++ con `n_init=10`.

| Métrica | Baseline val | Candidato val | Delta val | Baseline test | Candidato test | Delta test |
|---|---:|---:|---:|---:|---:|---:|
| Silhouette | 0.3137 | 0.3268 | +4.2% | 0.1076 | 0.1366 | +27.0% |

**Lectura:** el candidato mejora en ambas ventanas y conviene frente al baseline. El +27.0% de test equivale a solo 0.029 en valor absoluto, y el silhouette baja mucho de validación a test en los dos modelos. Los grupos sirven para explorar patrones, pero no conviene usarlos como segmentación fija sin volver a revisarlos con datos recientes.

## Evidencia

Los runs están en el experimento `mibici-equipo01` de MLflow. Sus figuras están en los artefactos de cada run, en `figures/`.
