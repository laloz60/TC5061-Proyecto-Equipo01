# A/B testing P2 (Riesgo): V1 champion vs V2 ventana extendida

- Fecha: 2026-11-09 | commit `e1e2ce230fcc4e96f1476ca824cf218cb5b964d3`
- Semilla: `42` (misma para entrenar y evaluar)
- **Set único de evaluación**: split `test` (2025-07-01 a 2026-09-30), 4,231,300 filas, 2.56% en riesgo
- V1: `models:/mibici-equipo01-p2-riesgo@champion` (versión 2, run `1cd16f9f`, C=0.1, umbral=0.500)
- V2: `models:/mibici-equipo01-p2-riesgo@ab-candidate` (versión 4, run `890243a0`, C=0.1, umbral=0.700)

## Metodología

1. La V2 se entrenó con la ventana extendida `2023-05-01..2025-06-30` (splits train + val); C y el umbral se eligieron en un holdout interno (`2025-04-01..2025-06-30`) y el modelo final se reajustó sobre toda la ventana.
2. V1 y V2 se evaluaron sobre las MISMAS 4,231,300 filas del split `test`: mismas features, misma etiqueta, misma semilla. La única variable del experimento es el modelo (y su umbral propio).
3. ROC AUC es umbral-independiente; precisión, recall y F1 macro usan el umbral que le corresponde a cada versión (el suyo, elegido en validación, no en el set de prueba).
4. La diferencia de aciertos se contrasta con el **test exacto de McNemar** sobre los pares discordantes (binomtest bilateral).

## Tabla comparativa

| métrica | V1 (champion) | V2 (ab-candidate) | delta (V2 - V1) |
|---|---:|---:|---:|
| roc_auc | 0.8239 | 0.8248 | +0.0008 |
| average_precision | 0.1283 | 0.1287 | +0.0005 |
| precision | 0.0805 | 0.1144 | +0.0339 |
| recall | 0.6819 | 0.3773 | -0.3046 |
| f1_macro | 0.5129 | 0.5638 | +0.0509 |

## Lectura de la diferencia

El ROC AUC prácticamente no se mueve (+0.0008): las dos versiones ordenan casi igual las horas de riesgo, así que el modelo aprendió un patrón muy parecido. La V2 opera con umbral 0.70 frente a 0.50 de la V1: gana precisión (+0.0339) a costa de recall (-0.3046), y de ahí sale el F1 macro (+0.0509).

## Sensibilidad al umbral (modelo × umbral)

| configuración | umbral | roc_auc | precision | recall | f1_macro |
|---|---:|---:|---:|---:|---:|
| V1 @ su umbral | 0.50 | 0.8239 | 0.0805 | 0.6819 | 0.5129 |
| V2 @ su umbral | 0.70 | 0.8248 | 0.1144 | 0.3773 | 0.5638 |
| V1 @ umbral de la V2 | 0.70 | 0.8239 | 0.1148 | 0.3717 | 0.5640 |
| V2 @ umbral de la V1 | 0.50 | 0.8248 | 0.0803 | 0.6865 | 0.5123 |

A igual umbral (0.50) los modelos empatan en F1 macro (0.5129 vs 0.5123): la mejora de la V2 al operar con su propio umbral viene de la calibración del corte, no de un mejor ranking.

## Significancia (McNemar exacto)

- Pares donde solo acierta V1: **32,983**
- Pares donde solo acierta V2: **527,409**
- Pares discordantes: 560,392
- Test bilateral de McNemar: **p < 0.0001** (umbral de decisión: 0.05)

## Veredicto

La V2 gana en el despliegue (F1 macro +0.0509, McNemar p < 0.0001), PERO a igual umbral los modelos empatan (-0.0007): la mejora la explica el umbral (0.70 vs 0.50), no el reentrenamiento. Promover la V2 es inocuo y deja el corte afinado; el ROC AUC, que no depende del umbral, apenas se mueve (+0.0008).
