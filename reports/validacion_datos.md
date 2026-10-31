# Validación de datos con DeepChecks

Revisión de los viajes de MiBici y de las features antes de entrenar. El resultado completo está en [`data_validation/resumen.json`](data_validation/resumen.json).

## Cómo reproducirlo

    make validation_env   # solo la primera vez
    make validate

DeepChecks 0.19.1 no funciona con scikit-learn 1.9 ni con NumPy 2, que son las versiones con las que se entrenan los modelos. Por eso corre en un entorno aparte, definido en `requirements-validation.txt`.

**Alcance:** se analizaron 15,376,153 viajes de la descarga más reciente. El manifiesto del equipo registra 15,376,127, porque MiBici actualizó el CSV de 2023-05 después de generarlo. Faltantes y duplicados se cuentan sobre todas las filas. Los chequeos de DeepChecks y el drift usan muestras de 200,000 filas con semilla 42.

## Resultados

| Chequeo | Resultado | Problema | Acción |
|---|---|---|---|
| Tipos de datos | Las 8 columnas coinciden con el esquema; sin tipos mezclados en la muestra | No | La validación falla si el esquema cambia |
| Valores faltantes | `genero` 104,550 (0.68%); `anio_nacimiento` 7,592 (0.05%); las demás, 0 | Menor | Se conservan sin imputar: los modelos no usan esas columnas |
| Duplicados | 758 filas exactas y 307 viajes iguales con otro `viaje_id` | Sí | Se eliminan las 758 en las features; las 307 se conservan |
| Drift train vs test | Ninguna columna alcanza el umbral de 0.2 | No | Sin cambios; `mes` queda en observación |

## Duplicados

Las 758 filas exactas están todas en 2026-09 y son el 0.15% de ese mes. Repiten también su `viaje_id`, así que se tratan como un error de registro.

Los otros 307 viajes coinciden en usuario, horarios y estaciones, pero tienen otro `viaje_id`:

| Mes | Viajes | Ventana |
|---|---:|---|
| 2023-11 | 116 | Train |
| 2024-09 | 70 | Train |
| 2024-10 | 64 | Train |
| 2026-05 | 6 | Test |
| 2026-06 | 21 | Test |
| 2026-07 | 14 | Test |
| 2026-08 | 8 | Test |
| 2026-09 | 8 | Test |

**Impacto:** antes de la corrección, cada duplicado sumaba un viaje de más en `salidas` y `llegadas`. Las 758 filas exactas caen en test, así que no afectaron el entrenamiento, pero sí las métricas de esa ventana.

**Acción tomada:**

- `mibici/features.py` elimina los `viaje_id` repetidos antes de construir las features. La ejecución confirmó 758 filas eliminadas, y una prueba cubre este comportamiento.
- El Parquet y el manifiesto no se modifican, para conservar los datos tal como los publica la fuente.
- Los 307 viajes se conservan porque no se puede confirmar que sean errores. Son el 0.002% del total.

**Limitación:** `mibici/proxy_riesgo.py` y P3 leen el Parquet directamente, así que todavía incluyen las 758 filas. No se modificaron porque el efecto esperado es mínimo: 0.005% de los viajes, todos en un solo mes.

DeepChecks reportó 0% de duplicados en la muestra, sin incluir los identificadores. Una muestra puede dejar fuera la mayoría de un problema tan poco frecuente, y por eso los duplicados también se cuentan sobre todas las filas.

## Drift entre train y test

Se compararon train (2023-05 a 2024-12) y test (2025-07 a 2026-09). `viaje_id` y `usuario_id` quedan fuera porque compararlos entre periodos no dice nada sobre el comportamiento de los datos.

| Datos | Columna con mayor puntaje | Puntaje | Método |
|---|---|---:|---|
| Viajes | `anio_nacimiento` | 0.0785 | Kolmogorov-Smirnov |
| Features | `mes` | 0.1810 | Cramér's V |
| Features (sin contar `mes`) | `neto_media_7d` | 0.0865 | Kolmogorov-Smirnov |

**Lectura:** no hay señal de que los datos de test se comporten muy distinto a los de train. El puntaje de `mes` es el más cercano al umbral y es esperable, porque las dos ventanas cubren distintas proporciones de cada mes. Puede afectar a los modelos que usan `mes` como variable, así que conviene revisarlo si el error de test crece en meses poco representados en train.

## Datos personales

El Parquet contiene `usuario_id`, `genero` y `anio_nacimiento`. El resumen versionado solo guarda conteos y puntajes agregados, y una prueba verifica que no incluya identificadores de usuario. Los reportes HTML opcionales no se versionan.

## Seguridad del entorno de validación

`pip-audit` sobre `requirements-validation.txt` reporta `setuptools` 81.0.0 (PYSEC-2026-3447 y GHSA-xmj3-9gf7-h52w). Se acepta el riesgo porque `deepchecks` 0.19.1 necesita `pkg_resources`, que `setuptools` ya no incluye desde la versión 82, y las correcciones están en las versiones 83.0.0 y 84.1.0.

Este entorno solo se usa para validar datos localmente: no entrena los modelos ni se despliega. Falta revisar el detalle de ambas vulnerabilidades. Si sale una versión de `deepchecks` que no dependa de `pkg_resources`, conviene actualizar.
