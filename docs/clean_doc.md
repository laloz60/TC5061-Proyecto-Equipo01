# Reglas de Limpieza de Datos y Tratamiento de Datos Personales

Este documento describe cómo se preparan y protegen los datos abiertos de viajes de MiBici usados en el proyecto.

El proyecto descarga archivos CSV mensuales de viajes abarcando el periodo de mayo de 2023 - septiembre de 2026 (`references/definicion_datos.md`) y un archivo de nomenclatura de estaciones desde el portal de datos abiertos de MiBici.

----------------

## Limpieza de Datos
La limpieza se basa en `references/definicion_datos.md` y en `mibici/dataset.py` (`normalize_columns`, `read_csv_normalized`, `parse_trip_datetime`, `build_unified_parquet` y `build_nomenclatura`). La agregación y la incorporación de información de estaciones se describen en `mibici/features.py` (`aggregate_station_hour` y `add_station_features`).

### Normalización

- Se convierten todos los nombres de columnas a formato canónico, como:

| Columna | Nombre normalizado |
|---|---|
| Viaje_Id | `viaje_id` |
| Usuario_Id | `usuario_id` |
| Genero | `genero` |
| Año_de_nacimiento | `anio_nacimiento` |
| Inicio_del_viaje | `inicio_viaje` |
| Fin_del_viaje | `fin_viaje` |
| Origen_Id | `origen_id` |
| Destino_Id | `destino_id` |

- Codificación (Encoding): Se implementó una regla específica en la ingesta (`read_csv_normalized`), en caso de salir un `UnicodeDecodeError`, vuelve a intentar la lectura con Latin-1.

### Valores Nulos "NULL"
- Los `NULL` se reemplazan por un valor nulo cuando aparece (valor faltante). 

- Los valores vacíos de fecha también se convierten en nulos. `parse_trip_datetime` reconoce fechas en formato `DD/MM/YYYY HH:MM` y fechas `ISO 8601`. Si una fecha no vacía no se puede interpretar, la ingesta genera un error. Los valores numéricos que no pueden convertirse se transforman en nulos.

## Duplicados 

La ingesta actual no elimina filas duplicadas por `viaje_id`; conserva las filas leídas de los archivos fuente.

### Identificadores

Las columnas `origen_id` y `destino_id` se convierten a valores numéricos enteros anulables (`Int32`). Los valores que no pueden convertirse se transforman en valores faltantes.

Al incorporar la información de las estaciones con `left`, se conservan las filas del conjunto principal. Cuando no existe una coincidencia en la nomenclatura, los campos provenientes de esta quedan nulos.

En la agregación por estación y hora, los identificadores nulos no se incluyen en el `groupby`; un viaje con origen nulo podría no contar como salida y uno con destino nulo podría no contar como llegada.

-------
## Datos Personales y Privacidad

Se encuentra en `references/definicion_datos.md`, `mibici/features.py`, `.gitignore` y `mibici/modeling/train.py`

Dentro de los datasets microdatos abiertos de MiBici, los camposdelicados o identificados con riesgo de reidentificación son:
- `usuario_id`: Identificador único asignado al usuario del servicio.
- `genero`: Género reportado por el usuario.
- `anio_nacimiento`: Año de nacimiento (para saber la edad).

No se publican agregados de menos de cinco viajes cuando involucren género o edad. Esta regla se presenta como política documentada; no se afirma que el código de entrenamiento la aplique automáticamente.

### Agregación
Para generar las características se leen `inicio_viaje`, `fin_viaje`, `origen_id` y `destino_id`. Los viajes se agregan en conteos de salidas y llegadas por estación y hora. 

En esta etapa de preparación de características no se seleccionan `usuario_id`, `viaje_id`, `genero` ni `anio_nacimiento` como variables predictoras para cálculos descritos. Esto no significa que dichos campos se eliminen del archivo Parquet unificado.

### Git / MLflow
El proyecto utiliza MLflow para dar seguimiento a los experimentos de regresión (P1), clasificación (P2) y clustering (P3). Pudiendo obtener:  

- Parámetros e hiperparámetros de los modelos.    
- Etiquetas para identificar experimentos, familias de modelos y etapas de trabajo.    
- Métricas de evaluación, como MAE, RMSE, R², Precision, Recall, F1 y Silhouette Score, según el tipo de experimento.    
- Firmas de modelos y ejemplos de entrada.    
- Modelos y artefactos de evaluación, como matrices de confusión, curvas Precision-Recall, mapas y gráficas de resultados.    

El contenido registrado varía según el experimento y la rama. No todos los modelos registran necesariamente los mismos artefactos.

El archivo .env está incluido en .gitignore, por lo que se excluye del seguimiento de Git. Estas reglas ayudan a evitar que archivos locales y conjuntos de datos se incorporen accidentalmente al repositorio. 