# Definición de datos y condiciones de uso: MiBici

## 1. Fuente

- **Portal:** Datos Abiertos de MiBici: https://mibici.net/es/datos-abiertos
- **Viajes:** un CSV por mes con el patrón
  `https://mibici.net/site/assets/files/<asset_id>/datos_abiertos_AAAA_MM.csv`.
  El `asset_id` cambia cada mes y no es secuencial, por eso el catálogo completo
  está fijo en `mibici/dataset.py` (`_ASSET_IDS`).
- **Nomenclatura de estaciones:**
  https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv
- La URL exacta y el hash SHA-256 de cada archivo descargado están en
  `data/manifest.json`.

## 2. Tamaño del dataset

El portal publica el histórico desde 2014-12. El equipo usa solo el periodo
**postpandemia** (la emergencia por COVID-19 terminó en mayo de 2023), para que
los patrones reflejen la operación actual del sistema.

| Concepto | Valor |
|---|---|
| Periodo | 2023-05 a 2026-09 |
| Archivos mensuales | 41 |
| Viajes | 15,376,127 |
| Estaciones (nomenclatura) | 484 |
| Parquet unificado | `data/raw/mibici_viajes_2023_05_a_2026_09.parquet` (~339 MB) |

Para reproducirlo: `python -m mibici.dataset`. El hash del Parquet en
`data/manifest.json` permite verificar que se obtuvieron exactamente los mismos datos.

## 3. Variables

### 3.1 Viajes (CSV mensual → esquema normalizado)

| Columna original | Columna en el proyecto | Tipo | Descripción |
|---|---|---|---|
| Viaje_Id | `viaje_id` | Int64 | Identificador del viaje |
| Usuario_Id | `usuario_id` | Int64 | Identificador del usuario (seudónimo) |
| Genero | `genero` | string | M / F / nulo |
| Año_de_nacimiento | `anio_nacimiento` | Int16 | Año de nacimiento declarado |
| Inicio_del_viaje | `inicio_viaje` | datetime | Fecha y hora de salida |
| Fin_del_viaje | `fin_viaje` | datetime | Fecha y hora de llegada |
| Origen_Id | `origen_id` | Int32 | Estación de salida |
| Destino_Id | `destino_id` | Int32 | Estación de llegada |

### 3.2 Nomenclatura de estaciones

| Columna original | Columna en el proyecto | Descripción |
|---|---|---|
| id | `estacion_id` | Identificador de estación (une con origen/destino) |
| name | `nombre` | Nombre de la estación |
| latitude / longitude | `latitud` / `longitud` | Ubicación |
| dpcapacity | `capacidad` | Número de anclajes |

### 3.3 Problemas de calidad resueltos en la ingesta

- El encoding varía entre UTF-8 y latin-1 según el mes.
- El encabezado `Año_de_nacimiento` llega dañado en algunos meses (p. ej. 2023-04),
  así que las columnas se identifican por un fragmento estable del nombre.
- Hay literales `NULL` sin comillas (p. ej. en `Genero`, 2024-2026); se convierten a nulo.
- Las fechas de 2023-11, 2024-09 y 2024-10 vienen como `dd/mm/yyyy`; el resto, en ISO 8601.

## 4. Construcción de las variables objetivo

Los viajes se agregan a una tabla **estación × hora** (`mibici/features.py`,
`aggregate_station_hour`): `salidas` = viajes con origen en la estación en esa
hora, y `llegadas` = viajes con destino en esa hora. Las horas sin viajes se
rellenan con 0.

| Pregunta | Tipo | Variable objetivo |
|---|---|---|
| P1 | Regresión | `salidas` de la estación en la hora t+1 |
| P2 | Clasificación | 1 si la estación queda en riesgo de vaciarse o saturarse, según el saldo `neto = salidas − llegadas` comparado con `capacidad` (diseño del proxy en `references/proxy_riesgo_p2.md`, issue #6) |
| P3 | Clustering | Sin objetivo: perfil de 24 dimensiones (proporción de salidas por hora del día) por estación |

Los modelos solo usan `inicio_viaje`, `fin_viaje`, `origen_id` y `destino_id`.
`usuario_id`, `genero` y `anio_nacimiento` **no se usan como features**.

## 5. Condiciones de uso y datos personales

### 5.1 Qué publica MiBici

El portal de datos abiertos enlaza dos documentos: **Términos y condiciones** y
**Aviso de privacidad**. Ninguno de los dos define una licencia explícita de
reutilización para los CSV de datos abiertos:

- Los **Términos y condiciones** regulan el uso del servicio de bicicletas
  (suscripciones, horarios, tiempos de uso, cobros); no hablan de los datos.
- El **Aviso de privacidad integral** es de la Agencia Metropolitana de Servicios
  de Infraestructura para la Movilidad del AMG (AMIM), responsable del tratamiento
  de los datos personales conforme a la *Ley General de Protección de Datos
  Personales en Posesión de Sujetos Obligados* y su equivalente del Estado de
  Jalisco. Cubre datos de personal, visitantes y trámites; no menciona de forma
  específica los registros de viajes publicados.

Por lo tanto, el equipo usa los datos solo con fines **académicos y no
comerciales**, cita la fuente (https://mibici.net/es/datos-abiertos) y aplica las
reglas de la sección 5.2.

### 5.2 Tratamiento de datos personales en el proyecto

El aviso de privacidad define como dato personal cualquier información de una
persona *identificada o identificable*, y una persona es identificable cuando su
identidad puede determinarse **directa o indirectamente**. Una secuencia de viajes
del mismo `usuario_id` (horas, estaciones de origen y destino) podría permitir
identificar a alguien de forma indirecta, así que el equipo aplica estas reglas:

- `usuario_id` es un seudónimo: se trata como anónimo, nunca se intenta
  reidentificar a nadie ni se cruza con otras fuentes.
- No se reconstruyen ni se publican trayectorias de usuarios individuales.
- `genero` y `anio_nacimiento` son cuasi-identificadores. No se usan en los
  modelos ni se publican cruzados con estación y hora.
- **Umbral de publicación:** no se publican agregados con menos de 5 viajes
  cuando involucren atributos de usuario (género o edad).
- Los datos no contienen datos personales sensibles (salud, origen étnico,
  biométricos, etc.) en el sentido del aviso de privacidad.
- Los datos crudos (`data/raw/`) no se versionan en Git (`.gitignore`); solo
  `data/manifest.json`.

### 5.3 Reglas del servicio que afectan los datos

Algunas reglas de los Términos y condiciones explican patrones que aparecen en
los datos y se deben tener en cuenta al modelar:

| Regla del servicio | Efecto en los datos |
|---|---|
| Solo se pueden liberar bicicletas de 5:00 a 00:59; anclar es posible las 24 h | Casi no hay `salidas` de 1:00 a 4:59, pero sí puede haber `llegadas` |
| Viajes sin cargo de menos de 30 min (45 min los domingos) | La mayoría de los viajes son cortos; duraciones mayores son excepciones |
| Uso continuo de más de 8 h suspende la cuenta | Viajes de más de 8 h son anómalos (bici no anclada o error de registro) |
| Si la estación está llena, el usuario pide 10 min extra para anclar en otra | Las estaciones saturadas desvían llegadas a estaciones vecinas (relevante para P2) |
| Una bicicleta a la vez por usuario | Un `usuario_id` no puede tener viajes traslapados |
| Usuarios de 16 años o más (de 16 a 17 con tutor) | Valores de `anio_nacimiento` que impliquen menos de 16 años son errores |
| Servicio en Guadalajara, Zapopan y Tlaquepaque | Todas las estaciones están en esos tres municipios |

## 6. Partición temporal sin fuga

Definida en `mibici/config.py` y aplicada en `features.assign_split`:

| Conjunto | Periodo | Meses |
|---|---|---|
| Train | 2023-05 .. 2024-12 | 20 |
| Validación | 2025-01 .. 2025-06 | 6 |
| Prueba | 2025-07 .. 2026-09 | 15 |

- La partición es por fecha, nunca aleatoria: el modelo no ve el futuro.
- Todas las features históricas (lags de 1h, 24h y 168h, media móvil de 7 días,
  promedio de las 4 semanas previas) se desplazan al menos 1 hora, para no usar
  la hora que se predice.
