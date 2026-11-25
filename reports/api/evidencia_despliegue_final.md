# Evidencia: despliegue con el Model Registry real (issue #194)

Verificación del sistema completo levantado con Docker Compose y los modelos `@champion` del Model Registry del equipo (no el registry simulado del CI, `MIBICI_MOCK_REGISTRY=0`). Sirve también como ensayo de la demo del video.

| Dato | Valor |
|---|---|
| Fecha | 2026-11-23 |
| Commit | `27be8c2` (`main`) |
| Equipo | macOS, Docker 29.8.0 (12 CPU, 7.7 GiB asignados a Docker) |
| Registry | `mlflow.db` y `mlruns/` del equipo, montados en el servicio `mlflow` |

## 1. Arranque con un solo comando

```bash
docker compose up -d --build --wait
```

Terminó en **13 s** con las imágenes ya construidas; los dos servicios quedaron `healthy`:

```
NAME              IMAGE                  STATUS                    PORTS
mibici-api-1      mibici-api:latest      Up 5 seconds (healthy)    0.0.0.0:8000->8000/tcp
mibici-mlflow-1   mibici-mlflow:latest   Up 11 seconds (healthy)   0.0.0.0:5001->5000/tcp
```

## 2. Carga de los modelos `@champion`

Log de la API (`docker compose logs api`): se conecta al servicio `mlflow` por la red interna y descarga los artefactos por el proxy `mlflow-artifacts:/`, el flujo documentado para el contenedor:

```
Conectando al Model Registry en: http://mlflow:5000
p1: modelo cargado via proxy mlflow-artifacts:/1/models/m-1627f3d59ced4700844564b88998226e/artifacts
p2: modelo cargado via proxy mlflow-artifacts:/1/models/m-64e60b83f5e74fc39d61dbedf9753e55/artifacts
p3: modelo cargado via proxy mlflow-artifacts:/1/models/m-f301c3bd3ab546c098feb182f56bb15b/artifacts
Modelos cargados: ['p1', 'p2', 'p3']
```

Los modelos cargados coinciden con las versiones `@champion` del registry:

| Modelo registrado | Versión `@champion` | Logged model | Run de origen |
|---|---|---|---|
| `mibici-equipo01-p1-demanda` | 2 | `m-1627f3d5…` | `c8ccdd71…` |
| `mibici-equipo01-p2-riesgo` | 2 | `m-64e60b83…` | `1cd16f9f…` |
| `mibici-equipo01-p3-grupos` | 2 | `m-f301c3bd…` | `398be41f…` |

## 3. Prueba de humo (`scripts/test_api.sh`)

```bash
bash scripts/test_api.sh http://localhost:8000
```

| Prueba | Respuesta | HTTP |
|---|---|---|
| `GET /health` | `{"status":"healthy","modelos_cargados":["p1","p2","p3"]}` | 200 |
| `POST /predict/p1-demanda` | `{"demanda_predicha_viajes":5.6206…}` | 200 |
| `POST /predict/p2-riesgo` | `{"riesgo_desabasto_o_saturacion":1,"probabilidad_riesgo":0.7449…}` | 200 |
| `POST /predict/p3-clustering` (perfil tipo oficina) | `{"cluster_asignado":1}` | 200 |
| Payload inválido | Validación de Pydantic | 422 |

También responden la documentación interactiva (`http://localhost:8000/docs`, 200) y la UI de MLflow (`http://localhost:5001`, 200).

## 4. Peticiones reales con escenarios contrastantes

Para mostrar que responde el modelo real (el registry simulado del CI devuelve valores fijos):

| Escenario | P1 demanda predicha | P2 riesgo (probabilidad) |
|---|---|---|
| Lunes 8:00, estación con alta demanda reciente (12 salidas en la última hora, saldo neto −6, capacidad 20) | **12.90 viajes** | **1 (0.9995)** |
| Domingo 3:00, estación sin actividad | **0.00 viajes** | **0 (0.0281)** |

P3 con el mismo perfil enviado como conteos de viajes y como proporciones (después de la corrección de la sección 6):

| Perfil | Clúster con conteos | Clúster con proporciones |
|---|---|---|
| Oficina (picos 8 h y 18 h) | 1 | 1 |
| Residencial (pico 7 h) | 1 | 1 |
| Ocio (tarde-noche) | 2 | 2 |
| Plano | 1 | 1 |
| Todo en ceros | 422 | — |
| Con un valor negativo | 422 | — |

## 5. Consumo de recursos en reposo (`docker stats`)

```
NAME              CPU %     MEM USAGE / LIMIT   MEM %
mibici-api-1      1.36%     1.116GiB / 4GiB     27.89%
mibici-mlflow-1   2.34%     1.985GiB / 3GiB     66.18%
```

Consistente con el análisis de FinOps (MLflow ~1.9 GiB, API ~1.1 GiB en reposo).

## 6. Hallazgo y corrección: formato de entrada de P3

**Hallazgo.** El modelo de P3 se entrenó con **perfiles horarios normalizados** (cada fila suma 1; ver `_perfil_ventana` en `mibici/modeling/train.py`), pero la API pasaba `perfil_horario` al modelo tal cual. `scripts/test_api.sh` y `locustfile.py` enviaban conteos (`[1, 2, …, 24]`), fuera de la distribución de entrenamiento: la petición respondía 200, pero el clúster no era interpretable.

**Corrección** (evitar el sesgo entre entrenamiento y servicio, *training-serving skew*):
- `mibici/api/schemas.py`: `ClusterInput` normaliza el perfil a proporciones con un validador de Pydantic. Acepta conteos o proporciones (un perfil ya normalizado no cambia) y rechaza con 422 los valores negativos o un perfil sin viajes. La descripción y el ejemplo aparecen en Swagger.
- `tests/test_api_schemas.py`: 6 pruebas (normalización, conteos = proporciones, perfil ya normalizado sin cambios y 3 entradas inválidas).
- `scripts/test_api.sh` y `locustfile.py`: el ejemplo de P3 es ahora un perfil realista de una estación tipo oficina.

Con la corrección, los conteos y las proporciones del mismo perfil producen el mismo clúster (sección 4).

## 7. Apagado

```bash
docker compose down     # mlflow.db y mlruns/ quedan en el host
```
