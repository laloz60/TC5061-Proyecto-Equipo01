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
| `POST /predict/p3-clustering` | `{"cluster_asignado":2}` | 200 |
| Payload inválido | Validación de Pydantic | 422 |

También responden la documentación interactiva (`http://localhost:8000/docs`, 200) y la UI de MLflow (`http://localhost:5001`, 200).

## 4. Peticiones reales con escenarios contrastantes

Para mostrar que responde el modelo real (el registry simulado del CI devuelve valores fijos):

| Escenario | P1 demanda predicha | P2 riesgo (probabilidad) |
|---|---|---|
| Lunes 8:00, estación con alta demanda reciente (12 salidas en la última hora, saldo neto −6, capacidad 20) | **12.90 viajes** | **1 (0.9995)** |
| Domingo 3:00, estación sin actividad | **0.00 viajes** | **0 (0.0281)** |

P3 con perfiles horarios normalizados (proporción de viajes por hora, suma 1):

| Perfil | Clúster |
|---|---|
| Oficina (picos 8 h y 18 h) | 1 |
| Residencial (pico 7 h) | 1 |
| Ocio (tarde-noche) | 2 |
| Plano | 1 |

## 5. Consumo de recursos en reposo (`docker stats`)

```
NAME              CPU %     MEM USAGE / LIMIT   MEM %
mibici-api-1      1.36%     1.116GiB / 4GiB     27.89%
mibici-mlflow-1   2.34%     1.985GiB / 3GiB     66.18%
```

Consistente con el análisis de FinOps (MLflow ~1.9 GiB, API ~1.1 GiB en reposo).

## 6. Hallazgo: formato de entrada de P3

El modelo de P3 se entrenó con **perfiles horarios normalizados** (cada fila suma 1; ver `_perfil_ventana` en `mibici/modeling/train.py`), pero la API recibe `perfil_horario` sin normalizar ni documentar ese formato. `scripts/test_api.sh` envía conteos (`[1, 2, …, 24]`), que quedan fuera de la distribución de entrenamiento: la petición responde 200, pero el clúster asignado no es interpretable. Con proporciones, el modelo sí distingue perfiles (sección 4).

Opciones: normalizar el perfil en `mibici/api/main.py` antes de predecir, o documentar en `mibici/api/schemas.py` que la entrada son proporciones y ajustar el ejemplo de `test_api.sh`.

## 7. Apagado

```bash
docker compose down     # mlflow.db y mlruns/ quedan en el host
```
