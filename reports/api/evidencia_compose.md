# Evidencia Actividad 10 — Sistema orquestado con Docker Compose (Issues #156 y #157)

**Equipo 01 — TC5061 MLOps**
**Fecha:** 2026-11-18
**Responsable:** laloz60 (Gerardo Carreño)
**Criterio de rúbrica:** Orquestación con Docker Compose — 25 pts

## Contexto

Hasta la Actividad 8 el sistema se levantaba en dos pasos manuales: el servidor
MLflow en el host y el contenedor de la API apuntando a
`host.docker.internal:5000`. Aquí se orquestan los dos servicios dentro de una
misma red y se reduce el arranque a un solo comando.

## Archivos

| Archivo | Cambio |
|---|---|
| `docker-compose.yml` | **Nuevo.** Servicios `mlflow` y `api`, red `mibici-net`, healthchecks, `mem_limit`, `env_file` opcional |
| `docker/mlflow.Dockerfile` | **Nuevo.** Imagen del servidor de tracking (`mlflow==3.17.2`, `--workers 2`) |
| `Dockerfile` | `MLFLOW_TRACKING_URI` pasa de `host.docker.internal:5000` a `mlflow:5000` (issue #157) |
| `.dockerignore` | Excluye del contexto `tests/`, `scripts/`, `Makefile`, `MLproject`, `python_env.yaml`, `docker-compose.yml` |
| `.env.example` | Documenta `MLFLOW_PORT` y que `.env` es opcional para Compose |
| `Makefile` | Nuevos targets: `compose-up`, `compose-ps`, `compose-test`, `compose-down`, `compose-logs` |

## Un solo comando

```bash
$ make compose-up
docker compose up -d --build --wait

 Container mibici-mlflow-1 Starting
 Container mibici-mlflow-1 Healthy
 Container mibici-api-1 Starting
 Container mibici-api-1 Waiting
 Container mibici-api-1 Healthy
make compose-up  12.2s total
```

`--wait` bloquea hasta que los dos servicios reporten `healthy`, de modo que el
comando no regresa antes de tiempo.

## Estado de los servicios

```
$ docker compose ps
NAME              IMAGE              SERVICE   STATUS                    PORTS
mibici-api-1      mibici-api:latest  api       Up 8 seconds (healthy)    0.0.0.0:8000->8000/tcp
mibici-mlflow-1   mibici-mlflow      mlflow    Up 14 seconds (healthy)   0.0.0.0:5001->5000/tcp
```

## Configuraciones avanzadas (nivel "Excelente" de la rúbrica)

| Elemento | Implementación | Por qué |
|---|---|---|
| **Red dedicada** | `networks: mibici-net` (bridge) | La API resuelve `http://mlflow:5000` por nombre y el tráfico de modelos no se mezcla con la red por defecto |
| **Volúmenes** | `./mlflow.db` y `./mlruns` como bind mounts | El Model Registry y los 1.3 GB de artefactos viven en el host y sobreviven a `docker compose down` |
| **Variables de entorno** | `env_file` (opcional) + `environment` | `environment` tiene prioridad: dentro de la red la URI es `http://mlflow:5000`, no la del host |
| **Dependencia ordenada** | `depends_on: condition: service_healthy` | La API no consulta el registry hasta que MLflow responde 200 en `/health` |
| **Healthchecks** | En ambos servicios | `--wait` funciona y el estado es observable con `docker compose ps` |
| **Límites de recursos** | `mem_limit` 3g / 4g | Evita OOM y da el denominador para `docker stats` en el análisis FinOps |
| **Reinicio** | `restart: unless-stopped` | El sistema se recupera solo si un servicio muere |

## Smoke test end-to-end

```bash
$ bash scripts/test_api.sh http://localhost:8000
== 1. Health check (GET /health) ==
{"status":"healthy","modelos_cargados":["p1","p2","p3"]}   HTTP 200
== 2. Prediccion P1 ==  {"demanda_predicha_viajes":5.620645371371248}   HTTP 200
== 3. Prediccion P2 ==  {"riesgo_desabasto_o_saturacion":1,...}          HTTP 200
== 4. Prediccion P3 ==  {"cluster_asignado":2}                           HTTP 200
== 5. Entrada invalida ==                                               HTTP 422
```

Los tres modelos se cargan por la red interna, no por el host:

```
INFO  | Conectando al Model Registry en: http://mlflow:5000
SUCCESS | p1: modelo cargado via proxy mlflow-artifacts:/1/models/m-1627f3d59ced4700844564b88998226e/artifacts
SUCCESS | p2: modelo cargado via proxy mlflow-artifacts:/1/models/m-64e60b83f5e74fc39d61dbedf9753e55/artifacts
SUCCESS | p3: modelo cargado via proxy mlflow-artifacts:/1/models/m-f301c3bd3ab546c098feb182f56bb15b/artifacts
SUCCESS | Modelos cargados: ['p1', 'p2', 'p3']
```

```bash
$ docker network inspect mibici_mibici-net --format '{{range .Containers}}{{.Name}} -> {{.IPv4Address}}{{"\n"}}{{end}}'
mibici-mlflow-1 -> 172.18.0.2/16
mibici-api-1 -> 172.18.0.3/16
```

## Consumo de recursos (insumo para el FinOps, issue #162)

```bash
$ docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
NAME                 CPU %   MEM USAGE / LIMIT   MEM %
mibici-api-1         0.23%   896.1MiB / 4GiB     21.88%
mibici-mlflow-1      6.84%   1.775GiB / 3GiB     59.17%
```

## Hallazgos durante la validación

1. **Puerto 5000 ocupado en macOS.** El Receptor AirPlay (`ControlCenter`)
   escucha en `*:5000` y Docker no puede hacer bind (*address already in use*).
   Es el error documentado en el README. Resolución: el puerto publicado en el
   host quedó parametrizado como `${MLFLOW_PORT:-5001}`; dentro de la red los
   servicios siguen hablando en el 5000. Para volver a 5000 hay que desactivar
   el Receptor AirPlay y poner `MLFLOW_PORT=5000` en `.env`.

2. **Un rango de puertos rompe la reproducibilidad.** Se probó
   `"8000-8009:8000"` para habilitar `--scale api=2`, pero Docker elige el
   puerto al azar dentro del rango: en una corrida publicó la API en **8001**
   con el 8000 libre y `scripts/test_api.sh` falló con `HTTP 000`. Se volvió al
   mapeo fijo `"8000:8000"`.

3. **MLflow con 4 workers se acercaba al límite.** Medido con
   `/sys/fs/cgroup/memory.stat`: 1.89 GB de memoria anónima contra el `2g`
   inicial (94%). Se pasó a `--workers 2` y `mem_limit: 3g` (59% de uso).

## Verificación de reproducibilidad

Secuencia completa ejecutada, no solo el primer arranque:

```bash
docker compose down          # quita contenedores y red, conserva mlflow.db y mlruns/
make compose-up              # vuelve a healthy en 12 s con las 3 predicciones OK
```

## Cómo se cumple la rúbrica

| Nivel | Requisito | Evidencia |
|---|---|---|
| **Excelente** | Todos los servicios, un solo comando, redes + volúmenes + variables de entorno | `docker compose up -d --wait` en 12 s; sección "Configuraciones avanzadas" |
| Satisfactorio | Servicios necesarios, un solo comando | Ídem |
| Insuficiente | Parcial o con ajustes manuales | Sin ajustes manuales: `make compose-up` y `make compose-test` |
