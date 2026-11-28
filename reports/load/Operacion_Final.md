# Evidencia de operación final — CI/CD, GHCR y Locust

**Fecha de ejecución:** 2026-11-24  
**Rama / commit verificado en `main`:** `5dac2c2`  
**Stack:** `docker compose up -d --build --wait` con Model Registry real (`MIBICI_MOCK_REGISTRY=0`, modelos `@champion` cargados: `["p1", "p2", "p3"]`).

---

## 1. Pipeline CI/CD en verde en `main` y verificación de GHCR

* **Workflow (`CI/CD Pipeline`):** 6 jobs en verde sobre `main` (`quality`, `tests` en matriz `ubuntu-latest` y `macos-latest`, `docker-build`, `api-smoke` y `notify`).
* **Evidencia visual:** `reports/figures/CI_Main_FInal.png`
* **Imagen publicada en GHCR:** `ghcr.io/laloz60/mibici-api:latest`
  * **Index digest (OCI):** `sha256:99ec7dbf0a1f30bc72bb5e62e345ab32c781ff4d6cf6b7ca08283b0a023e5630`
  * **Manifest (`linux/amd64`):** `sha256:b58a53d0ed9037fc16867ca1757f77386fa437ae8d9400b4399495581c826c8f`
  * **Correspondencia con `main`:** coincide con la publicación del commit `5dac2c2` (run `45760445855`, `2026-11-25`), validado vía `docker buildx imagetools inspect` y el log del job `docker-build` en GitHub Actions.

---

## 2. Verificación funcional previa (`scripts/test_api.sh`)

* `GET /health`: `HTTP 200` — `{"status":"healthy","modelos_cargados":["p1","p2","p3"]}`
* `POST /predict/p1-demanda`: `HTTP 200` — `{"demanda_predicha_viajes":5.5739}`
* `POST /predict/p2-riesgo`: `HTTP 200` — `{"riesgo_desabasto_o_saturacion":1,"probabilidad_riesgo":0.7442}`
* `POST /predict/p3-clustering`: `HTTP 200` — `{"cluster_asignado":4}`
* Payload inválido (validación Pydantic): `HTTP 422`

---

## 3. Corrida de carga con Locust (Escenario A — sistema final)

Ejecutada contra `http://localhost:8000` en modo *headless* (`-u 50 -r 1 -t 2m`, rampa de 50 s + 70 s sostenidos) sobre el sistema levantado con Docker Compose y el `locustfile.py` actualizado (perfil horario normalizado de P3).  
**Artefactos generados:** `reports/load/final_A_stats.csv`, `reports/load/final_A_stats_history.csv` y `reports/load/final_A.html`.

### Desglose por endpoint

| Método | Endpoint | Peticiones | Fallas (%) | RPS | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Máx (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `POST` | `/predict/p1-demanda` | 987 | 0 (0.00%) | 8.29 | 56 | 52 | 100 | 150 | 190 |
| `POST` | `/predict/p2-riesgo` | 913 | 0 (0.00%) | 7.66 | 20 | 17 | 42 | 58 | 92 |
| `POST` | `/predict/p3-clustering` | 468 | 0 (0.00%) | 3.93 | 8 | 8 | 22 | 34 | 45 |
| `GET` | `/health` | 50 | 0 (0.00%) | 0.42 | 7 | 8 | 15 | 16 | 16 |
| — | **Aggregated** | **2,418** | **0 (0.00%)** | **20.30** | **32** | **29** | **85** | **120** | **190** |

---

## 4. Comparativa contra el Entregable 7 (Escenario A, 120 s)

| Corrida (Escenario A, 50 usuarios, 120 s) | Peticiones OK | RPS | p50 (ms) | p95 (ms) | p99 (ms) | Tasa de error |
|---|---:|---:|---:|---:|---:|---:|
| **Entregable 7 — Corrida base (1 inst.)** | 2,312 | 19.66 | 55 | 81 | 99 | 0.00% |
| **Entregable 7 — FinOps (1 réplica)** | 1,784 | 15.00 | 250 | 2,300 | — | 0.00% |
| **Entregable 7 — FinOps (2 réplicas + Nginx)** | 2,161 | 18.17 | 77 | 260 | — | 0.00% |
| **Actividad 11 — Sistema final (`final_A`, 1 réplica)** | **2,418** | **20.30** | **29** | **85** | **120** | **0.00%** |

### Lectura comparativa

* **Estabilidad:** La tasa de error se mantuvo en `0.00%` en los cuatro endpoints, confirmando que la normalización de `ClusterInput` para P3 y la carga real de los tres modelos `@champion` vía `mlflow-artifacts:/` operan sin degradaciones bajo carga concurrente.
* **Rendimiento y latencia:** El `p95` agregado (`85 ms`) es consistente con la corrida base del Entregable 7 (`81 ms`), con una mediana menor (`29 ms` vs. `55 ms`). El ligero incremento en peticiones totales (`2,418` vs. `2,312`) y RPS promedio (`20.30` vs. `19.66`) se debe tanto al hardware del equipo local como a la tasa de rampa (`-r 1` frente a `-r 0.8` en `scripts/load_test.sh`), estabilizándose cerca del techo en lazo cerrado de ~25 RPS con 50 usuarios activos.
* **Perfil por modelo:** `POST /predict/p1-demanda` (`RandomForestRegressor`) sigue concentrando el mayor tiempo de cómputo (`p50 = 52 ms`, `p95 = 100 ms`), mientras que `P2` (`p95 = 42 ms`) y `P3` (`p95 = 22 ms`) responden por debajo de los `50 ms`.
