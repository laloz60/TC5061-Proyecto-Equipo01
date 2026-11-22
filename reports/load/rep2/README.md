# Pruebas en réplica 2 (`rep2`)

- La corrida oficial del issue es la de `reports/load/`

- Alcance de `rep2/`: Pruebas complementarias (escenarios A, B y C) ejecutadas directamente contra una segunda instancia (`mibici-api-2` en `http://localhost:8001`) en el mismo host local para observar el comportamiento.

- Hallazgo clave: En `rep2` también se obtuvo 0.00% de error, con RPS de 19.70 (A), 22.79 (B) y 42.12 (C), y `p50/p95/p99` de 57/100/170 ms (A), 59/130/3,300 ms (B) y 190/1,300/2,400 ms (C), confirmando contención local al saturar el equipo.