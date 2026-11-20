# Interpretación de pruebas de carga 

## Resumen de resultados oficial — `reports/load/`

| Escenario | Configuración | Peticiones OK | Tasa de error | RPS prom. | p50 / p95 / p99 (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A (Rampa suave)** | 120 s (rampa y sostenido) | 2,312 | 0.00% | 19.66 | 55 / 81 / 99 |
| **B (Carga sostenida)** | 180 s (50 usuarios fijos) | 4,376 | 0.00% | 24.57 | 56 / 97 / 930 |
| **C (Pico rápido)** | 60 s (subida a 100 usuarios) | 2,855 | 0.00% | 49.30 | 61 / 310 / 640 |


- Sin errores (0.00% fallos). En ninguno de los escenarios se presentan errores 5xx ni fallos de conexión; el servicio mantiene disponibilidad bajo rampa, carga sostenida y pico.
- El cuello de botella es de latencia en colas altas, no de fallos. Aunque la mediana (`p50`) permanece ágil y estable entre 55 ms y 61 ms, las colas altas se deterioran con mayor carga o ráfagas: `p99` sube de 99 ms (Escenario A) a 930 ms (Escenario B), mientras que `p95` pasa de 81 ms (A) a 310 ms (C), donde también se alcanza el mayor ritmo de peticiones (~49.30 RPS).
- Algunas causas probables, pueden ser la contención por ejecución síncrona / concurrencia limitada en la API y competencia por CPU/memoria en el host local.

## Áreas de mejora

- Evaluar el número de workers/concurrencia y el desacoplamiento de dependencias pesadas en tiempo de ejecución para amortiguar picos (`p95`/`p99`) en escenarios tipo C y cargas prolongadas tipo B.
- Evitar que la huella de memoria o cómputo de MLflow compita con la inferencia durante las pruebas o en producción (alineado con el dimensionamiento de la pila).

Nota: Se corrieron pruebas complementarias de 2 réplicas en reports/load/rep2/ (la corrida oficial del issue es la de reports/load/).