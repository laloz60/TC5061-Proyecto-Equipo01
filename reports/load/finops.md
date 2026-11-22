# Análisis FinOps: costo por inferencia

## Supuestos
- Proveedor de referencia: AWS EC2, región `us-east-1`, Linux, precios consultados el 2026-11-19 (compute.doit.com, calculator.holori.com).
- Toda la pila (api + mlflow + nginx) corre en una sola instancia.
- Recursos medidos con `docker stats` cada 2 s (`scripts/load_test.sh`); en docker stats 100 % de CPU = 1 vCPU. Las pruebas se corrieron en una laptop de 8 núcleos lógicos con Docker limitado a 5.8 GiB; se asume que 1 núcleo de la laptop equivale a 1 vCPU de EC2.
- Datos: Locust de `reports/load/{A,B,C}_1r_stats.csv` y `{A,B,C}_2r_stats.csv`, de la **misma corrida** en la que se midió `docker_stats.csv`. Por eso no se usan `A/B/C_stats.csv` (corrida de #161, hecha en otro equipo sin `docker stats`): mezclar peticiones de una máquina con recursos de otra daría un costo sin sustento. Esa corrida tuvo menor latencia con 1 réplica (p95 de 81–310 ms contra 1300–4700 ms aquí), lo que confirma que los valores absolutos dependen del hardware. La comparación de 1 contra 2 réplicas solo vale dentro de un mismo equipo.
- Pico del host = máximo de la suma de todos los contenedores en una misma muestra (no la suma de los picos de cada servicio, que no ocurren al mismo tiempo).
- Instancia mínima: memoria pico × 1.2 + 0.5 GiB (SO y Docker) y vCPU pico × 1.2 deben caber en la instancia.
- Locust usa usuarios en lazo cerrado con espera de 1–3 s entre peticiones: el RPS máximo teórico es ~usuarios / 2 s (25 RPS en A y B, 50 RPS en C).
- Spot: precio aproximado; varía por zona y por día.

| instancia | vCPU | RAM | on-demand (USD/h) | spot (USD/h) |
|---|---|---|---|---|
| t3.small | 2 | 2 GiB | 0.0208 | ~0.0078 |
| t3.medium | 2 | 4 GiB | 0.0416 | ~0.0157 |
| t3.large | 2 | 8 GiB | 0.0832 | ~0.0314 |
| t3.xlarge | 4 | 16 GiB | 0.1664 | ~0.0628 |

## Fórmulas
- **Tiempo:** costo de la prueba = precio_hora × duración_h
- **Costo por inferencia:** (precio_hora × duración_h) / peticiones_exitosas
- Fórmula base, para comparar: precio_hora / (RPS × 3600)
- **Recursos:** vCPU necesarios = CPU% / 100; uso = observado / provisionado

## Resultados por escenario

Escenarios (#161): **A** rampa 50 usuarios en 120 s, **B** 50 usuarios sostenidos 180 s, **C** pico de 100 usuarios en ~2 s durante 60 s. Generado con `python scripts/costo_inferencia.py` (`reports/load/costos_inferencia.csv`):

| escenario | réplicas | peticiones OK | fallas | RPS | p95 (ms) | pico host (vCPU / GiB) | instancia | costo prueba (USD) | USD / 1M inf. | USD / 1M inf. spot |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 1784 | 0 % | 15.00 | 2300 | 1.56 / 3.00 | t3.large | 0.0028 | 1.55 | 0.59 |
| A | 2 | 2161 | 0 % | 18.17 | 260 | 1.54 / 3.81 | t3.large | 0.0028 | 1.28 | 0.48 |
| B | 1 | 3709 | 0 % | 20.68 | 1300 | 1.48 / 2.90 | t3.medium | 0.0021 | **0.56** | **0.21** |
| B | 2 | 4157 | 0 % | 23.30 | 400 | 2.35 / 3.85 | t3.xlarge | 0.0083 | 2.00 | 0.76 |
| C | 1 | 1302 | 0 % | 22.35 | 4700 | 1.61 / 2.92 | t3.medium | 0.0007 | 0.53 | 0.20 |
| C | 2 | 1646 | 0 % | 28.10 | 4300 | 3.25 / 4.13 | t3.xlarge | 0.0028 | 1.68 | 0.64 |

- Las 6 corridas costarían en total **0.0195 USD** on-demand.
- La fórmula base (precio_hora / RPS) da valores a menos de 3 % de la fórmula por peticiones exitosas, porque no hubo fallas.
- La primera corrida de C con 2 réplicas tuvo 21 % de errores 502 por la configuración de nginx (corregida en #162); la tabla usa la corrida posterior al arreglo.
- En C con 1 réplica el sistema se satura: atiende 22.35 de ~50 RPS posibles y el p95 sube a 4.7 s.

## Escalabilidad: 1 contra 2 réplicas

| escenario | p95 1 → 2 réplicas | mediana 1 → 2 | RPS 1 → 2 | USD / 1M inf. 1 → 2 |
|---|---|---|---|---|
| A | 2300 → 260 ms (**−89 %**) | 250 → 77 ms | 15.0 → 18.2 (+21 %) | 1.55 → 1.28 (−17 %) |
| B | 1300 → 400 ms (**−69 %**) | 280 → 90 ms | 20.7 → 23.3 (+13 %) | 0.56 → 2.00 (+257 %) |
| C | 4700 → 4300 ms (−9 %) | 2600 → 1200 ms | 22.4 → 28.1 (+26 %) | 0.53 → 1.68 (+216 %) |

- Con carga moderada (A y B), la segunda réplica reduce el p95 entre 69 % y 89 %.
- En el pico (C) la mediana baja a menos de la mitad, pero el p95 casi no cambia: el pico de 100 usuarios en 2 s supera lo que absorben las dos réplicas.
- El RPS sube solo 13–26 % porque Locust está en lazo cerrado: está acotado por el número de usuarios, no por la capacidad de la api.
- Cada réplica de la api suma ~1.1 GiB y ~1.1–1.6 vCPU en pico. En B y C eso obliga a pasar de t3.medium a t3.xlarge (4× el precio por hora), y el costo por inferencia sube 3.2–3.6×.
- En A el costo por inferencia **baja** con 2 réplicas (−17 %): las dos configuraciones caben en t3.large y la segunda atiende más peticiones con la misma instancia.

## Right-sizing y spot

- **Instancia mínima por escenario:** t3.medium alcanza para 1 réplica en B y C (72–73 % de memoria y 74–81 % de CPU en pico). En A con 1 réplica el pico de memoria es 3.0 GiB; con el margen de 20 % queda en 4.1 GiB y pasa a t3.large por solo 0.1 GiB.
- **t3.small no alcanza** en ningún escenario: mlflow solo ocupa ~1.8 GiB. La propuesta de sacar mlflow del host de inferencia está en `reports/load/optimizaciones.md` (#164).
- **Spot** reduce el costo por inferencia ~62 % en todos los escenarios (por ejemplo, B con 1 réplica pasa de 0.56 a 0.21 USD por millón).
- **Créditos de CPU de t3:** las t3 tienen un CPU base garantizado (t3.medium 20 % por vCPU = 0.4 vCPU; t3.large 0.6; t3.xlarge 1.6) y lo que se use por encima en modo *unlimited* se cobra a 0.05 USD por vCPU-hora. El promedio de CPU del host fue 0.96–1.14 vCPU con 1 réplica. Si la carga de B (1.05 vCPU promedio) fuera continua 24/7 en t3.medium, se pagarían (1.05 − 0.4) × 0.05 = 0.033 USD/h extra: **+78 %** sobre el precio on-demand (0.074 USD/h en total). Para carga sostenida conviene comparar con una familia sin créditos (m5/m6i).

## Recomendaciones

Las optimizaciones con su dato de respaldo, ahorro, esfuerzo y riesgo están en `reports/load/optimizaciones.md` (#164).
