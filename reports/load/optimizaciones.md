# Optimizaciones FinOps

Supuestos de precios, fórmulas y criterio de instancia mínima: ver `reports/load/finops.md` (AWS EC2 us-east-1, Linux, precios del 2026-11-19; ALB verificado el 2026-11-22 en aws.amazon.com/elasticloadbalancing/pricing; memoria pico × 1.2 + 0.5 GiB y vCPU pico × 1.2). Costos por escenario en `reports/load/costos_inferencia.csv` (#163).

## Datos base

Picos por servicio medidos con `docker stats` durante las corridas de Locust A, B y C (`reports/load/docker_stats_picos.csv`, #162):

| configuración | api | mlflow | nginx | instancia mínima actual | USD / 1M inferencias |
|---|---|---|---|---|---|
| 1 réplica | 1131–1159 MiB, 1.23–1.53 vCPU | 1835–1916 MiB, **CPU promedio 2.5–5 %** | – | t3.medium (B, C) / t3.large (A) | 0.53–1.55 |
| 2 réplicas | 2066–2294 MiB, 1.51–3.21 vCPU | 1827–1932 MiB, CPU promedio 2.2–3.6 % | 6–9 MiB, ≤ 0.09 vCPU | t3.large (A) / t3.xlarge (B, C) | 1.28–2.00 |

## Optimizaciones

| Optimización | Dato que la respalda | Ahorro estimado | Esfuerzo | Riesgo |
|---|---|---|---|---|
| **Sacar mlflow del host de inferencia** (right-sizing) | mlflow ocupa 1.8–1.9 GiB (~63 % de la memoria del host con 1 réplica) y bajo carga su CPU promedio es 2.5–5 %: la api solo lo usa al arrancar (`lifespan` carga los 3 modelos `@champion` y después predice desde memoria). Sin mlflow, 1 réplica cabe en t3.small: 1.13 GiB × 1.2 + 0.5 = 1.86 GiB ≤ 2 GiB y 1.53 vCPU × 1.2 = 1.84 ≤ 2 | **50 %** en B y C (t3.medium → t3.small; B: 0.56 → 0.28 USD / 1M); **75 %** en A (t3.large → t3.small; 1.55 → 0.39 USD / 1M) | Medio: copiar los modelos a la imagen o a un volumen y dejar mlflow solo para entrenamiento y registro | Medio: actualizar un `@champion` exige reconstruir o reiniciar la api |
| **Escalado horizontal vs. vertical** | 2 réplicas en el mismo host bajan el p95 89 % en A (2300 → 260 ms) y 69 % en B (1300 → 400 ms), pero en B y C obligan a t3.xlarge y el costo por inferencia sube 3.2–3.6×. Escalar **verticalmente** no ayuda: con 1 réplica la api no pasa de 1.53 vCPU aunque el host tenía 8 núcleos (un solo proceso de uvicorn) | Escalar **horizontalmente en hosts chicos** (sin mlflow): 2 × t3.small (0.0416 USD/h) + ALB (0.0225 USD/h fijo + 0.008 USD por LCU-hora; ~1 LCU con < 30 RPS) = ~0.072 USD/h contra t3.xlarge (0.1664 USD/h): **~57 %** menos para la configuración de 2 réplicas | Medio: requiere el cambio de mlflow y un balanceador administrado | Bajo-medio: más instancias que operar |
| **Instancias spot** | t3.large spot ~0.0314 USD/h contra 0.0832 on-demand; mismo ~62 % en todos los escenarios (B, 1 réplica: 0.56 → 0.21 USD / 1M) | **~62 %** | Bajo: mismo despliegue | Alto: AWS puede reclamar la instancia con 2 min de aviso; usar solo para réplicas extra de pico |
| **Instancias reservadas / Savings Plan (1 año)** | t3.small 0.0130 USD/h con Savings Plan contra 0.0208 on-demand | **~37 %** sobre la réplica base | Bajo: compra en consola | Bajo-medio: compromiso de pago por 1 año aunque no se use |
| **Evitar el recargo por créditos de CPU de t3** | El CPU promedio del host fue 0.96–1.14 vCPU con 1 réplica, contra 0.4 vCPU de base en t3.medium. En modo *unlimited* el excedente cuesta 0.05 USD por vCPU-hora: con la carga de B continua, (1.05 − 0.4) × 0.05 = 0.033 USD/h | Evita hasta **+78 %** sobre el precio on-demand (0.0416 → 0.074 USD/h) si la carga es sostenida; con carga intermitente como A o C, t3 sigue conviniendo | Bajo: monitorear `CPUSurplusCreditsCharged` y, si la carga es continua, cambiar de familia | Bajo |
| **Caché de modelos en el arranque** | La api tarda **45–53 s** en estar *healthy* tras `docker compose up` (3 corridas), de los cuales ~20 s son el arranque de mlflow y el resto la descarga de los modelos; `start_period` está en 90 s | En costo por hora es poco (45 s de t3.medium ≈ 0.0005 USD por arranque), pero cada réplica nueva atendería en segundos: importa en picos como C, donde el p95 llegó a 4.7 s | Medio (el mismo cambio que sacar mlflow) | Bajo |
| **Batching de predicciones** | Hoy cada endpoint predice 1 fila por petición (`pd.DataFrame([data.model_dump()])`) | No medido: requiere un endpoint por lotes y un escenario de Locust que lo use | Medio | Bajo |
| **Redis + worker RQ** (opcional) | En el pico C con 1 réplica no hubo fallas (0 %), pero el p95 fue 4.7 s y el sistema atendió 22.35 de ~50 RPS posibles: el cuello de botella es CPU de la api | **No se recomienda**: una cola no agrega capacidad de CPU, solo convierte la espera en asincronía, y suma 2 servicios. Escalar réplicas da −9 % de p95 y −54 % de mediana en C (2600 → 1200 ms) | Alto: 2 servicios nuevos y cambio del contrato de la api | Medio: más piezas que operar |

## Prioridad recomendada

1. **Sacar mlflow del host de inferencia:** reduce el costo por inferencia 50–75 % sin tocar el modelo y es requisito de las demás.
2. **Savings Plan** para la réplica base (−37 %) y vigilar el recargo por créditos de CPU si la carga es continua.
3. **Escalar horizontalmente** con t3.small detrás de un balanceador solo cuando el p95 lo exija (en B pasa de 1.3 s a 0.4 s), usando **spot** (−62 %) para las réplicas de pico.
