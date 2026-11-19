# Análisis FinOps: costo por inferencia

## Supuestos
- Proveedor de referencia: AWS EC2, región `us-east-1`, Linux, precios consultados el 2026-11-19 (compute.doit.com, calculator.holori.com).
- Toda la pila (api + mlflow + nginx) corre en una sola instancia.
- Recursos medidos con `docker stats` cada 2 s (`scripts/load_test.sh`); en docker stats 100 % de CPU = 1 vCPU.
- Instancia mínima: memoria pico × 1.2 + 0.5 GiB (SO y Docker) y vCPU pico × 1.2 deben caber en la instancia.
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
<!-- pegar la tabla que imprime scripts/costo_inferencia.py -->

## Escalabilidad: 1 contra 2 réplicas
<!-- RPS, p95 y costo por millón de inferencias con 1 y 2 réplicas -->

## Right-sizing y spot
<!-- instancia mínima por escenario y ahorro con spot -->

## Optimizaciones

Datos base medidos con `docker stats` (escenario C con `SIN_LOCUST=1`, **sin carga**; se actualizan con las corridas de Locust):

| configuración | api | mlflow | nginx | total | instancia mínima |
|---|---|---|---|---|---|
| 1 réplica | 925 MiB | 1824 MiB | – | 2.7 GiB | t3.medium (0.0416 USD/h) |
| 2 réplicas | 2330 MiB (~1.14 GiB c/u) | 1937 MiB | 4 MiB | 4.2 GiB | t3.large (0.0832 USD/h) |

| Optimización | Dato que la respalda | Ahorro estimado | Esfuerzo | Riesgo |
|---|---|---|---|---|
| **Sacar mlflow del host de inferencia** (right-sizing) | mlflow ocupa 1824 MiB (66 % de la pila), pero la api solo lo usa al arrancar: `lifespan` carga los 3 modelos `@champion` y después predice desde memoria. Sin mlflow, 1 réplica cabe en t3.small (0.9 GiB × 1.2 + 0.5 = 1.6 GiB ≤ 2 GiB) y 2 réplicas en t3.medium (2.3 × 1.2 + 0.5 = 3.2 GiB ≤ 4 GiB) | **50 %**: t3.medium → t3.small (0.0416 → 0.0208 USD/h); con 2 réplicas, t3.large → t3.medium (0.0832 → 0.0416 USD/h) | Medio: copiar los modelos a la imagen o a un volumen y dejar mlflow solo para entrenamiento y registro | Medio: actualizar un `@champion` exige reconstruir o reiniciar la api |
| **Instancias spot** | t3.large spot ~0.0314 USD/h contra 0.0832 on-demand (us-east-1, 2026-11-19) | **~62 %** | Bajo: mismo despliegue | Alto: AWS puede reclamar la instancia con 2 min de aviso; solo sirve con otra instancia de respaldo |
| **Instancias reservadas / Savings Plan (1 año)** | t3.small 0.0130 USD/h con Savings Plan contra 0.0208 on-demand | **~37 %** | Bajo: compra en consola | Bajo-medio: compromiso de pago por 1 año aunque no se use |
| **Escalado horizontal vs. vertical** | 2 réplicas suman ~1.14 GiB por réplica y obligan a pasar de t3.medium a t3.large (+100 % de costo por hora). 2 × t3.medium y 1 × t3.large cuestan lo mismo (0.0832 USD/h) | Pendiente: solo conviene si el RPS sube más de 2× o el p95 baja con 2 réplicas (escenarios A/B/C) | Bajo: ya existe `docker-compose.scale.yml` | Bajo |
| **Caché de modelos en el arranque** | La api tarda **45–53 s** en estar *healthy* tras `docker compose up` (3 corridas), de los cuales ~20 s son el arranque de mlflow y el resto la descarga de los modelos; `start_period` está en 90 s | Por medir: con los modelos en la imagen, cada réplica nueva atiende en segundos. En costo por hora es poco (45 s de t3.medium ≈ 0.0005 USD), pero acorta la reacción ante picos como el escenario C | Medio (el mismo cambio que sacar mlflow) | Bajo |
| **Batching de predicciones** | Hoy cada endpoint predice 1 fila por petición (`pd.DataFrame([data.model_dump()])`) | Pendiente: medir RPS y costo por inferencia con un endpoint por lotes | Medio: endpoint nuevo y escenario de Locust | Bajo |
| **Redis + worker RQ** (opcional) | — | Pendiente: solo se justifica si el escenario C muestra fallas o un p95 alto | Alto: 2 servicios nuevos | Medio: más piezas que operar |

**Prioridad recomendada:** primero sacar mlflow del host de inferencia, porque reduce el costo a la mitad sin tocar el modelo. Después Savings Plan para la carga base y spot solo para réplicas extra en picos.
