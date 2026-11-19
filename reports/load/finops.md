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

## Recomendaciones
<!-- se detallan en #164 -->
