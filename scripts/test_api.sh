#!/usr/bin/env bash
# Prueba de la API MiBici en el contenedor (issues #122 y #125).
# Uso: ./scripts/test_api.sh [URL_BASE]
#   URL_BASE por defecto: http://localhost:8000
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "== 1. Health check (GET /health) =="
# 503 si los modelos no cargaron desde el Model Registry
curl -sS -m 10 "${BASE_URL}/health" -w "\nHTTP status: %{http_code}\n"
echo

# Payload con las 15 features del input schema de los modelos P1/P2
# (mibici/api/schemas.py - verificado contra el Model Registry, issue #126)
REQUEST_BODY='{
  "hora_del_dia": 8,
  "dia_semana": 5,
  "mes": 10,
  "es_fin_de_semana": 0,
  "salidas_lag_1h": 4.0,
  "salidas_lag_24h": 6.0,
  "salidas_lag_168h": 5.0,
  "salidas_media_7d": 5.5,
  "salidas_prom_4sem": 5.2,
  "neto_lag_1h": -1.0,
  "neto_lag_24h": 0.5,
  "neto_media_7d": 0.2,
  "capacidad": 30.0,
  "latitud": 20.6597,
  "longitud": -103.3496
}'

echo "== 2. Prediccion P1 (POST /predict/p1-demanda) =="
echo "Request:"; echo "${REQUEST_BODY}"
curl -sS -m 30 -X POST "${BASE_URL}/predict/p1-demanda" \
  -H "Content-Type: application/json" \
  -d "${REQUEST_BODY}" \
  -w "\nHTTP status: %{http_code}\n"
echo

echo "== 3. Prediccion P2 (POST /predict/p2-riesgo) =="
curl -sS -m 30 -X POST "${BASE_URL}/predict/p2-riesgo" \
  -H "Content-Type: application/json" \
  -d "${REQUEST_BODY}" \
  -w "\nHTTP status: %{http_code}\n"
echo

echo "== 4. Prediccion P3 (POST /predict/p3-clustering) =="
curl -sS -m 30 -X POST "${BASE_URL}/predict/p3-clustering" \
  -H "Content-Type: application/json" \
  -d '{"perfil_horario": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]}' \
  -w "\nHTTP status: %{http_code}\n"
echo

echo "== 5. Entrada invalida (esperado 422 con validacion Pydantic) =="
curl -sS -m 10 -X POST "${BASE_URL}/predict/p1-demanda" \
  -H "Content-Type: application/json" \
  -d '{"campo_inexistente": "x"}' \
  -o /dev/null -w "HTTP status: %{http_code}\n"

echo
echo "Prueba terminada."
