#!/usr/bin/env bash
# Prueba del endpoint de la API MiBici corriendo en el contenedor (issue #125).
# Uso: ./scripts/test_api.sh [URL_BASE]
#   URL_BASE por defecto: http://localhost:8000
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "== 1. Health check =="
curl -sS -m 10 "${BASE_URL}/health" | tee /tmp/mibici_health.json
echo; echo

echo "== 2. Predicción (POST /predict) =="
# Payload de ejemplo: ajustar a las variables reales del modelo elegido (issue #127)
REQUEST_BODY='{
  "estacion": 123,
  "dia_semana": 5,
  "hora": 8,
  "mes": 10
}'

echo "Request:"
echo "${REQUEST_BODY}"
echo

echo "Response:"
curl -sS -m 20 \
  -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d "${REQUEST_BODY}" \
  -w "\nHTTP status: %{http_code}\n" | tee /tmp/mibici_predict.json

echo
echo "== 3. Entrada inválida (debe responder 422) =="
curl -sS -m 10 \
  -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"campo_inexistente": "x"}' \
  -o /dev/null -w "HTTP status: %{http_code}\n"

echo
echo "Prueba terminada. Respuestas guardadas en /tmp/mibici_health.json y /tmp/mibici_predict.json"
