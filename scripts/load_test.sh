#!/usr/bin/env bash
# Prueba de carga + monitoreo de recursos (issue #162, Actividad 10).
#
# Levanta el sistema, corre un escenario de Locust y, en paralelo, muestrea
# `docker stats` cada 2 s en reports/load/docker_stats.csv. Al terminar
# (o si algo falla) detiene el muestreo y baja los servicios.
#
# Uso (desde la raiz del repo, en Git Bash):
#   bash scripts/load_test.sh <A|B|C> [replicas]
#     replicas: 1 (por defecto) o 2 -> agrega docker-compose.scale.yml
#               (nginx como balanceador + 2 replicas de la API)
#   Salidas de Locust: reports/load/<ESC>_<N>r_*.csv y locust_<ESC>_<N>r.html
#
# Variables opcionales:
#   LOCUST_CMD   comando de Locust          (por defecto: locust)
#   SIN_LOCUST=1 solo mide docker stats durante la duracion del escenario
#                (sirve para probar el script sin locustfile.py)
#   INTERVALO    segundos entre muestras    (por defecto: 2)
set -euo pipefail

ESCENARIO="${1:?Uso: bash scripts/load_test.sh <A|B|C> [replicas]}"
REPLICAS="${2:-1}"
LOCUST_CMD="${LOCUST_CMD:-locust}"
INTERVALO="${INTERVALO:-2}"
HOST="http://localhost:8000"
OUT_DIR="reports/load"
STATS_CSV="${OUT_DIR}/docker_stats.csv"

# Escenarios del issue #161: usuarios, usuarios nuevos por segundo, duracion
case "$ESCENARIO" in
  A) USUARIOS=50;  SPAWN=0.8; DURACION=120s ;;  # rampa ~60 s + ~60 s sostenidos
  B) USUARIOS=50;  SPAWN=10;  DURACION=180s ;;  # carga sostenida 3 min
  C) USUARIOS=100; SPAWN=50;  DURACION=60s  ;;  # pico: 100 usuarios en ~2 s
  *) echo "Escenario invalido: $ESCENARIO (usa A, B o C)" >&2; exit 1 ;;
esac

case "$REPLICAS" in
  1) COMPOSE=(docker compose -f docker-compose.yml); SUFIJO="_1r" ;;
  2) COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.scale.yml); SUFIJO="_2r" ;;
  *) echo "replicas debe ser 1 o 2" >&2; exit 1 ;;
esac
ETIQUETA="${ESCENARIO}${SUFIJO}"

mkdir -p "$OUT_DIR"

echo "== Levantando servicios (api x${REPLICAS}) =="
"${COMPOSE[@]}" up -d --wait --scale "api=${REPLICAS}"

STATS_PID=""
limpiar() {
  if [[ -n "$STATS_PID" ]]; then kill "$STATS_PID" 2>/dev/null || true; fi
  echo "== Bajando servicios =="
  "${COMPOSE[@]}" down
}
trap limpiar EXIT

if [[ ! -f "$STATS_CSV" ]]; then
  echo "timestamp,escenario,replicas,contenedor,cpu_perc,mem_usada,mem_limite,mem_perc" > "$STATS_CSV"
fi

# Solo los contenedores de este proyecto (no otros que tengas corriendo)
CONTENEDORES=$("${COMPOSE[@]}" ps -q | tr -d '\r')

muestrear() {
  while true; do
    ts=$(date +%Y-%m-%dT%H:%M:%S)
    # MemUsage viene como "1.2GiB / 4GiB": se separa en dos columnas
    # shellcheck disable=SC2086  # CONTENEDORES se separa por palabras a proposito
    docker stats --no-stream \
      --format "{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" $CONTENEDORES \
      | sed -e "s# / #,#" -e "s#^#${ts},${ESCENARIO},${REPLICAS},#" >> "$STATS_CSV"
    sleep "$INTERVALO"
  done
}
muestrear &
STATS_PID=$!

echo "== Escenario ${ETIQUETA}: ${USUARIOS} usuarios, spawn ${SPAWN}/s, ${DURACION} =="
if [[ "${SIN_LOCUST:-0}" == "1" ]]; then
  sleep "${DURACION%s}"
  echo "== Listo: ${STATS_CSV} (sin Locust) =="
else
  # Locust sale con codigo 1 si hubo peticiones fallidas: no debe cortar el script
  $LOCUST_CMD -f locustfile.py --host "$HOST" \
    -u "$USUARIOS" -r "$SPAWN" -t "$DURACION" --headless \
    --html "${OUT_DIR}/locust_${ETIQUETA}.html" \
    --csv "${OUT_DIR}/${ETIQUETA}" \
    || echo "Aviso: Locust termino con peticiones fallidas (revisar ${OUT_DIR}/${ETIQUETA}_failures.csv)"
  echo "== Listo: ${STATS_CSV}, ${OUT_DIR}/${ETIQUETA}_*.csv y ${OUT_DIR}/locust_${ETIQUETA}.html =="
fi
