#!/usr/bin/env bash
# Validación de reproducibilidad desde un clon limpio (issue #190).
#
# Clona el repositorio en una carpeta nueva, levanta un servidor MLflow propio
# en otro puerto (no toca el mlflow.db ni los mlruns/ del equipo) y ejecuta el
# pipeline completo con `mlflow run`:
#   ingest -> features -> baselines -> modelos elegidos -> register
# Después crea un entorno con requirements.txt, corre pytest y verifica el
# Model Registry del clon. Guarda la salida de cada paso y un resumen en
# Markdown con tiempos y resultado.
#
# Uso:
#   bash scripts/validar_clon_limpio.sh [RAMA] [DESTINO] [PUERTO]
#     RAMA     rama a clonar           (por defecto: main)
#     DESTINO  carpeta del clon        (por defecto: /tmp/mibici-clon-limpio)
#     PUERTO   puerto del MLflow local (por defecto: 5002)
#
# Requisitos: git, curl, mlflow y uv en el PATH (por ejemplo, con el .venv del
# proyecto activado y `pip install uv==0.13.5`). La ingesta descarga ~1.2 GB.
set -uo pipefail

BRANCH="${1:-main}"
DEST="${2:-/tmp/mibici-clon-limpio}"
PORT="${3:-5002}"
# REPO_URL se puede sobrescribir para clonar desde una copia local
# (p. ej. REPO_URL="$PWD" para validar un commit antes de hacer push).
REPO_URL="${REPO_URL:-https://github.com/laloz60/TC5061-Proyecto-Equipo01.git}"
EXPERIMENT="mibici-equipo01"

for cmd in git curl mlflow uv; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Falta '$cmd' en el PATH."
    [[ "$cmd" == "uv" ]] && echo "Instálalo con: pip install uv==0.13.5"
    exit 1
  fi
done

if [[ -e "$DEST" ]]; then
  echo "La carpeta $DEST ya existe. Bórrala o indica otro destino: un clon limpio debe empezar vacío."
  exit 1
fi

echo "== Clonando $REPO_URL (rama $BRANCH) en $DEST"
git clone --quiet -b "$BRANCH" "$REPO_URL" "$DEST" || exit 1
cd "$DEST" || exit 1

LOG_DIR="$DEST/validacion_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/resumen.md"

# Servidor MLflow propio del clon, en segundo plano
echo "== Levantando MLflow en el puerto $PORT"
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns \
  --port "$PORT" >"$LOG_DIR/mlflow_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && break
  sleep 1
done
if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "MLflow no respondió en el puerto $PORT; revisa $LOG_DIR/mlflow_server.log"
  exit 1
fi
export MLFLOW_TRACKING_URI="http://127.0.0.1:${PORT}"

{
  echo "# Validación de reproducibilidad desde clon limpio (issue #190)"
  echo
  echo "| Dato | Valor |"
  echo "|---|---|"
  echo "| Fecha | $(date '+%Y-%m-%d %H:%M') |"
  echo "| Rama / commit | \`$BRANCH\` / \`$(git rev-parse --short HEAD)\` |"
  echo "| Sistema | $(uname -sm) |"
  echo "| MLflow (CLI) | $(mlflow --version 2>/dev/null | awk '{print $NF}') |"
  echo "| uv | $(uv --version 2>/dev/null | awk '{print $2}') |"
  echo "| Tracking URI | \`$MLFLOW_TRACKING_URI\` (servidor propio del clon) |"
  echo
  echo "| # | Paso | Comando | Resultado | Duración |"
  echo "|---|---|---|---|---|"
} >"$SUMMARY"

STEP=0
FAILED=0
run_step() {
  # run_step <nombre> <critico:si|no> <comando...>
  local name="$1" critical="$2"
  shift 2
  STEP=$((STEP + 1))
  local log="$LOG_DIR/$(printf '%02d' "$STEP")_${name}.log"
  local start end dur status
  echo "== [$STEP] $name"
  start=$(date +%s)
  if "$@" >"$log" 2>&1; then status="OK"; else status="FALLÓ"; FAILED=$((FAILED + 1)); fi
  end=$(date +%s)
  dur=$((end - start))
  printf -v dur_txt '%dm %02ds' $((dur / 60)) $((dur % 60))
  echo "   $status en $dur_txt (log: $log)"
  echo "| $STEP | $name | \`$*\` | $status | $dur_txt |" >>"$SUMMARY"
  if [[ "$status" != "OK" && "$critical" == "si" ]]; then
    echo "   Paso crítico fallido; se detiene la validación."
    echo >>"$SUMMARY"
    echo "**Validación detenida en el paso $STEP ($name).** Revisa \`$log\`." >>"$SUMMARY"
    exit 1
  fi
}

MR=(mlflow run . --experiment-name "$EXPERIMENT" --env-manager uv)

# 1. Datos y features
run_step ingest si "${MR[@]}" -e ingest
run_step features si "${MR[@]}" -e features
# 2. Baselines
run_step train_p1_regla no "${MR[@]}" -e train -P question=p1 -P model=regla
run_step train_p2_dummy no "${MR[@]}" -e train -P question=p2 -P model=dummy
run_step train_p3_random no "${MR[@]}" -e train -P question=p3 -P model=random
# 3. Modelo elegido de cada pregunta
run_step train_p1_rf no "${MR[@]}" -e train -P question=p1 -P model=rf -P n_estimators=200
run_step train_p2_logreg no "${MR[@]}" -e train -P question=p2 -P model=logreg
run_step train_p3_kmeanspp no "${MR[@]}" -e train -P question=p3 -P model=k-means++ -P n_init=10 -P n_clusters=3
# 4. Model Registry (registry propio del clon, no el del equipo)
run_step register no "${MR[@]}" -e register

# 5. Pruebas y verificación del registry con el entorno de requirements.txt
run_step venv_requirements no bash -c "uv venv --quiet --python 3.12 .venv && uv pip install --quiet --python .venv/bin/python -r requirements.txt"
run_step pytest no .venv/bin/python -m pytest -q
run_step verificar_registry no .venv/bin/python scripts/verificar_registry.py

{
  echo
  if [[ "$FAILED" -eq 0 ]]; then
    echo "**Resultado:** los $STEP pasos terminaron correctamente."
  else
    echo "**Resultado:** $FAILED de $STEP pasos fallaron; revisa sus logs en \`validacion_logs/\`."
  fi
  echo
  echo "Resumen de pytest: \`$(tail -n 1 "$LOG_DIR"/*_pytest.log 2>/dev/null)\`"
} >>"$SUMMARY"

echo
echo "== Resumen en $SUMMARY"
cat "$SUMMARY"
[[ "$FAILED" -eq 0 ]]
