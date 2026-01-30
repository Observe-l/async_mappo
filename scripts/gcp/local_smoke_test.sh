#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="/home/lwh/anaconda3/envs/default/bin/python"

HOST="127.0.0.1"
PORT="1883"
PREFIX="gcp"
NUM_AGENTS="4"
MAX_STEPS="800"
RUL_THRESHOLD="7"
EXP_TYPE="transformer_s_t7"
DEVICES=(edge-00 edge-01 edge-02 edge-03)

# Disable TF loading on edges for fast smoke tests. Set to a SavedModel dir to enable.
TF_MODEL_DIR="${TF_MODEL_DIR:-}"

LOG_DIR="/tmp"
SERVER_LOG="$LOG_DIR/gcp_server.log"

# Clean up any previous runs
pkill -f "scripts/iot/edge_client.py" >/dev/null 2>&1 || true
pkill -f "run_gcp_server_mqtt.py" >/dev/null 2>&1 || true
rm -f "$LOG_DIR/gcp_edge_"*.log "$LOG_DIR/gcp_edge_"*.pid "$SERVER_LOG" || true

start_client() {
  local dev="$1"
  local log="$LOG_DIR/gcp_edge_${dev#edge-}.log"
  echo "[TEST] starting client $dev -> $log"
  "$PY" -u scripts/iot/edge_client.py \
    --device-id "$dev" \
    --host "$HOST" --port "$PORT" \
    --topic-prefix "$PREFIX" \
    --fresh-actor \
    --tf-model-dir "$TF_MODEL_DIR" \
    >"$log" 2>&1 &
  echo $! > "$LOG_DIR/gcp_edge_${dev#edge-}.pid"
}

stop_clients() {
  for dev in "${DEVICES[@]}"; do
    local pidf="$LOG_DIR/gcp_edge_${dev#edge-}.pid"
    if [[ -f "$pidf" ]]; then
      kill "$(cat "$pidf")" >/dev/null 2>&1 || true
    fi
  done
}

trap stop_clients EXIT

for dev in "${DEVICES[@]}"; do
  start_client "$dev"
done

echo "[TEST] clients started; running server (steps=$MAX_STEPS)"

"$PY" -u scripts/gcp/run_gcp_server_mqtt.py \
  --host "$HOST" --port "$PORT" \
  --topic-prefix "$PREFIX" \
  --num-agents "$NUM_AGENTS" \
  --max-steps "$MAX_STEPS" \
  --episode-length "$MAX_STEPS" \
  --rul-threshold "$RUL_THRESHOLD" \
  --exp-type "$EXP_TYPE" \
  --devices "$(IFS=,; echo "${DEVICES[*]}")" \
  >"$SERVER_LOG" 2>&1

echo "[TEST] server finished ok"

echo
for f in "$SERVER_LOG" \
         "$LOG_DIR/gcp_edge_00.log" \
         "$LOG_DIR/gcp_edge_01.log" \
         "$LOG_DIR/gcp_edge_02.log" \
         "$LOG_DIR/gcp_edge_03.log"; do
  echo "===== tail $f ====="
  tail -n 30 "$f" || true
  echo
 done
