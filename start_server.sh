#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9910}"
RELOAD="${RELOAD:-0}"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

old_pids="$(ps -eo pid=,args= | awk -v port="$PORT" '/uvicorn app.main:app/ && $0 ~ ("--port " port "([[:space:]]|$)") {print $1}')"
if [[ -n "$old_pids" ]]; then
  echo "[start_server] 清理旧进程: $old_pids (port=$PORT)"
  kill $old_pids || true
  sleep 1
fi

reload_args=()
if [[ "$RELOAD" == "1" ]]; then
  reload_args+=(--reload)
fi

echo "[start_server] 启动服务: http://$HOST:$PORT"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" "${reload_args[@]}"
