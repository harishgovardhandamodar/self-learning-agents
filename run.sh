#!/usr/bin/env bash
# Run the Self-Learning Agent Lab: API on :8220, frontend at /.
set -euo pipefail
cd "$(dirname "$0")"
python3 -c "import fastapi, uvicorn" 2>/dev/null || pip install -r requirements.txt
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8220}"
exec python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT"
