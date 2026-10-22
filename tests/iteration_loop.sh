#!/usr/bin/env bash
# The lab's 50-iteration gate.
#
# Each round:
#   1. runs the mechanism sweep (scenarios x evolvers x iterations)
#   2. runs the unit and API tests
#   3. runs the browser-driven UI audit against the live container
#
# Rounds 1..N (50 by default). Stops at the first failure and prints what
# broke, so a regression is caught by the round it appears in, not fifty
# rounds later.
#
#   ./tests/iteration_loop.sh [rounds] [iterations-per-cell]
#
# Cost: about a minute per round, so 50 rounds is roughly an hour. The
# long-horizon stream is measured with a 12-task hold-out inside the gate
# (it costs nine checkpoints times eight families per run); the reference
# bands measure it at full size.
#
# Requires: python3, node + playwright-core, and the container running
# (docker compose up -d --build).
set -uo pipefail
cd "$(dirname "$0")/.."

ROUNDS=${1:-50}
ITERS=${2:-3}
BASE=${SLA_URL:-http://localhost:8220}
AUDIT_OUT=${SLA_AUDIT_OUT:-/tmp/sla_ui_audit}

if ! curl -sf -m 5 "$BASE/api/health" >/dev/null; then
  echo "ERROR: lab not reachable at $BASE - start it with: docker compose up -d --build" >&2
  exit 2
fi

# playwright-core lives outside the repo (it is a 40MB dev dependency).
# The audit resolves it from NODE_PATH or a local install.
if ! NODE_PATH="${NODE_PATH:-/tmp/node_modules}" node -e "require('playwright-core')" 2>/dev/null; then
  echo "ERROR: playwright-core not found. Install it with:" >&2
  echo "  npm install --prefix /tmp playwright-core" >&2
  exit 2
fi
export NODE_PATH="${NODE_PATH:-/tmp/node_modules}"

FAIL=0
for round in $(seq 1 "$ROUNDS"); do
  echo "=============================================================="
  echo "round $round/$ROUNDS   sweep iterations=$ITERS"
  echo "=============================================================="

  # An unreachable lab is an infrastructure problem, not a regression.
  # Saying so keeps a container restart from being read as a broken build.
  if ! curl -sf -m 5 "$BASE/api/health" >/dev/null; then
    echo "round $round: the lab at $BASE is unreachable - is the container up?"
    FAIL=2
    break
  fi

  python3 -m tests.sweep --iterations "$ITERS" --block 20 --eval 30 --jobs 8 \
    --scenarios topic,workflow,domain,compound,healthcare-drift,trading-drift,science-drift,domain-workflow,revisit,interleaved,recurring,long-horizon,hard-order \
    --eval-by-scenario "long-horizon=12" \
    --out "/tmp/sla_sweep_r${round}.json" 2>&1 | tail -8
  sweep_rc=$?
  if [ $sweep_rc -ne 0 ]; then
    echo "round $round: sweep FAILED (rc=$sweep_rc)"
    FAIL=1
    break
  fi

  python3 -m pytest tests/ -q 2>&1 | tail -3
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "round $round: unit tests FAILED"
    FAIL=1
    break
  fi

  # Start each round from a clean history so the UI checks are deterministic
  # (the run list is a bounded window, so leftovers would hide new runs).
  curl -sf -X DELETE "$BASE/api/runs" >/dev/null 2>&1

  node tests/ui_audit.js "$BASE" "$AUDIT_OUT" 2>&1 | tail -12
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "round $round: UI audit FAILED (see $AUDIT_OUT/report.json)"
    FAIL=1
    break
  fi

  echo "round $round: OK"
done

if [ $FAIL -eq 0 ]; then
  echo "all $ROUNDS rounds passed"
elif [ $FAIL -eq 2 ]; then
  echo "INFRASTRUCTURE: the lab was unreachable - fix the container and re-run"
else
  echo "FAILED - see the output above"
fi
exit $FAIL
