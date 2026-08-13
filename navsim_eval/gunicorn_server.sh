#!/bin/bash
set -euo pipefail
# NAVSIM Reward Function API Server
# Reads PROJECT_ROOT and DATA_ROOT from environment variables.
# Used by scripts/run_adas.sh and scripts/run_rl_train.sh.

: "${PROJECT_ROOT:?Error: PROJECT_ROOT is not set}"
: "${DATA_ROOT:?Error: DATA_ROOT is not set}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export CACHE_SPLIT="${CACHE_SPLIT:-navtrain}"
export CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_${CACHE_SPLIT}}"

HOST="0.0.0.0"
PORT="${REWARD_SERVER_PORT:-8901}"
NUM_WORKERS="${NAVSIM_SERVER_WORKERS:-1}"

echo "[INFO] Starting Gunicorn server on $HOST:$PORT with $NUM_WORKERS workers..."

gunicorn navsim.planning.script.run_gunicorn_server:app \
    -w $NUM_WORKERS \
    -k uvicorn.workers.UvicornWorker \
    -b $HOST:$PORT \
    --timeout 150 \
    --log-level info
