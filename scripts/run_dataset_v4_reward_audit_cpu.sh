#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
RUN_ID=v4_reward_audit_20260831_r4
RUN_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/semantic_audit"
RUN_DIR="$RUN_ROOT/$RUN_ID"
S1_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/rollout_bank/v3_s1_screen8000_g4_seed20260827"
MANIFEST="$RUN_ROOT/v4_risk_ratio_audit_20260831/results/frozen_current_visible_risk50_2000.txt"
LABELS="$RUN_ROOT/v4_experiment_closure_cpu_20260831_r1/results/train_current_visible_exclusive_labels.csv"
CACHE_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap/metric_cache"
REWARD_PORT=8901

for path in "$PYTHON" "$S1_DIR/COMPLETE" "$S1_DIR/rollouts.jsonl" "$MANIFEST" "$LABELS" \
    "$CACHE_ROOT/metadata/scene_metric_cache.csv"; do
    [[ -e "$path" ]] || { echo "Missing reward audit input: $path" >&2; exit 1; }
done
[[ "$(cat "$S1_DIR/exit_code")" == 0 ]] || { echo "S1 did not exit successfully" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite run directory: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
if (exec 3<>"/dev/tcp/127.0.0.1/$REWARD_PORT") 2>/dev/null; then
    exec 3>&- 3<&- || true
    echo "Port $REWARD_PORT is in use" >&2
    exit 1
fi
if pidof gunicorn >/dev/null 2>&1; then
    echo "A gunicorn process is already running" >&2
    exit 1
fi
[[ "$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')" -ge 5242880 ]] || { echo "Reward audit requires 5 GiB free" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=%s\nsource_run=%s\nmanifest=%s\nlabels=%s\ncache=%s\nserver_workers=8\nclient_workers=8\ndev_accessed=false\nfinal_accessed=false\n' \
    "$RUN_ID" "$S1_DIR" "$MANIFEST" "$LABELS" "$CACHE_ROOT" > "$RUN_DIR/run.env"
sha256sum "$S1_DIR/rollouts.jsonl" "$MANIFEST" "$LABELS" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    if [[ -n "${REWARD_SERVER_PID:-}" ]]; then
        kill "$REWARD_SERVER_PID" 2>/dev/null || true
        wait "$REWARD_SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -ne 0 ]]; then touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

(
    export PROJECT_ROOT
    export DATA_ROOT="$WORKSPACE_ROOT/data/navsim"
    export OPENSCENE_DATA_ROOT="$DATA_ROOT"
    export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
    export NAVSIM_EXP_ROOT="$WORKSPACE_ROOT/exp_root"
    export CACHE_PATH="$CACHE_ROOT"
    cd "$PROJECT_ROOT/navsim_eval"
    exec "$WORKSPACE_ROOT/envs/navsim/bin/gunicorn" \
        navsim.planning.script.run_gunicorn_server:app \
        -w 8 -k uvicorn.workers.UvicornWorker \
        -b "127.0.0.1:$REWARD_PORT" --timeout 300
) > "$RUN_DIR/reward_server.log" 2>&1 &
REWARD_SERVER_PID=$!

for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$REWARD_PORT/ping" >/dev/null && break
    sleep 2
done
curl -fsS "http://127.0.0.1:$REWARD_PORT/ping" >/dev/null

"$PYTHON" "$PROJECT_ROOT/projects/dataset_v3/v4_reward_audit.py" replay \
    --input "$S1_DIR/rollouts.jsonl" \
    --manifest "$MANIFEST" \
    --output "$RUN_DIR/enriched_risk50_reward.jsonl" \
    --report "$RUN_DIR/replay_report.json" \
    --workers 8

"$PYTHON" "$PROJECT_ROOT/projects/dataset_v3/v4_reward_audit.py" audit \
    --input "$RUN_DIR/enriched_risk50_reward.jsonl" \
    --manifest "$MANIFEST" \
    --labels "$LABELS" \
    --output-dir "$RUN_DIR/results"

sha256sum "$RUN_DIR/enriched_risk50_reward.jsonl" "$RUN_DIR/replay_report.json" \
    "$RUN_DIR/results/candidate_reward_rows.csv" "$RUN_DIR/results/reward_audit_report.json" \
    > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
