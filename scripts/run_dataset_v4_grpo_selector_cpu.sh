#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
RUN_DIR="$EXPERIMENT_ROOT/semantic_audit/v4_safety_bucket_selector_20260901"
GPU_B="$EXPERIMENT_ROOT/formal_runs/v4_risk50_safety_g4_b4_seed20260827"
SCREEN_ROLLOUTS="$EXPERIMENT_ROOT/rollout_bank/v3_s1_screen8000_g4_seed20260827/rollouts.jsonl"
CONFIRM_ROLLOUTS="$EXPERIMENT_ROOT/rollout_bank/v3_s1_confirm908_g4_seed20260828/rollouts.jsonl"
CONFIRM_SOURCE="$EXPERIMENT_ROOT/rollout_bank/v3_s1_candidate_freeze_20260829/candidate_908.txt"
LABELS="$EXPERIMENT_ROOT/semantic_audit/v4_experiment_closure_cpu_20260831_r1/results/train_current_visible_exclusive_labels.csv"
BASELINE="$EXPERIMENT_ROOT/semantic_audit/v4_risk_ratio_audit_20260831/results/frozen_current_visible_risk50_2000.txt"
RANDOM_MANIFEST="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap/random_train_2000.txt"
SCREEN_PARQUET="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap/hf/grpo_screen.parquet"
CACHE_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap/metric_cache"
REWARD_PORT=8901

for path in "$PYTHON" "$GPU_B/exit_code" "$SCREEN_ROLLOUTS" "$CONFIRM_ROLLOUTS" \
    "$CONFIRM_SOURCE" "$LABELS" "$BASELINE" "$RANDOM_MANIFEST" "$SCREEN_PARQUET" \
    "$CACHE_ROOT/metadata/scene_metric_cache.csv"; do
    [[ -e "$path" ]] || { echo "Missing V4 safety selector input: $path" >&2; exit 1; }
done
[[ ! -e "$GPU_B/RUNNING" ]] || { echo "GPU-B is still running" >&2; exit 1; }
GPU_B_EXIT_CODE=$(cat "$GPU_B/exit_code")
if [[ -e "$GPU_B/COMPLETE" && "$GPU_B_EXIT_CODE" == 0 ]]; then
    GPU_B_TERMINAL=COMPLETE
elif [[ -e "$GPU_B/FAILED" && "$GPU_B_EXIT_CODE" != 0 ]]; then
    GPU_B_TERMINAL=FAILED
else
    echo "GPU-B does not have a consistent terminal state" >&2
    exit 1
fi
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite V4 safety selector run" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]] || {
    echo "GPU is still in use" >&2
    exit 1
}
if (exec 3<>"/dev/tcp/127.0.0.1/$REWARD_PORT") 2>/dev/null; then
    exec 3>&- 3<&- || true
    echo "Reward port is still in use" >&2
    exit 1
fi
[[ "$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')" -ge 10485760 ]] || {
    echo "V4 safety selector requires 10 GiB free" >&2
    exit 1
}

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v4_safety_bucket_selector_20260901\nreward=safety_continuous\nanchor_trials=30,25,20\ngpu_b_dependency=resource_exclusion_only\ngpu_b_terminal=%s\ngpu_b_exit_code=%s\ndev_accessed=false\nfinal_accessed=false\ngpu_used=false\n' \
    "$GPU_B_TERMINAL" "$GPU_B_EXIT_CODE" > "$RUN_DIR/run.env"

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

tail -n +2 "$LABELS" | cut -d, -f1 | tr -d '\r' > "$RUN_DIR/risk_pool_4005.txt"
grep -Fxf "$RUN_DIR/risk_pool_4005.txt" "$CONFIRM_SOURCE" > "$RUN_DIR/risk_confirm.txt"
[[ "$(grep -cve '^[[:space:]]*$' "$RUN_DIR/risk_pool_4005.txt")" == 4005 ]] || {
    echo "Risk-pool manifest is not 4,005 tokens" >&2
    exit 1
}
[[ "$(grep -cve '^[[:space:]]*$' "$RUN_DIR/risk_confirm.txt")" -gt 0 ]] || {
    echo "Risk-pool confirm intersection is empty" >&2
    exit 1
}
sha256sum "$SCREEN_ROLLOUTS" "$CONFIRM_ROLLOUTS" "$LABELS" "$BASELINE" "$RANDOM_MANIFEST" \
    "$SCREEN_PARQUET" "$RUN_DIR/risk_pool_4005.txt" "$RUN_DIR/risk_confirm.txt" \
    > "$RUN_DIR/input_sha256.txt"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=
"$PYTHON" -m pytest tests/test_v4_grpo_selector.py tests/test_v4_reward_audit.py -q \
    --basetemp="/tmp/pytest_v4_safety_selector_20260901_$$"
"$PYTHON" -m compileall -q projects/dataset_v3/v4_grpo_selector.py
bash -n scripts/run_dataset_v4_grpo_selector_cpu.sh
git diff --check

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

"$PYTHON" -m projects.dataset_v3.v4_reward_audit replay \
    --input "$SCREEN_ROLLOUTS" \
    --manifest "$RUN_DIR/risk_pool_4005.txt" \
    --output "$RUN_DIR/screen_safety_enriched.jsonl" \
    --report "$RUN_DIR/screen_replay_report.json" \
    --workers 8
"$PYTHON" -m projects.dataset_v3.v4_reward_audit replay \
    --input "$CONFIRM_ROLLOUTS" \
    --manifest "$RUN_DIR/risk_confirm.txt" \
    --output "$RUN_DIR/confirm_safety_enriched.jsonl" \
    --report "$RUN_DIR/confirm_replay_report.json" \
    --workers 8

kill "$REWARD_SERVER_PID"
wait "$REWARD_SERVER_PID" 2>/dev/null || true
REWARD_SERVER_PID=

"$PYTHON" -m projects.dataset_v3.v4_grpo_selector \
    --risk-labels "$LABELS" \
    --screen-enriched "$RUN_DIR/screen_safety_enriched.jsonl" \
    --confirm-enriched "$RUN_DIR/confirm_safety_enriched.jsonl" \
    --baseline-manifest "$BASELINE" \
    --random-manifest "$RANDOM_MANIFEST" \
    --screen-parquet "$SCREEN_PARQUET" \
    --data-root "$WORKSPACE_ROOT/data" \
    --output-dir "$RUN_DIR/results"

find "$RUN_DIR/results" -maxdepth 1 -type f -print0 | sort -z | \
    xargs -0 sha256sum > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
