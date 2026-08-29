#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
RUN_ID=v3_s1_confirm908_g4_seed20260828
RUN_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/rollout_bank"
RUN_DIR="$RUN_ROOT/$RUN_ID"
CANDIDATE_DIR="$RUN_ROOT/v3_s1_candidate_freeze_20260829"
PARQUET="$CANDIDATE_DIR/candidate_908.parquet"
MANIFEST="$CANDIDATE_DIR/candidate_908.txt"
MODEL="$WORKSPACE_ROOT/models/sft_stage2"
CACHE_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap/metric_cache"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
REWARD_PORT=8901

for path in "$PYTHON" "$PARQUET" "$MANIFEST" "$CANDIDATE_DIR/COMPLETE" "$MODEL" "$CACHE_ROOT/metadata/scene_metric_cache.csv"; do
    [[ -e "$path" ]] || { echo "Missing V3-S1 Confirm input: $path" >&2; exit 1; }
done
[[ "$(grep -cve '^[[:space:]]*$' "$MANIFEST")" == 908 ]] || { echo "Confirm manifest is not 908 tokens" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite run directory: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/adas/$RUN_ID" ]] || { echo "ADAS output already exists" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" ]] || { echo "Debug output already exists" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]] || { echo "GPU is already in use" >&2; exit 1; }
[[ -z "$(fuser "$REWARD_PORT/tcp" 2>/dev/null || true)" ]] || { echo "Port $REWARD_PORT is in use" >&2; exit 1; }
[[ "$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')" -ge 20971520 ]] || { echo "Confirm requires 20 GiB free" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
SOURCE_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=%s\nsource_commit=%s\nmodel=%s\nparquet=%s\nmanifest=%s\nrollout_n=4\nseed=20260828\ntemperature=1.0\ntop_p=1.0\n' \
    "$RUN_ID" "$SOURCE_COMMIT" "$MODEL" "$PARQUET" "$MANIFEST" > "$RUN_DIR/run.env"
sha256sum "$PARQUET" "$MANIFEST" "$CANDIDATE_DIR/candidate_freeze_report.json" > "$RUN_DIR/input_sha256.txt"
sha256sum "$MODEL"/model-*.safetensors "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$RUN_DIR/model_sha256.txt"
cp "$PROJECT_ROOT/EasyR1/examples/config_v3_s1_single_gpu.yaml" "$RUN_DIR/resolved_source_config.yaml"

cleanup() {
    status=$?
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; fi
    if [[ -n "${REWARD_SERVER_PID:-}" ]]; then kill "$REWARD_SERVER_PID" 2>/dev/null || true; wait "$REWARD_SERVER_PID" 2>/dev/null || true; fi
    "$WORKSPACE_ROOT/envs/curious/bin/ray" stop --force >/dev/null 2>&1 || true
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -ne 0 ]]; then touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

printf 'timestamp,memory_used_mib,memory_free_mib,utilization_percent\n' > "$RUN_DIR/gpu_memory.csv"
(
    while [[ -e "$RUN_DIR/RUNNING" ]]; do
        printf '%s,' "$(date +%s)"
        nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' '
        sleep 1
    done
) >> "$RUN_DIR/gpu_memory.csv" &
GPU_MONITOR_PID=$!

(
    export PROJECT_ROOT
    export DATA_ROOT="$WORKSPACE_ROOT/data/navsim"
    export OPENSCENE_DATA_ROOT="$DATA_ROOT"
    export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
    export NAVSIM_EXP_ROOT="$WORKSPACE_ROOT/exp_root"
    export CACHE_PATH="$CACHE_ROOT"
    cd "$PROJECT_ROOT/navsim_eval"
    exec "$WORKSPACE_ROOT/envs/navsim/bin/gunicorn" navsim.planning.script.run_gunicorn_server:app \
        -w 4 -k uvicorn.workers.UvicornWorker -b "127.0.0.1:$REWARD_PORT" --timeout 150
) > "$RUN_DIR/reward_server.log" 2>&1 &
REWARD_SERVER_PID=$!

for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:$REWARD_PORT/ping" >/dev/null && break; sleep 2; done
curl -fsS "http://127.0.0.1:$REWARD_PORT/ping" >/dev/null
touch "$RUN_DIR/STARTED"

export EXP_NAME="$RUN_ID"
export NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag
cd "$EASYR1_ROOT"
"$PYTHON" -m verl.trainer.main_adas \
    config=examples/config_v3_s1_single_gpu.yaml \
    data.train_files="$PARQUET@train" data.val_files="$PARQUET@train" \
    data.image_dir="$WORKSPACE_ROOT/data" data.seed=20260828 \
    worker.actor.model.model_path="$MODEL" worker.rollout.seed=20260828 \
    worker.reward.reward_function="$EASYR1_ROOT/verl/utils/reward_score/navsim/navsim_reward_text.py:compute_score_fast" \
    trainer.experiment_name="$RUN_ID" trainer.save_checkpoint_path="$RUN_DIR/tracker"

cp "$EASYR1_ROOT/checkpoints/adas/$RUN_ID/adas_scores.csv" "$RUN_DIR/adas_scores.csv"
mapfile -t rollout_files < <(find "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" -maxdepth 1 -type f -name 'generations_*.jsonl')
[[ "${#rollout_files[@]}" -eq 1 ]] || { echo "Expected one rollout file, found ${#rollout_files[@]}" >&2; exit 1; }
cp "${rollout_files[0]}" "$RUN_DIR/rollouts.jsonl"

"$PYTHON" - "$RUN_DIR/rollouts.jsonl" "$MANIFEST" "$RUN_DIR/diagnosis.json" <<'PY'
import collections, json, math, pathlib, statistics, sys
rollout_path, manifest_path, output_path = map(pathlib.Path, sys.argv[1:])
tokens = [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]
groups = collections.defaultdict(list)
for line in rollout_path.read_text().splitlines():
    row = json.loads(line); groups[str(row["token"])].append(row)
if set(groups) != set(tokens) or any(len(groups[token]) != 4 for token in tokens):
    raise SystemExit("Confirm token/group coverage mismatch")
fields = ("raw_response", "parsed_ok", "poses", "response_length", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "history_comfort", "pdms", "pdms_scaled", "overall_score")
rows = [row for token in tokens for row in groups[token]]
for row in rows:
    if any(field not in row for field in fields): raise SystemExit("Confirm evidence field missing")
    if any(not math.isfinite(float(row[field])) for field in fields[4:]): raise SystemExit("Confirm non-finite metric")
report = {"groups":len(groups), "rollouts":len(rows), "group_size":4, "parse_success_rate":statistics.fmean(bool(row["parsed_ok"]) for row in rows), "clipping_count":sum(int(row["response_length"]) >= 512 for row in rows)}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
PY

sha256sum "$RUN_DIR/rollouts.jsonl" "$RUN_DIR/diagnosis.json" "$RUN_DIR/adas_scores.csv" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
