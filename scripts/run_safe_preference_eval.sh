#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-}"
MODE="${2:-}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_post_training"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
TRAIN_ENV="$WORKSPACE_ROOT/envs/llamafactory-gpu-py311"
DATA_PATH="$EASYR1_ROOT/data/QA_navtrain_poutine_style_full"
TRAIN_MANIFEST="$WORKSPACE_ROOT/manifests/train_tokens.txt"
DEV_MANIFEST="$WORKSPACE_ROOT/manifests/dev_tokens.txt"
HELDOUT_MANIFEST="$WORKSPACE_ROOT/manifests/heldout_tokens.txt"
EXPECTED_DEV_SHA256=49dd1fae7f8e77589a27af832835bce8f705c0c5b9062145e180890bf3934cfd
M1_DIR="$WORKSPACE_ROOT/experiments/safe_preference/m1_matched_tier_a_seed20260812"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/safe_preference"
MODEL_ROOT="$WORKSPACE_ROOT/models/safe_preference"
E0_RUN="$WORKSPACE_ROOT/experiments/safe_grpo/e0_stage2_dev_seed20260812"
CACHE_PATH="$WORKSPACE_ROOT/exp_root/metric_cache_released_5656"
SEED=20260812
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8901}"
REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_group_fast

case "$METHOD" in
    m2)
        NAME=m2_rsft
        TRAIN_RUN="$EXPERIMENT_ROOT/m2_rsft_seed20260812"
        EXPORT_CONFIG="$PROJECT_ROOT/sft/preference/m2_rsft_export.yaml"
        ;;
    m3)
        NAME=m3_easyneg_dpo
        TRAIN_RUN="$EXPERIMENT_ROOT/m3_easyneg_dpo_seed20260812"
        EXPORT_CONFIG="$PROJECT_ROOT/sft/preference/m3_easyneg_dpo_export.yaml"
        ;;
    m4)
        NAME=m4_hardneg_dpo
        TRAIN_RUN="$EXPERIMENT_ROOT/m4_hardneg_dpo_seed20260812"
        EXPORT_CONFIG="$PROJECT_ROOT/sft/preference/m4_hardneg_dpo_export.yaml"
        ;;
    *)
        echo "Usage: $0 {m2|m3|m4} {prepare|verify|eval}" >&2
        exit 2
        ;;
esac

MERGED_DIR="$MODEL_ROOT/${NAME}_seed20260812_merged"
PREP_RUN="$EXPERIMENT_ROOT/${NAME}_eval_prepare_seed20260812"
EXP_NAME="${NAME}_dev_seed20260812"
RUN_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
DEV_LOCK="$EXPERIMENT_ROOT/${METHOD^^}_DEV_ACCESSED"
M2_RUN="$EXPERIMENT_ROOT/m2_rsft_dev_seed20260812"
M3_RUN="$EXPERIMENT_ROOT/m3_easyneg_dpo_dev_seed20260812"
M2_LOCK="$EXPERIMENT_ROOT/M2_DEV_ACCESSED"
M3_LOCK="$EXPERIMENT_ROOT/M3_DEV_ACCESSED"
M4_LOCK="$EXPERIMENT_ROOT/M4_DEV_ACCESSED"

[[ "$MODE" == prepare || "$MODE" == verify || "$MODE" == eval ]] || {
    echo "Usage: $0 {m2|m3|m4} {prepare|verify|eval}" >&2
    exit 2
}
for path in "$TRAIN_RUN/adapter" "$TRAIN_RUN/train_exit_code" "$EXPORT_CONFIG" "$M1_DIR/dataset_sha256.txt"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ $(cat "$TRAIN_RUN/train_exit_code") == 0 ]] || { echo "Formal training did not exit successfully." >&2; exit 1; }
cd "$PROJECT_ROOT"
[[ -z $(git status --short) ]] || { echo "Source checkout is dirty." >&2; exit 1; }
(cd "$M1_DIR" && sha256sum --check dataset_sha256.txt)

verify_merged() {
    "$TRAIN_ENV/bin/python" - "$MERGED_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
required = ("config.json", "processor_config.json", "tokenizer_config.json", "model.safetensors.index.json")
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"Merged model is missing files: {missing}")
index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
shards = set(index.get("weight_map", {}).values())
if not shards or any(not (root / shard).is_file() for shard in shards):
    raise SystemExit("Merged model shard index is incomplete.")
PY
}

if [[ "$MODE" == prepare ]]; then
    [[ ! -e "$PREP_RUN" ]] || { echo "Refusing to overwrite prepare directory: $PREP_RUN" >&2; exit 1; }
    [[ ! -e "$MERGED_DIR" ]] || { echo "Refusing to overwrite merged model: $MERGED_DIR" >&2; exit 1; }
    mkdir -p "$PREP_RUN" "$MODEL_ROOT"
    touch "$PREP_RUN/RUNNING"
    cleanup_prepare() {
        status=$?
        rm -f "$PREP_RUN/RUNNING"
        printf '%s\n' "$status" > "$PREP_RUN/prepare_exit_code"
    }
    trap cleanup_prepare EXIT
    git rev-parse HEAD > "$PREP_RUN/source_commit.txt"
    git status --short > "$PREP_RUN/source_status.txt"
    cp "$EXPORT_CONFIG" "$PREP_RUN/resolved_export.yaml"
    sha256sum "$EXPORT_CONFIG" "$TRAIN_RUN/adapter/adapter_model.safetensors" > "$PREP_RUN/export_inputs.sha256"
    "$TRAIN_ENV/bin/llamafactory-cli" export "$EXPORT_CONFIG" > "$PREP_RUN/export.log" 2>&1
    verify_merged
    find "$MERGED_DIR" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$PREP_RUN/merged_sha256.txt"
    touch "$PREP_RUN/COMPLETE"
    exit 0
fi

if [[ "$MODE" == verify ]]; then
    [[ -d "$PREP_RUN" && -d "$MERGED_DIR" ]] || { echo "Failed prepare output is missing." >&2; exit 1; }
    [[ $(cat "$PREP_RUN/prepare_exit_code") == 1 ]] || { echo "Verify requires prepare exit code 1." >&2; exit 1; }
    [[ ! -e "$PREP_RUN/COMPLETE" && ! -e "$PREP_RUN/merged_sha256.txt" ]] || {
        echo "Verify refuses an already completed prepare." >&2
        exit 1
    }
    cp "$PREP_RUN/prepare_exit_code" "$PREP_RUN/prepare_attempt0_exit_code"
    verify_cleanup() {
        status=$?
        printf '%s\n' "$status" > "$PREP_RUN/prepare_retry1_exit_code"
        if [[ "$status" -eq 0 ]]; then
            printf '0\n' > "$PREP_RUN/prepare_exit_code"
        fi
    }
    trap verify_cleanup EXIT
    git rev-parse HEAD > "$PREP_RUN/verification_source_commit.txt"
    verify_merged > "$PREP_RUN/verification_retry1.log" 2>&1
    find "$MERGED_DIR" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$PREP_RUN/merged_sha256.txt"
    touch "$PREP_RUN/COMPLETE"
    exit 0
fi

for path in "$PREP_RUN/COMPLETE" "$PREP_RUN/merged_sha256.txt" "$MERGED_DIR" "$DATA_PATH" \
    "$TRAIN_MANIFEST" "$DEV_MANIFEST" "$HELDOUT_MANIFEST" "$CACHE_PATH/metadata" \
    "$E0_RUN/COMPLETE" "$E0_RUN/dev_rollouts.jsonl"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ $(sha256sum "$DEV_MANIFEST" | cut -d ' ' -f1) == "$EXPECTED_DEV_SHA256" ]] || {
    echo "Frozen dev manifest hash changed." >&2
    exit 1
}
[[ $(grep -cve '^[[:space:]]*$' "$DEV_MANIFEST") -eq 566 ]] || { echo "Expected 566 dev tokens." >&2; exit 1; }
[[ $(grep -ve '^[[:space:]]*$' "$DEV_MANIFEST" | sort -u | wc -l) -eq 566 ]] || {
    echo "Frozen dev manifest contains duplicate tokens." >&2
    exit 1
}
[[ -z $(comm -12 <(sort "$TRAIN_MANIFEST") <(sort "$DEV_MANIFEST")) ]] || {
    echo "Frozen train and dev manifests overlap." >&2
    exit 1
}
[[ -z $(comm -12 <(sort "$DEV_MANIFEST") <(sort "$HELDOUT_MANIFEST")) ]] || {
    echo "Frozen dev and legacy-held-out manifests overlap." >&2
    exit 1
}
(cd / && sha256sum --check "$PREP_RUN/merged_sha256.txt")
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite dev run: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$DEV_LOCK" ]] || { echo "Dev access is already locked: $DEV_LOCK" >&2; exit 1; }
case "$METHOD" in
    m2)
        [[ ! -e "$M3_LOCK" && ! -e "$M4_LOCK" ]] || { echo "Dev evaluation order is invalid." >&2; exit 1; }
        ;;
    m3)
        [[ -e "$M2_LOCK" && -e "$M2_RUN/COMPLETE" && ! -e "$M4_LOCK" ]] || {
            echo "M3 dev requires completed M2 dev and no M4 access." >&2
            exit 1
        }
        ;;
    m4)
        [[ -e "$M2_LOCK" && -e "$M2_RUN/COMPLETE" && -e "$M3_LOCK" && -e "$M3_RUN/COMPLETE" ]] || {
            echo "M4 dev requires completed M2 and M3 dev." >&2
            exit 1
        }
        ;;
esac
[[ ! -e "$EASYR1_ROOT/checkpoints/debug/$EXP_NAME" ]] || { echo "Debug output already exists." >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/adas/$EXP_NAME" ]] || { echo "ADAS output already exists." >&2; exit 1; }
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]') ]] || {
    echo "GPU is already in use." >&2
    exit 1
}
if command -v fuser >/dev/null 2>&1; then
    ! fuser "$REWARD_SERVER_PORT/tcp" >/dev/null 2>&1 || { echo "Reward port is already in use." >&2; exit 1; }
fi

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
cleanup_eval() {
    status=$?
    if [[ -n "${REWARD_SERVER_PID:-}" ]]; then
        kill "$REWARD_SERVER_PID" 2>/dev/null || true
        wait "$REWARD_SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
}
trap cleanup_eval EXIT
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --short > "$RUN_DIR/source_status.txt"
cp "$DEV_MANIFEST" "$RUN_DIR/dev_tokens.txt"
cp "$PREP_RUN/merged_sha256.txt" "$RUN_DIR/merged_sha256.txt"
printf 'method=%s\nexperiment=%s\nseed=%s\ndev_manifest=%s\nmerged_model=%s\nrollouts_per_token=1\ntemperature=0.6\ntop_p=0.95\nmax_response_length=512\none_time_dev_access=true\n' \
    "$METHOD" "$EXP_NAME" "$SEED" "$DEV_MANIFEST" "$MERGED_DIR" > "$RUN_DIR/run.env"
sha256sum "$RUN_DIR/dev_tokens.txt" > "$RUN_DIR/dev_manifest.sha256"
exec > "$RUN_DIR/run.log" 2>&1

(
    export PROJECT_ROOT DATA_ROOT="$PROJECT_ROOT/datasets/navsim" CACHE_PATH
    export REWARD_SERVER_PORT OPENSCENE_DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
    export NAVSIM_EXP_ROOT="$WORKSPACE_ROOT/exp_root"
    cd "$PROJECT_ROOT/navsim_eval"
    exec "$WORKSPACE_ROOT/envs/navsim/bin/gunicorn" \
        navsim.planning.script.run_gunicorn_server:app \
        -w 1 -k uvicorn.workers.UvicornWorker \
        -b "127.0.0.1:$REWARD_SERVER_PORT" --timeout 150
) > "$RUN_DIR/reward_server.log" 2>&1 &
REWARD_SERVER_PID=$!
for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$REWARD_SERVER_PORT/ping" >/dev/null && break
    sleep 2
done
curl -fsS "http://127.0.0.1:$REWARD_SERVER_PORT/ping" >/dev/null
set -o noclobber
printf 'method=%s\nrun_dir=%s\n' "$METHOD" "$RUN_DIR" > "$DEV_LOCK"
set +o noclobber

export EXP_NAME NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag
export NAVSIM_REWARD_URL="http://127.0.0.1:$REWARD_SERVER_PORT"
cd "$EASYR1_ROOT"
"$WORKSPACE_ROOT/envs/curious/bin/python" -m verl.trainer.main_adas \
    config=examples/config_vla_single_gpu_lora.yaml \
    data.train_files="$DATA_PATH@train" \
    data.val_files="$DATA_PATH@train" \
    data.image_dir="$PROJECT_ROOT/datasets" \
    data.token_filter_file="$DEV_MANIFEST" \
    data.val_token_filter_file="$DEV_MANIFEST" \
    data.shuffle=false \
    data.max_response_length=512 \
    worker.actor.model.model_path="$MERGED_DIR" \
    worker.actor.model.lora.rank=0 \
    worker.reward.reward_function="$REWARD_FUNCTION" \
    worker.rollout.n=1 \
    worker.rollout.temperature=0.6 \
    worker.rollout.top_p=0.95 \
    worker.rollout.seed="$SEED" \
    worker.rollout.enforce_eager=false \
    worker.rollout.gpu_memory_utilization=0.55 \
    worker.rollout.max_num_batched_tokens=4608 \
    worker.rollout.disable_tqdm=true \
    trainer.find_last_checkpoint=false \
    trainer.experiment_name="$EXP_NAME"

mapfile -t rollout_files < <(find "checkpoints/debug/$EXP_NAME" -maxdepth 1 -name 'generations_*.jsonl' -type f)
[[ ${#rollout_files[@]} -eq 1 ]] || { echo "Expected one dev rollout file, found ${#rollout_files[@]}." >&2; exit 1; }
cp "${rollout_files[0]}" "$RUN_DIR/dev_rollouts.jsonl"
cp "checkpoints/adas/$EXP_NAME/adas_scores.csv" "$RUN_DIR/adas_scores.csv"
"$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
    "$RUN_DIR/dev_rollouts.jsonl" \
    --manifest "$DEV_MANIFEST" \
    --expected-rollouts 1 \
    --max-response-length 512 \
    > "$RUN_DIR/final_dev_metrics.json"
"$WORKSPACE_ROOT/envs/curious/bin/python" - \
    "$RUN_DIR/dev_rollouts.jsonl" "$E0_RUN/dev_rollouts.jsonl" "$DEV_MANIFEST" <<'PY'
import json
import pathlib
import sys

candidate_path, baseline_path, manifest_path = map(pathlib.Path, sys.argv[1:])
load = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
candidate = load(candidate_path)
baseline = load(baseline_path)
manifest = [line.strip() for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
tokens = [str(row["token"]) for row in candidate]
if len(candidate) != 566 or len(set(tokens)) != 566 or set(tokens) != set(manifest):
    raise SystemExit("Dev rollout coverage is not exactly the frozen 566-token set.")
if tokens != [str(row["token"]) for row in baseline]:
    raise SystemExit("Dev rollout order differs from the frozen Stage-2/E2 evaluation order.")
required = {
    "pdms_scaled", "pdms", "safe", "no_at_fault_collisions", "drivable_area_compliance",
    "ego_progress", "time_to_collision_within_bound", "history_comfort", "reward_latency_ms",
    "parsed_ok", "response_length", "poses",
}
for row in candidate:
    missing = required - row.keys()
    if missing:
        raise SystemExit(f"Dev rollout {row['token']} is missing metrics: {sorted(missing)}")
PY
touch "$RUN_DIR/COMPLETE"
