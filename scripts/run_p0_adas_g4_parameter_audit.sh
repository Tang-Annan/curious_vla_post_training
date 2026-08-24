#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/safe_grpo"
RUN_DIR="$EXPERIMENT_ROOT/p0_adas_g4_parameter_audit_seed20260812"
D0_RUN="$EXPERIMENT_ROOT/d0_stage2_train_n4_seed20260812"
ADAS_SCORES="$D0_RUN/adas_scores.csv"
TRAIN_MANIFEST="$WORKSPACE_ROOT/manifests/train_tokens.txt"
DEV_MANIFEST="$WORKSPACE_ROOT/manifests/dev_tokens.txt"
HELDOUT_MANIFEST="$WORKSPACE_ROOT/manifests/heldout_tokens.txt"
RANDOM_MANIFEST="$WORKSPACE_ROOT/manifests/dev_subsets/train_seed20260812_1000.txt"

for path in "$PYTHON" "$D0_RUN/COMPLETE" "$ADAS_SCORES" "$TRAIN_MANIFEST" "$DEV_MANIFEST" \
    "$HELDOUT_MANIFEST" "$RANDOM_MANIFEST"; do
    [[ -e "$path" ]] || { echo "Missing P0 input: $path" >&2; exit 1; }
done
[[ $(cat "$D0_RUN/exit_code") == 0 ]] || { echo "D0 did not complete successfully." >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "P0 output already exists: $RUN_DIR" >&2; exit 1; }
echo "4ade4d7fac38eaec1685249058b1ffe51a402b12d92d606f91cd0eed50930785  $ADAS_SCORES" | sha256sum -c -

cd "$PROJECT_ROOT"
[[ -z $(git status --porcelain) ]] || { echo "Source checkout is not clean." >&2; exit 1; }
mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --porcelain > "$RUN_DIR/source_status.txt"
sha256sum "$ADAS_SCORES" "$TRAIN_MANIFEST" "$DEV_MANIFEST" "$HELDOUT_MANIFEST" \
    "$RANDOM_MANIFEST" > "$RUN_DIR/input_sha256.txt"
printf 'stage=p0_adas_g4_parameter_audit\nseed=20260812\ngpu=none\nn_rollout=4\ngroup_size=4\nstd_threshold=0.01\nconfidence_threshold=0.10\nepsilon_candidates=0.20,0.35\nmaximum_pool_ratio=0.80\n' \
    > "$RUN_DIR/run.env"

cleanup() {
    status=$?
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    if [[ $status -ne 0 ]]; then
        touch "$RUN_DIR/FAILED"
    fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES='' "$PYTHON" "$PROJECT_ROOT/projects/safe_grpo/audit_adas_g4_parameters.py" \
    --adas-scores "$ADAS_SCORES" \
    --train-manifest "$TRAIN_MANIFEST" \
    --dev-manifest "$DEV_MANIFEST" \
    --heldout-manifest "$HELDOUT_MANIFEST" \
    --random-manifest "$RANDOM_MANIFEST" \
    --output-dir "$RUN_DIR" \
    --seed 20260812 > "$RUN_DIR/run.log"

[[ -s "$RUN_DIR/p0_report.json" ]] || { echo "Missing P0 report." >&2; exit 1; }
"$PYTHON" - "$RUN_DIR/p0_report.json" "$RUN_DIR/adas_g4_train_seed20260812_1000.txt" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = pathlib.Path(sys.argv[2])
if bool(report["manifest_written"]) != manifest.exists():
    raise SystemExit("P0 report and manifest presence disagree.")
if manifest.exists():
    tokens = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(tokens) != 1000 or len(set(tokens)) != 1000:
        raise SystemExit("P0 ADAS manifest must contain 1,000 unique tokens.")
PY
if [[ -e "$RUN_DIR/adas_g4_train_seed20260812_1000.txt" ]]; then
    [[ -z $(comm -23 \
        <(sort "$RUN_DIR/adas_g4_train_seed20260812_1000.txt") \
        <(sort "$TRAIN_MANIFEST")) ]] || { echo "P0 manifest contains tokens outside train." >&2; exit 1; }
    [[ -z $(comm -12 \
        <(sort "$RUN_DIR/adas_g4_train_seed20260812_1000.txt") \
        <(sort "$DEV_MANIFEST")) ]] || { echo "P0 manifest overlaps dev." >&2; exit 1; }
    [[ -z $(comm -12 \
        <(sort "$RUN_DIR/adas_g4_train_seed20260812_1000.txt") \
        <(sort "$HELDOUT_MANIFEST")) ]] || { echo "P0 manifest overlaps held-out." >&2; exit 1; }
fi
result_files=(source_commit.txt source_status.txt input_sha256.txt run.env run.log p0_group_stats.csv p0_report.json)
if [[ -e "$RUN_DIR/adas_g4_train_seed20260812_1000.txt" ]]; then
    result_files+=(eligible_pool.txt adas_g4_train_seed20260812_1000.txt)
fi
(
    cd "$RUN_DIR"
    sha256sum "${result_files[@]}" > result_sha256.txt
    sha256sum -c result_sha256.txt
)
touch "$RUN_DIR/COMPLETE"
