#!/usr/bin/env bash
set -euo pipefail

STAGE=""
RUN_ID=""
PYTHON=""
PROJECT_ROOT=""
DATA_ROOT=""
DATASET_DIR=""
TRAIN_PARQUET=""
DEV_PARQUET=""
MANIFEST_DIR=""
CACHE_MANIFEST=""
CACHE_DIR=""
FINAL_MANIFEST=""
STAGE2_MODEL=""
EXPERIMENT_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --train-parquet) TRAIN_PARQUET="$2"; shift 2 ;;
        --dev-parquet) DEV_PARQUET="$2"; shift 2 ;;
        --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
        --cache-manifest) CACHE_MANIFEST="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --final-manifest) FINAL_MANIFEST="$2"; shift 2 ;;
        --stage2-model) STAGE2_MODEL="$2"; shift 2 ;;
        --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

for value in STAGE RUN_ID PYTHON PROJECT_ROOT DATA_ROOT DATASET_DIR TRAIN_PARQUET DEV_PARQUET \
    MANIFEST_DIR CACHE_MANIFEST CACHE_DIR FINAL_MANIFEST STAGE2_MODEL EXPERIMENT_ROOT; do
    [[ -n "${!value}" ]] || { echo "Missing required argument for $value" >&2; exit 2; }
done
[[ "$STAGE" == "d0" ]] || { echo "Unsupported Dataset V2 stage: $STAGE" >&2; exit 2; }

for path in "$PYTHON" "$PROJECT_ROOT" "$DATA_ROOT" "$DATASET_DIR" "$TRAIN_PARQUET" "$DEV_PARQUET" \
    "$MANIFEST_DIR" "$CACHE_MANIFEST" "$CACHE_DIR" "$FINAL_MANIFEST" "$STAGE2_MODEL"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

RUN_DIR="$EXPERIMENT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite run directory: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$DATASET_DIR/V2_DATA_FROZEN" ]] || { echo "Dataset V2 is already frozen" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
AVAILABLE_KB=$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')
[[ "$AVAILABLE_KB" -ge 26214400 ]] || { echo "Dataset V2 requires at least 25 GiB free" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
SOURCE_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'stage=%s\nrun_id=%s\ntrain_parquet=%s\ndev_parquet=%s\ncache_manifest=%s\ncache_dir=%s\nexperiment_root=%s\n' \
    "$STAGE" "$RUN_ID" "$TRAIN_PARQUET" "$DEV_PARQUET" "$CACHE_MANIFEST" "$CACHE_DIR" "$EXPERIMENT_ROOT" \
    > "$RUN_DIR/run.env"

cleanup() {
    status=$?
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
}
trap cleanup EXIT

"$PYTHON" "$PROJECT_ROOT/projects/dataset_v2/freeze_dataset_v2.py" \
    --data-root "$DATA_ROOT" \
    --dataset-dir "$DATASET_DIR" \
    --manifest-dir "$MANIFEST_DIR" \
    --train-parquet "$TRAIN_PARQUET" \
    --dev-parquet "$DEV_PARQUET" \
    --cache-manifest "$CACHE_MANIFEST" \
    --cache-dir "$CACHE_DIR" \
    --final-manifest "$FINAL_MANIFEST" \
    --stage2-model "$STAGE2_MODEL" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$RUN_DIR/freeze_report.json" \
    > "$RUN_DIR/freeze_report.stdout.json"

touch "$RUN_DIR/COMPLETE"
