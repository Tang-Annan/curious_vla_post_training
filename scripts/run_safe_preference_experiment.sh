#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-}"
MODE="${2:-}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
REPO_ROOT="$WORKSPACE_ROOT/src/curious_vla_post_training"
TRAIN_ENV="${TRAIN_ENV:-$WORKSPACE_ROOT/envs/llamafactory-gpu-py311}"
M1_DIR="$WORKSPACE_ROOT/experiments/safe_preference/m1_matched_tier_a_seed20260812"

case "$METHOD:$MODE" in
    m2:smoke|m2:resume-check) CONFIG="$REPO_ROOT/sft/preference/m2_rsft_smoke.yaml" ;;
    m2:train) CONFIG="$REPO_ROOT/sft/preference/m2_rsft.yaml" ;;
    m3:smoke|m3:resume-check) CONFIG="$REPO_ROOT/sft/preference/m3_easyneg_dpo_smoke.yaml" ;;
    m3:train) CONFIG="$REPO_ROOT/sft/preference/m3_easyneg_dpo.yaml" ;;
    m4:smoke|m4:resume-check) CONFIG="$REPO_ROOT/sft/preference/m4_hardneg_dpo_smoke.yaml" ;;
    m4:train) CONFIG="$REPO_ROOT/sft/preference/m4_hardneg_dpo.yaml" ;;
    *) echo "Usage: $0 {m2|m3|m4} {smoke|resume-check|train}" >&2; exit 2 ;;
esac

test -x "$TRAIN_ENV/bin/llamafactory-cli"
test -f "$CONFIG"
test -f "$M1_DIR/dataset_sha256.txt"
cd "$REPO_ROOT"
test -z "$(git status --short)"
(cd "$M1_DIR" && sha256sum --check dataset_sha256.txt)
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
if command -v fuser >/dev/null 2>&1; then
    ! fuser 8901/tcp >/dev/null 2>&1
fi

OUTPUT_DIR="$($TRAIN_ENV/bin/python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output_dir"])
PY
)"
RUN_ROOT="$(dirname "$OUTPUT_DIR")"
TRAIN_ARGS=("$CONFIG")
if [[ "$MODE" == "resume-check" ]]; then
    test -d "$OUTPUT_DIR/checkpoint-20"
    TRAIN_ARGS+=(max_steps=21 save_strategy=no)
else
    test ! -e "$RUN_ROOT"
    mkdir -p "$RUN_ROOT"
    git rev-parse HEAD > "$RUN_ROOT/source_commit.txt"
    git status --short > "$RUN_ROOT/source_status.txt"
    git -C "$REPO_ROOT/LLaMA-Factory" rev-parse HEAD > "$RUN_ROOT/llamafactory_commit.txt"
    git -C "$REPO_ROOT/LLaMA-Factory" status --short > "$RUN_ROOT/llamafactory_status.txt"
    test -z "$(cat "$RUN_ROOT/llamafactory_status.txt")"
    sha256sum "$CONFIG" > "$RUN_ROOT/config_sha256.txt"
    cp "$CONFIG" "$RUN_ROOT/resolved_config.yaml"
    cp "$M1_DIR/dataset_sha256.txt" "$RUN_ROOT/m1_dataset_sha256.txt"
    "$TRAIN_ENV/bin/python" - <<'PY' > "$RUN_ROOT/environment.txt"
import importlib.metadata
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
for name in ("llamafactory", "transformers", "accelerate", "peft", "trl", "datasets"):
    print(name, importlib.metadata.version(name))
PY
fi

set +e
"$TRAIN_ENV/bin/llamafactory-cli" train "${TRAIN_ARGS[@]}"
EXIT_CODE=$?
set -e
printf '%s\n' "$EXIT_CODE" > "$RUN_ROOT/${MODE}_exit_code"
exit "$EXIT_CODE"
