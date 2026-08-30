from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}/{key}" if prefix else key, child, output)
    else:
        output[prefix] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--experiment-log", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    rows = _jsonl(args.experiment_log)
    run_log = args.run_log.read_text(encoding="utf-8", errors="replace")

    checks: dict[str, Any] = {
        "config_ppo_epochs": config["worker"]["actor"]["ppo_epochs"] == 2,
        "config_max_steps": config["trainer"]["max_steps"] == 2,
        "config_global_batch_size": config["worker"]["actor"]["global_batch_size"] == 4,
    }

    steps = [int(row["step"]) for row in rows]
    checks["log_step_coverage"] = set(steps) == {1, 2}

    flattened = []
    for row in rows:
        flat: dict[str, Any] = {}
        _flatten("", row, flat)
        flat["step"] = int(row["step"])
        flattened.append(flat)
    update_rows = [flat for flat in flattened if "actor/epoch1/grad_norm" in flat]
    checks["update_step_count"] = len(update_rows) == 2
    checks["update_step_coverage"] = sorted(flat["step"] for flat in update_rows) == [1, 2]

    required_epoch_keys = [
        "actor/epoch1/grad_norm",
        "actor/epoch2/grad_norm",
        "actor/epoch1/log_ratio_mean",
        "actor/epoch2/log_ratio_mean",
        "actor/epoch1/log_ratio_p95",
        "actor/epoch2/log_ratio_p95",
        "actor/epoch1/log_ratio_p99",
        "actor/epoch2/log_ratio_p99",
        "actor/epoch1/ppo_kl",
        "actor/epoch2/ppo_kl",
        "actor/epoch1/pg_loss",
        "actor/epoch2/pg_loss",
        "actor/epoch1/pg_clipfrac_higher",
        "actor/epoch2/pg_clipfrac_higher",
        "actor/epoch1/pg_clipfrac_lower",
        "actor/epoch2/pg_clipfrac_lower",
    ]
    checks["epoch2_executed"] = True
    checks["all_epoch_metrics_finite"] = True
    for flat in update_rows:
        for key in required_epoch_keys:
            if key not in flat or not _finite(flat[key]):
                checks["all_epoch_metrics_finite"] = False
            if key.startswith("actor/epoch2/") and (key not in flat or not _finite(flat[key])):
                checks["epoch2_executed"] = False

    forbidden = ("OutOfMemoryError", "CUDA error", "RuntimeError", "Traceback", "killed")
    checks["no_error_patterns_in_run_log"] = not any(pattern in run_log for pattern in forbidden)

    passed = all(checks.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "steps": sorted(set(steps)),
        "epoch_telemetry": {
            str(flat["step"]): {
                key: flat[key]
                for key in required_epoch_keys
                if key in flat and isinstance(flat[key], (int, float))
            }
            for flat in sorted(update_rows, key=lambda item: item["step"])
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(f"PPO2 smoke verification failed: {checks}")


if __name__ == "__main__":
    main()
