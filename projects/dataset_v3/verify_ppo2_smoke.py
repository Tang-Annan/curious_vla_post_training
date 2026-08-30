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
    checks["log_step_coverage"] = sorted(set(steps)) == [1, 2]
    checks["log_step_count"] = len(steps) == 2

    per_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row["step"])
        per_step.setdefault(step, {}).update(row)

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
    for step in (1, 2):
        row = per_step.get(step)
        if row is None:
            checks["epoch2_executed"] = False
            continue
        for key in required_epoch_keys:
            if key not in row or not _finite(row[key]):
                checks["all_epoch_metrics_finite"] = False
            if key.startswith("actor/epoch2/") and (key not in row or not _finite(row[key])):
                checks["epoch2_executed"] = False

    forbidden = ("OutOfMemoryError", "CUDA error", "RuntimeError", "Traceback", "killed")
    checks["no_error_patterns_in_run_log"] = not any(pattern in run_log for pattern in forbidden)

    passed = all(checks.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "steps": sorted(per_step),
        "epoch_telemetry": {
            str(step): {
                key: row[key]
                for key in required_epoch_keys
                if key in row and isinstance(row[key], (int, float))
            }
            for step, row in sorted(per_step.items())
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(f"PPO2 smoke verification failed: {checks}")


if __name__ == "__main__":
    main()
