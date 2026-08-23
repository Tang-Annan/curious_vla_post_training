#!/usr/bin/env python3
"""Verify that a no-dev G=4 smoke run completed the frozen training protocol."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


REQUIRED_ROLLOUT_FIELDS = (
    "training_reward",
    "pdms",
    "pdms_scaled",
    "safe",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
)


def load_manifest(path: Path) -> set[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Training manifest contains duplicate tokens.")
    return set(tokens)


def finite_numbers(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from finite_numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from finite_numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def verify(
    rollouts_path: Path,
    training_log_path: Path,
    config_path: Path,
    gpu_memory_path: Path,
    manifest_path: Path,
    expected_steps: int,
    groups_per_step: int,
    group_size: int,
) -> dict:
    manifest = load_manifest(manifest_path)
    rows = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8-sig").splitlines() if line]
    expected_groups = expected_steps * groups_per_step
    expected_rollouts = expected_groups * group_size
    if len(rows) != expected_rollouts:
        raise ValueError(f"Expected {expected_rollouts} rollout rows, found {len(rows)}.")

    counts = Counter(str(row["token"]) for row in rows)
    outside = set(counts) - manifest
    if outside:
        raise ValueError(f"Smoke rollouts contain {len(outside)} tokens outside the training manifest.")
    if len(counts) != expected_groups or any(count != group_size for count in counts.values()):
        raise ValueError(f"Expected {expected_groups} unique groups with {group_size} rollouts each.")

    missing_fields = sum(any(field not in row for field in REQUIRED_ROLLOUT_FIELDS) for row in rows)
    if missing_fields:
        raise ValueError(f"{missing_fields} rollout rows are missing required reward fields.")

    log_rows = [
        json.loads(line) for line in training_log_path.read_text(encoding="utf-8-sig").splitlines() if line
    ]
    steps = [int(row["step"]) for row in log_rows]
    if steps != list(range(1, expected_steps + 1)):
        raise ValueError(f"Training log steps are incomplete or out of order: {steps}.")
    if any(not math.isfinite(number) for row in log_rows for number in finite_numbers(row)):
        raise ValueError("Training log contains a non-finite numeric value.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen_values = {
        "rollout_n": config["worker"]["rollout"]["n"],
        "rollout_batch_size": config["data"]["rollout_batch_size"],
        "actor_global_batch_size": config["worker"]["actor"]["global_batch_size"],
        "actor_update_micro_batch": config["worker"]["actor"]["micro_batch_size_per_device_for_update"],
        "actor_experience_micro_batch": config["worker"]["actor"]["micro_batch_size_per_device_for_experience"],
        "gradient_checkpointing": config["worker"]["actor"]["model"]["enable_gradient_checkpointing"],
        "max_steps": config["trainer"]["max_steps"],
        "skip_final_validation": config["trainer"]["skip_final_validation"],
    }
    expected_values = {
        "rollout_n": group_size,
        "rollout_batch_size": groups_per_step,
        "actor_global_batch_size": groups_per_step,
        "actor_update_micro_batch": 1,
        "actor_experience_micro_batch": 1,
        "gradient_checkpointing": True,
        "max_steps": expected_steps,
        "skip_final_validation": True,
    }
    if frozen_values != expected_values:
        raise ValueError(f"Resolved config violates the T0 protocol: {frozen_values!r}.")

    gpu_rows = []
    for line in gpu_memory_path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        if not line.strip():
            continue
        timestamp, used, free, utilization = [part.strip() for part in line.split(",")]
        gpu_rows.append(
            {
                "timestamp": int(timestamp),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_percent": int(utilization),
            }
        )
    if not gpu_rows:
        raise ValueError("GPU memory monitor did not record any samples.")

    return {
        "passed": True,
        "expected_steps": expected_steps,
        "groups_per_step": groups_per_step,
        "group_size": group_size,
        "groups": len(counts),
        "rollouts": len(rows),
        "parse_success_rate": sum(bool(row.get("parsed_ok", True)) for row in rows) / len(rows),
        "required_reward_field_coverage": 1.0,
        "resolved_config": frozen_values,
        "gpu_samples": len(gpu_rows),
        "peak_memory_used_mib": max(row["memory_used_mib"] for row in gpu_rows),
        "minimum_memory_free_mib": min(row["memory_free_mib"] for row in gpu_rows),
        "peak_gpu_utilization_percent": max(row["utilization_percent"] for row in gpu_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-memory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=10)
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    report = verify(
        args.rollouts,
        args.training_log,
        args.config,
        args.gpu_memory,
        args.manifest,
        args.expected_steps,
        args.groups_per_step,
        args.group_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
