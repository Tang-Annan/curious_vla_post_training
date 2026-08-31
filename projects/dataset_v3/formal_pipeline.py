from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from projects.dataset_v3.eval_pipeline import METRIC_FIELDS, TIERS, _classify_tier, _cvar20
from projects.dataset_v3.h0_pipeline import REQUIRED_ACTOR_METRICS, _flatten, _summarize
from projects.dataset_v3.s1_pipeline import read_manifest, sha256_file


SAFETY_FIELDS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
)
CELL_REWARD = {
    "V3-RR": "compute_score_raw_pdms",
    "V3-TR": "compute_score_raw_pdms",
    "V3-RC": "compute_score_cdt_task",
    "V3-TC": "compute_score_cdt_task",
    "V3-TC-PPO2": "compute_score_cdt_task",
    "V4-RISK50": "compute_score_raw_pdms",
}
CELL_METADATA = {"V4-RISK50": {"selector": "Risk50", "reward": "Raw-PDMS"}}
# V3-018 freezes ppo_epochs=2 only for the last TC-PPO2 optimizer attempt; the
# frozen M0 protocol keeps ppo_epochs=1 for the original matrix cells.
CELL_PPO_EPOCHS = {"V3-TC-PPO2": 2}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def analyze_training(args: argparse.Namespace) -> None:
    protocol = json.loads(args.m0_protocol.read_text(encoding="utf-8"))
    config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    if protocol.get("status") != "M0_FROZEN" or args.cell not in CELL_REWARD:
        raise ValueError("Formal training requires the frozen M0 protocol and a registered cell")
    optimization = protocol["optimization"]
    expected_steps = optimization["train_monitor_steps"]
    expected_groups = int(optimization["training_groups"])
    group_size = int(optimization["group_size"])
    groups_per_update = int(optimization["groups_per_update"])
    expected_updates = int(optimization["max_steps"])
    reward_name = CELL_REWARD[args.cell]
    lora = config["worker"]["actor"]["model"]["lora"]
    checks = {
        "max_steps": config["trainer"]["max_steps"] == expected_updates,
        "val_steps": [0, *config["trainer"]["val_steps"]] == expected_steps,
        "shuffle": config["data"]["shuffle"] is True,
        "data_seed": config["data"]["seed"] == args.seed,
        "rollout_seed": config["worker"]["rollout"]["seed"] == args.seed,
        "rollout_n": config["worker"]["rollout"]["n"] == group_size,
        "rollout_batch_size": config["data"]["rollout_batch_size"] == groups_per_update,
        "actor_global_batch_size": config["worker"]["actor"]["global_batch_size"] == groups_per_update,
        "learning_rate": math.isclose(
            config["worker"]["actor"]["optim"]["lr"], optimization["learning_rate"], rel_tol=0, abs_tol=1e-15
        ),
        "adv_estimator": config["algorithm"]["adv_estimator"] == optimization["advantage_estimator"],
        "kl": config["algorithm"]["use_kl_loss"] is True
        and config["algorithm"]["kl_coef"] == optimization["kl_coefficient"]
        and config["algorithm"]["kl_penalty"] == optimization["kl_penalty"],
        "ppo_epochs": config["worker"]["actor"]["ppo_epochs"] == CELL_PPO_EPOCHS.get(args.cell, optimization["ppo_epochs"]),
        "lora_rank": lora["rank"] == optimization["lora_rank"],
        "vision_lora_excluded": lora["exclude_modules"] == ".*visual.*",
        "reward_function": config["worker"]["reward"]["reward_function_name"] == reward_name,
        "model_path": config["worker"]["actor"]["model"]["model_path"] == str(args.model_path),
    }
    if not all(checks.values()):
        raise ValueError(f"Formal resolved config mismatch: {checks}")

    train_tokens = read_manifest(args.train_manifest)
    monitor_tokens = read_manifest(args.monitor_manifest)
    if len(train_tokens) != expected_groups or set(train_tokens) & set(monitor_tokens):
        raise ValueError("Formal train/monitor manifest mismatch")
    rollout_rows = _jsonl(args.rollouts)
    if any(not math.isfinite(float(row[field])) for row in rollout_rows for field in METRIC_FIELDS):
        raise ValueError("Formal rollout evidence contains non-finite metrics")
    train_rows = [row for row in rollout_rows if row["evidence_phase"] == "train"]
    monitor_rows = [row for row in rollout_rows if row["evidence_phase"] == "train_monitor"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        groups[str(row["token"])].append(row)
    if set(groups) != set(train_tokens) or any(len(rows) != group_size for rows in groups.values()):
        raise ValueError("Formal train rollout coverage mismatch")
    monitor_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in monitor_rows:
        monitor_by_step[int(row["evidence_step"])].append(row)
    if set(monitor_by_step) != set(expected_steps):
        raise ValueError("Formal monitor checkpoint mismatch")
    for step, rows in monitor_by_step.items():
        if len(rows) != len(monitor_tokens) or {str(row["token"]) for row in rows} != set(monitor_tokens):
            raise ValueError(f"Formal monitor coverage mismatch at step {step}")

    monitor_curve = {}
    for step in expected_steps:
        rows = monitor_by_step[step]
        tiers = [_classify_tier(bool(row["parsed_ok"]), {field: float(row[field]) for field in METRIC_FIELDS}) for row in rows]
        monitor_curve[str(step)] = {
            **{field: statistics.fmean(float(row[field]) for row in rows) for field in METRIC_FIELDS},
            "parse_rate": statistics.fmean(bool(row["parsed_ok"]) for row in rows),
            "clip_rate": statistics.fmean(int(row["response_length"]) >= 512 for row in rows),
            "strict_clear_rate": statistics.fmean(tier == "L3" for tier in tiers),
            "tier_rates": {tier: statistics.fmean(value == tier for value in tiers) for tier in TIERS},
        }

    group_stds = []
    mixed_tier = 0
    tier_counts: Counter[str] = Counter()
    for rows in groups.values():
        rewards = [float(row["training_reward"]) for row in rows]
        group_stds.append(statistics.stdev(rewards))
        tiers = [
            _classify_tier(bool(row["parsed_ok"]), {field: float(row[field]) for field in METRIC_FIELDS})
            for row in rows
        ]
        tier_counts.update("invalid" if tier is None else tier for tier in tiers)
        mixed_tier += len(set(tiers)) > 1

    log_rows = _jsonl(args.experiment_log)
    train_log_rows = [row for row in log_rows if int(row["step"]) > 0]
    if {int(row["step"]) for row in train_log_rows} != set(range(1, expected_updates + 1)):
        raise ValueError("Formal experiment log step coverage mismatch")
    metric_values: dict[str, list[float]] = defaultdict(list)
    for row in train_log_rows:
        flat: dict[str, Any] = {}
        _flatten("", row, flat)
        for key, value in flat.items():
            if isinstance(value, (int, float)) and key != "step":
                if not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite formal training metric: {key}")
                metric_values[key].append(float(value))
    missing_actor = [key for key in REQUIRED_ACTOR_METRICS if f"actor/{key}" not in metric_values]
    if missing_actor:
        raise ValueError(f"Missing formal actor metrics: {missing_actor}")

    final_monitor = monitor_curve[str(expected_steps[-1])]
    health = {
        "final_parse_rate": final_monitor["parse_rate"] >= protocol["resolved_config_gate"]["parse_rate_min"],
        "final_clip_rate": final_monitor["clip_rate"] <= protocol["resolved_config_gate"]["clip_rate_max"],
        "nonfinite": True,
    }
    metadata = CELL_METADATA.get(args.cell, protocol["matrix"]["cells"].get(args.cell))
    if metadata is None:
        raise ValueError(f"Formal cell metadata is missing: {args.cell}")
    report = {
        "status": "COMPLETE" if all(health.values()) else "FAILED_HEALTH_GATE",
        "cell": args.cell,
        "seed": args.seed,
        "selector": metadata["selector"],
        "reward": metadata["reward"],
        "processed_groups": len(groups),
        "rollout_queries": len(train_rows),
        "updates": expected_updates,
        "monitor_tokens": len(monitor_tokens),
        "monitor_curve": monitor_curve,
        "train_geometry": {
            "effective_group_rate": statistics.fmean(value > 1e-12 for value in group_stds),
            "exact_zero_group_rate": statistics.fmean(value <= 1e-12 for value in group_stds),
            "low_nonzero_group_rate": statistics.fmean(1e-12 < value < 0.05 for value in group_stds),
            "mixed_tier_group_rate": mixed_tier / len(groups),
            "tier_composition": dict(sorted(tier_counts.items())),
            "training_reward": _summarize([float(row["training_reward"]) for row in train_rows]),
            "parse_rate": statistics.fmean(bool(row["parsed_ok"]) for row in train_rows),
            "clip_rate": statistics.fmean(int(row["response_length"]) >= 512 for row in train_rows),
        },
        "training_metrics": {key: _summarize(values) for key, values in sorted(metric_values.items())},
        "health_gates": health,
        "resolved_config_checks": checks,
        "input_sha256": {
            "m0_protocol": sha256_file(args.m0_protocol),
            "train_manifest": sha256_file(args.train_manifest),
            "monitor_manifest": sha256_file(args.monitor_manifest),
            "experiment_config": sha256_file(args.experiment_config),
            "experiment_log": sha256_file(args.experiment_log),
            "rollouts": sha256_file(args.rollouts),
        },
        "dev_accessed": False,
        "final_accessed": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["status"] != "COMPLETE":
        raise ValueError(f"Formal training health gate failed: {health}")


def _load_scene_metrics(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        token = row["token"]
        if token in output:
            raise ValueError(f"Duplicate scene metric token: {token}")
        output[token] = {
            **row,
            **{field: float(row[field]) for field in METRIC_FIELDS},
            "strict_clear": row["strict_clear"].lower() == "true",
            "tier": row["tier"] or None,
        }
    return output


def _paired_bootstrap(
    rows: list[tuple[dict[str, Any], dict[str, Any]]], metrics: list[str], resamples: int, seed: int
) -> dict[str, dict[str, float]]:
    log_names = sorted({candidate["log_name"] for _, candidate in rows})
    by_log = {log_name: [pair for pair in rows if pair[1]["log_name"] == log_name] for log_name in log_names}
    counts = np.array([len(by_log[log_name]) for log_name in log_names], dtype=np.float64)
    differences = np.array(
        [
            [
                sum(float(candidate[metric]) - float(baseline[metric]) for baseline, candidate in by_log[log_name])
                for metric in metrics
            ]
            for log_name in log_names
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(log_names), size=(resamples, len(log_names)))
    sampled_counts = counts[sampled].sum(axis=1)
    samples = differences[sampled].sum(axis=1) / sampled_counts[:, None]
    point = differences.sum(axis=0) / counts.sum()
    return {
        metric: {
            "point_delta": float(point[index]),
            "ci_lower": float(np.quantile(samples[:, index], 0.025)),
            "ci_upper": float(np.quantile(samples[:, index], 0.975)),
        }
        for index, metric in enumerate(metrics)
    }


def compare_dev(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {args.output}")
    protocol = json.loads(args.m0_protocol.read_text(encoding="utf-8"))
    baseline = _load_scene_metrics(args.baseline)
    candidate = _load_scene_metrics(args.candidate)
    if set(baseline) != set(candidate) or len(candidate) != sum(protocol["tail_evaluation"]["dev_counts"].values()):
        raise ValueError("Paired Dev scene coverage mismatch")
    pairs = [(baseline[token], candidate[token]) for token in sorted(candidate)]
    if any(left["log_name"] != right["log_name"] or left["split"] != right["split"] for left, right in pairs):
        raise ValueError("Paired Dev metadata mismatch")
    metrics = [*METRIC_FIELDS, "strict_clear"]
    resamples = int(protocol["evaluation"]["bootstrap"]["resamples"])
    natural_pairs = [pair for pair in pairs if pair[1]["split"] == "dev_natural"]
    tail_pairs = [pair for pair in pairs if pair[1]["split"] == "dev_tail"]
    natural = _paired_bootstrap(natural_pairs, metrics, resamples, args.bootstrap_seed)
    tail = _paired_bootstrap(tail_pairs, metrics, resamples, args.bootstrap_seed + 1)
    transitions: Counter[str] = Counter()
    for left, right in tail_pairs:
        transitions[f"{left['tier'] or 'invalid'}->{right['tier'] or 'invalid'}"] += 1
    tail_safety = {field: tail[field]["point_delta"] for field in SAFETY_FIELDS}
    gate_spec = protocol["promotion"]["discovery"]
    gates = {
        "tail_point": tail["strict_clear"]["point_delta"] >= gate_spec["tail_primary_delta_min"],
        "tail_ci_upper": tail["strict_clear"]["ci_upper"] > gate_spec["tail_primary_ci_upper_gt"],
        "natural_point": natural["pdms_scaled"]["point_delta"] >= gate_spec["natural_primary_point_delta_min"],
        "natural_ci_lower": natural["pdms_scaled"]["ci_lower"] > gate_spec["natural_primary_ci_lower_severe_harm_floor"],
        "tail_safety_components": all(
            delta >= -gate_spec["safety_component_point_drop_max"] for delta in tail_safety.values()
        ),
    }
    tail_baseline_pdms = [float(left["pdms"]) for left, _ in tail_pairs]
    tail_candidate_pdms = [float(right["pdms"]) for _, right in tail_pairs]
    report = {
        "status": "PROMOTE_TO_CONFIRMATION" if all(gates.values()) else "CLOSED_BY_DISCOVERY_GATE",
        "contrast": args.contrast,
        "seed": args.seed,
        "bootstrap": {"resamples": resamples, "cluster": "log_name", "paired": True, "seed": args.bootstrap_seed},
        "natural": natural,
        "tail": tail,
        "tail_pdms_cvar20": {
            "baseline": _cvar20(tail_baseline_pdms),
            "candidate": _cvar20(tail_candidate_pdms),
            "delta": _cvar20(tail_candidate_pdms) - _cvar20(tail_baseline_pdms),
        },
        "tail_tier_transitions": dict(sorted(transitions.items())),
        "tail_safety_component_deltas": tail_safety,
        "promotion_gates": gates,
        "input_sha256": {
            "baseline": sha256_file(args.baseline),
            "candidate": sha256_file(args.candidate),
            "m0_protocol": sha256_file(args.m0_protocol),
        },
        "dev_accessed": True,
        "final_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze-training")
    analyze.add_argument("--m0-protocol", type=Path, required=True)
    analyze.add_argument("--cell", choices=tuple(CELL_REWARD), required=True)
    analyze.add_argument("--seed", type=int, required=True)
    analyze.add_argument("--model-path", type=Path, required=True)
    analyze.add_argument("--train-manifest", type=Path, required=True)
    analyze.add_argument("--monitor-manifest", type=Path, required=True)
    analyze.add_argument("--experiment-config", type=Path, required=True)
    analyze.add_argument("--experiment-log", type=Path, required=True)
    analyze.add_argument("--rollouts", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(function=analyze_training)

    compare = commands.add_parser("compare-dev")
    compare.add_argument("--m0-protocol", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--contrast", required=True)
    compare.add_argument("--seed", type=int, required=True)
    compare.add_argument("--bootstrap-seed", type=int, default=20260827)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(function=compare_dev)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
