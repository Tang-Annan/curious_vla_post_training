from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from projects.dataset_v3.data_prep import stable_key
from projects.dataset_v3.s1_pipeline import read_manifest, sha256_file


H0_GROUPS = 512
H0_INTENT_QUOTA = {"straight": 341, "left": 111, "right": 60}
H0_SEED = 20260829
CHECKPOINTS_BY_BATCH = {
    4: [0, 26, 51, 77, 102, 128],
    8: [0, 13, 26, 38, 51, 64],
}
SAFETY_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
)
MONITOR_METRICS = (
    "pdms",
    "pdms_scaled",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
)
REQUIRED_ACTOR_METRICS = (
    "pg_loss",
    "entropy_loss",
    "kl_loss",
    "ppo_kl",
    "pg_clipfrac_higher",
    "pg_clipfrac_lower",
    "grad_norm",
    "lr",
)


def select_hparam_tokens(
    random_tokens: list[str], master: dict[str, dict[str, str]], seed: int = H0_SEED
) -> list[str]:
    if len(random_tokens) != 2000 or len(set(random_tokens)) != 2000:
        raise ValueError("H0 requires the frozen 2,000-token Random manifest")
    selected = []
    for intent, quota in H0_INTENT_QUOTA.items():
        candidates = [token for token in random_tokens if master[token]["intent"] == intent]
        candidates.sort(key=lambda token: stable_key(seed, f"h0-{intent}", token))
        if len(candidates) < quota:
            raise ValueError(f"Insufficient H0 {intent} capacity")
        selected.extend(candidates[:quota])
    return sorted(selected, key=lambda token: stable_key(seed, "h0-order", token))


def _table_tokens(table: pa.Table) -> list[str]:
    return [str(answer["token"]) for answer in table.column("answer").to_pylist()]


def prepare(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"H0 output already exists: {args.output_dir}")
    random_tokens = read_manifest(args.random_manifest)
    monitor_tokens = read_manifest(args.monitor_manifest)
    with args.master_index.open(encoding="utf-8-sig", newline="") as handle:
        master = {row["token"]: row for row in csv.DictReader(handle)}
    if set(random_tokens) & set(monitor_tokens):
        raise ValueError("H0 optimizer and monitor tokens overlap")
    if any(master[token]["split"] != "grpo_screen" for token in random_tokens):
        raise ValueError("H0 Random input escaped grpo_screen")
    if any(master[token]["split"] != "train_monitor" for token in monitor_tokens):
        raise ValueError("H0 monitor input escaped train_monitor")

    selected = select_hparam_tokens(random_tokens, master, args.seed)
    random_table = pq.read_table(args.random_parquet)
    random_parquet_tokens = _table_tokens(random_table)
    if random_parquet_tokens != random_tokens:
        raise ValueError("Random parquet order differs from its frozen manifest")
    random_index = {token: index for index, token in enumerate(random_tokens)}
    monitor_table = pq.read_table(args.monitor_parquet)
    if _table_tokens(monitor_table) != monitor_tokens:
        raise ValueError("Monitor parquet order differs from its frozen manifest")

    args.output_dir.mkdir(parents=True)
    hparam_manifest = args.output_dir / "hparam_train_512.txt"
    hparam_parquet = args.output_dir / "hparam_train_512.parquet"
    frozen_monitor_manifest = args.output_dir / "train_monitor_256.txt"
    frozen_monitor_parquet = args.output_dir / "train_monitor_256.parquet"
    hparam_manifest.write_text("".join(f"{token}\n" for token in selected), encoding="utf-8")
    pq.write_table(random_table.take(pa.array([random_index[token] for token in selected])), hparam_parquet)
    shutil.copyfile(args.monitor_manifest, frozen_monitor_manifest)
    shutil.copyfile(args.monitor_parquet, frozen_monitor_parquet)

    protocol = {
        "status": "FROZEN_BEFORE_H0_PILOTS",
        "seed": args.seed,
        "hparam_train_groups": H0_GROUPS,
        "hparam_train_rollout_queries_g4": H0_GROUPS * 4,
        "hparam_intent_quota": H0_INTENT_QUOTA,
        "train_monitor_tokens": 256,
        "train_monitor_rollout_n": 1,
        "train_monitor_optimizer_overlap": 0,
        "lr_candidates": [1e-6, 3e-6],
        "lr_pilot_fixed": {"selector": "Random", "reward": "Raw-PDMS", "group_size": 4, "groups_per_update": 4},
        "monitor_steps_batch4": CHECKPOINTS_BY_BATCH[4],
        "monitor_processed_groups_batch4": [step * 4 for step in CHECKPOINTS_BY_BATCH[4]],
        "monitor_nominal_budget_fraction": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "lr_health_gate": {
            "final_parse_rate_min": 0.99,
            "final_clip_rate_max": 0.01,
            "final_safety_drop_vs_step0_max": 0.01,
            "max_abs_ppo_kl": 0.05,
            "mean_clip_fraction_max": 0.05,
            "max_grad_norm": 5.0,
        },
        "lr_selection_gate": "choose 3e-6 only if admissible and its final PDMS gain exceeds 1e-6 by >=0.005 and mean post-baseline PDMS gain by >=0.002; otherwise choose admissible 1e-6",
        "estimator_trigger": "R0 low-nonzero gate triggered; compare grpo against std_floor_grpo at selected LR",
        "std_floor": 0.05,
        "estimator_selection_gate": "choose std_floor_grpo only if admissible, final PDMS gain vs grpo >=0.003, mean post-baseline gain >=0.001, and no safety metric is lower by >0.005; otherwise retain grpo",
        "batch8_trigger": "only if selected estimator has grad_norm CV >0.5, mean clip fraction >0.02, or >=5% updates with grad_norm >=0.99",
        "fixed_optimization": {
            "ppo_epochs": 1,
            "lora_rank": 8,
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "vision_lora_excluded": True,
            "kl_loss": True,
            "kl_penalty": "low_var_kl",
            "kl_coefficient": 0.01,
            "shuffle": True,
        },
        "random_manifest_sha256": sha256_file(args.random_manifest),
        "random_parquet_sha256": sha256_file(args.random_parquet),
        "hparam_manifest_sha256": sha256_file(hparam_manifest),
        "hparam_parquet_sha256": sha256_file(hparam_parquet),
        "monitor_manifest_sha256": sha256_file(frozen_monitor_manifest),
        "monitor_parquet_sha256": sha256_file(frozen_monitor_parquet),
        "master_index_sha256": sha256_file(args.master_index),
        "dev_accessed": False,
        "final_accessed": False,
    }
    (args.output_dir / "h0_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}/{key}" if prefix else key, child, output)
    else:
        output[prefix] = value


def _summarize(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "cv_abs": std / abs(mean) if abs(mean) > 1e-12 else 0.0,
    }


def analyze(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    expected_steps = CHECKPOINTS_BY_BATCH[args.groups_per_update]
    expected_final_step = H0_GROUPS // args.groups_per_update
    checks = {
        "max_steps": config["trainer"]["max_steps"] == expected_final_step,
        "val_steps": [0, *config["trainer"]["val_steps"]] == expected_steps,
        "shuffle": config["data"]["shuffle"] is True,
        "data_seed": config["data"]["seed"] == H0_SEED,
        "rollout_seed": config["worker"]["rollout"]["seed"] == H0_SEED,
        "rollout_n": config["worker"]["rollout"]["n"] == 4,
        "rollout_batch_size": config["data"]["rollout_batch_size"] == args.groups_per_update,
        "actor_global_batch_size": config["worker"]["actor"]["global_batch_size"] == args.groups_per_update,
        "learning_rate": math.isclose(config["worker"]["actor"]["optim"]["lr"], args.lr, rel_tol=0, abs_tol=1e-15),
        "adv_estimator": config["algorithm"]["adv_estimator"] == args.estimator,
        "std_floor": config["algorithm"]["std_floor"] == 0.05,
        "kl": config["algorithm"]["use_kl_loss"] is True and config["algorithm"]["kl_coef"] == 0.01,
        "ppo_epochs": config["worker"]["actor"]["ppo_epochs"] == 1,
        "lora_rank": config["worker"]["actor"]["model"]["lora"]["rank"] == 8,
        "reward_function": config["worker"]["reward"]["reward_function_name"] == "compute_score_raw_pdms",
    }
    if not all(checks.values()):
        raise ValueError(f"H0 resolved config mismatch: {checks}")

    hparam_tokens = read_manifest(args.hparam_manifest)
    monitor_tokens = read_manifest(args.monitor_manifest)
    rollout_rows = [json.loads(line) for line in args.rollouts.read_text(encoding="utf-8").splitlines() if line]
    if any(not math.isfinite(float(row[field])) for row in rollout_rows for field in MONITOR_METRICS):
        raise ValueError("H0 rollout evidence contains non-finite metrics")
    train_rows = [row for row in rollout_rows if row["evidence_phase"] == "train"]
    monitor_rows = [row for row in rollout_rows if row["evidence_phase"] == "train_monitor"]
    train_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        train_groups[str(row["token"])].append(row)
    if set(train_groups) != set(hparam_tokens) or any(len(rows) != 4 for rows in train_groups.values()):
        raise ValueError("H0 train rollout coverage mismatch")
    monitor_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in monitor_rows:
        monitor_by_step[int(row["evidence_step"])].append(row)
    if set(monitor_by_step) != set(expected_steps):
        raise ValueError(f"H0 monitor checkpoints mismatch: {sorted(monitor_by_step)}")
    for step, rows in monitor_by_step.items():
        if len(rows) != 256 or {str(row["token"]) for row in rows} != set(monitor_tokens):
            raise ValueError(f"H0 monitor coverage mismatch at step {step}")

    checkpoint_summary = {}
    for step in expected_steps:
        rows = monitor_by_step[step]
        checkpoint_summary[str(step)] = {
            **{field: statistics.fmean(float(row[field]) for row in rows) for field in MONITOR_METRICS},
            "parse_rate": statistics.fmean(bool(row["parsed_ok"]) for row in rows),
            "clip_rate": statistics.fmean(int(row["response_length"]) >= 512 for row in rows),
        }

    exact_zero = 0
    low_nonzero = 0
    for rows in train_groups.values():
        std = statistics.stdev(float(row["training_reward"]) for row in rows)
        exact_zero += std <= 1e-12
        low_nonzero += 1e-12 < std < 0.05

    log_rows = [json.loads(line) for line in args.experiment_log.read_text(encoding="utf-8").splitlines() if line]
    train_log_rows = [row for row in log_rows if int(row["step"]) > 0]
    if {int(row["step"]) for row in train_log_rows} != set(range(1, expected_final_step + 1)):
        raise ValueError("H0 experiment log step coverage mismatch")
    metric_values: dict[str, list[float]] = defaultdict(list)
    for row in train_log_rows:
        flat: dict[str, Any] = {}
        _flatten("", row, flat)
        for key, value in flat.items():
            if isinstance(value, (int, float)) and key != "step":
                if not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite training metric: {key}")
                metric_values[key].append(float(value))
    missing_actor = [key for key in REQUIRED_ACTOR_METRICS if f"actor/{key}" not in metric_values]
    if missing_actor:
        raise ValueError(f"Missing H0 actor metrics: {missing_actor}")

    report = {
        "status": "COMPLETE",
        "lr": args.lr,
        "estimator": args.estimator,
        "groups_per_update": args.groups_per_update,
        "processed_groups": H0_GROUPS,
        "rollout_queries": H0_GROUPS * 4,
        "monitor_tokens": len(monitor_tokens),
        "monitor_checkpoints": checkpoint_summary,
        "train_geometry": {
            "groups": len(train_groups),
            "exact_zero_rate": exact_zero / len(train_groups),
            "low_nonzero_rate": low_nonzero / len(train_groups),
        },
        "training_metrics": {key: _summarize(values) for key, values in sorted(metric_values.items())},
        "resolved_config_checks": checks,
        "protocol_sha256": sha256_file(args.protocol),
        "hparam_manifest_sha256": sha256_file(args.hparam_manifest),
        "monitor_manifest_sha256": sha256_file(args.monitor_manifest),
        "experiment_config_sha256": sha256_file(args.experiment_config),
        "experiment_log_sha256": sha256_file(args.experiment_log),
        "rollouts_sha256": sha256_file(args.rollouts),
        "dev_accessed": False,
        "final_accessed": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _health(report: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    steps = sorted((int(step) for step in report["monitor_checkpoints"]), key=int)
    baseline = report["monitor_checkpoints"][str(steps[0])]
    final = report["monitor_checkpoints"][str(steps[-1])]
    metrics = report["training_metrics"]
    mean_clip = metrics["actor/pg_clipfrac_higher"]["mean"] + metrics["actor/pg_clipfrac_lower"]["mean"]
    gates = {
        "parse": final["parse_rate"] >= 0.99,
        "clipping": final["clip_rate"] <= 0.01,
        "safety": all(final[key] >= baseline[key] - 0.01 for key in SAFETY_METRICS),
        "ppo_kl": max(abs(metrics["actor/ppo_kl"]["min"]), abs(metrics["actor/ppo_kl"]["max"])) <= 0.05,
        "clip_fraction": mean_clip <= 0.05,
        "grad_norm": metrics["actor/grad_norm"]["max"] <= 5.0,
    }
    return all(gates.values()), gates


def _pdms_gains(report: dict[str, Any]) -> tuple[float, float]:
    steps = sorted(int(step) for step in report["monitor_checkpoints"])
    values = [report["monitor_checkpoints"][str(step)]["pdms"] for step in steps]
    return values[-1] - values[0], statistics.fmean(value - values[0] for value in values[1:])


def choose_lr(args: argparse.Namespace) -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    by_lr = {float(report["lr"]): report for report in reports}
    if set(by_lr) != {1e-6, 3e-6}:
        raise ValueError("LR decision requires exactly the 1e-6 and 3e-6 reports")
    health = {str(lr): _health(report) for lr, report in by_lr.items()}
    admissible = [lr for lr in (1e-6, 3e-6) if health[str(lr)][0]]
    if not admissible:
        raise ValueError("Both H0 LR candidates failed the preregistered health gate")
    selected = admissible[0]
    reason = "only_admissible_candidate"
    if 1e-6 in admissible and 3e-6 in admissible:
        low_final, low_auc = _pdms_gains(by_lr[1e-6])
        high_final, high_auc = _pdms_gains(by_lr[3e-6])
        if high_final - low_final >= 0.005 and high_auc - low_auc >= 0.002:
            selected, reason = 3e-6, "higher_lr_passed_pre_registered_pdms_margin"
        else:
            selected, reason = 1e-6, "conservative_lr_retained_below_margin"
    output = {
        "status": "LR_FROZEN",
        "selected_lr": selected,
        "reason": reason,
        "health": {key: {"admissible": value[0], "gates": value[1]} for key, value in health.items()},
        "pdms_gains": {str(lr): {"final": _pdms_gains(report)[0], "post_baseline_mean": _pdms_gains(report)[1]} for lr, report in by_lr.items()},
        "report_sha256": {str(path): sha256_file(path) for path in args.reports},
        "dev_accessed": False,
        "final_accessed": False,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def choose_estimator(args: argparse.Namespace) -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    by_estimator = {report["estimator"]: report for report in reports}
    if set(by_estimator) != {"grpo", "std_floor_grpo"}:
        raise ValueError("Estimator decision requires grpo and std_floor_grpo reports")
    standard, floor = by_estimator["grpo"], by_estimator["std_floor_grpo"]
    standard_health, standard_gates = _health(standard)
    floor_health, floor_gates = _health(floor)
    if not standard_health and not floor_health:
        raise ValueError("Both H0 estimators failed the preregistered health gate")
    standard_final, standard_auc = _pdms_gains(standard)
    floor_final, floor_auc = _pdms_gains(floor)
    standard_last = standard["monitor_checkpoints"][str(max(map(int, standard["monitor_checkpoints"])))]
    floor_last = floor["monitor_checkpoints"][str(max(map(int, floor["monitor_checkpoints"])))]
    safety_ok = all(floor_last[key] >= standard_last[key] - 0.005 for key in SAFETY_METRICS)
    choose_floor = (
        floor_health
        and floor_final - standard_final >= 0.003
        and floor_auc - standard_auc >= 0.001
        and safety_ok
    )
    selected = "std_floor_grpo" if choose_floor else "grpo"
    output = {
        "status": "ESTIMATOR_FROZEN",
        "selected_estimator": selected,
        "std_floor": 0.05 if choose_floor else None,
        "reason": "std_floor_passed_pre_registered_monitor_margin" if choose_floor else "standard_grpo_retained",
        "health": {
            "grpo": {"admissible": standard_health, "gates": standard_gates},
            "std_floor_grpo": {"admissible": floor_health, "gates": floor_gates},
        },
        "pdms_gains": {
            "grpo": {"final": standard_final, "post_baseline_mean": standard_auc},
            "std_floor_grpo": {"final": floor_final, "post_baseline_mean": floor_auc},
        },
        "floor_safety_vs_grpo_gate": safety_ok,
        "report_sha256": {str(path): sha256_file(path) for path in args.reports},
        "dev_accessed": False,
        "final_accessed": False,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--random-manifest", type=Path, required=True)
    prepare_parser.add_argument("--random-parquet", type=Path, required=True)
    prepare_parser.add_argument("--monitor-manifest", type=Path, required=True)
    prepare_parser.add_argument("--monitor-parquet", type=Path, required=True)
    prepare_parser.add_argument("--master-index", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=H0_SEED)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.set_defaults(func=prepare)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--protocol", type=Path, required=True)
    analyze_parser.add_argument("--hparam-manifest", type=Path, required=True)
    analyze_parser.add_argument("--monitor-manifest", type=Path, required=True)
    analyze_parser.add_argument("--experiment-config", type=Path, required=True)
    analyze_parser.add_argument("--experiment-log", type=Path, required=True)
    analyze_parser.add_argument("--rollouts", type=Path, required=True)
    analyze_parser.add_argument("--lr", type=float, required=True)
    analyze_parser.add_argument("--estimator", choices=("grpo", "std_floor_grpo"), required=True)
    analyze_parser.add_argument("--groups-per-update", type=int, choices=(4, 8), required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.set_defaults(func=analyze)

    for command, function in (("choose-lr", choose_lr), ("choose-estimator", choose_estimator)):
        decision_parser = subparsers.add_parser(command)
        decision_parser.add_argument("--reports", type=Path, nargs=2, required=True)
        decision_parser.add_argument("--output", type=Path, required=True)
        decision_parser.set_defaults(func=function)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
