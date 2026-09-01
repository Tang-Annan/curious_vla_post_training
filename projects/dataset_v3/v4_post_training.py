from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from projects.dataset_v3.eval_pipeline import METRIC_FIELDS, TIERS, _classify_tier
from projects.dataset_v3.formal_pipeline import _paired_bootstrap
from projects.dataset_v3.h0_pipeline import _flatten
from projects.dataset_v3.s1_pipeline import read_manifest, sha256_file
from projects.dataset_v3.v4_experiment_closure import revised_dev_tiers, truthy


EXPECTED_DEV_SCENES = 416
EXPECTED_MONITOR_STEPS = (0, 100, 200, 300, 400, 500)
CANDIDATE_SAFETY_FIELDS = (
    "candidate_no_at_fault_collisions",
    "candidate_drivable_area_compliance",
    "candidate_driving_direction_compliance",
    "candidate_traffic_light_compliance",
    "time_to_collision_within_bound",
)
COMPARISON_METRICS = (
    *METRIC_FIELDS,
    *(field for field in CANDIDATE_SAFETY_FIELDS if field not in METRIC_FIELDS),
    "strict_clear",
)
CONTROL_PDMS_MARGIN = -0.01
SAFETY_DROP_MARGIN = -0.005


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenes": len(rows),
        "metrics": {
            metric: statistics.fmean(float(row[metric]) for row in rows)
            for metric in COMPARISON_METRICS
        },
    }


def slice_name_membership(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "all_dev": True,
        "risk": truthy(row["eval_tier1"]),
        "control": not truthy(row["eval_tier1"]),
        "response_complexity": truthy(row["response_complexity"]),
        "natural": row["split"] == "dev_natural",
        "tail": row["split"] == "dev_tail",
    }


def classify_contrast(
    deltas: dict[str, dict[str, dict[str, float]]], *, require_risk_safety_gain: bool
) -> dict[str, Any]:
    risk = deltas["risk"]
    control = deltas["control"]
    all_dev = deltas["all_dev"]
    gates = {
        "risk_pdms_scaled_positive": risk["pdms_scaled"]["point_delta"] > 0.0,
        "risk_strict_clear_positive": risk["strict_clear"]["point_delta"] > 0.0,
        "all_dev_pdms_scaled_nonnegative": all_dev["pdms_scaled"]["point_delta"] >= 0.0,
        "control_pdms_scaled_noninferior": control["pdms_scaled"]["point_delta"] >= CONTROL_PDMS_MARGIN,
        "risk_safety_noninferior": all(
            risk[field]["point_delta"] >= SAFETY_DROP_MARGIN for field in CANDIDATE_SAFETY_FIELDS
        ),
        "control_safety_noninferior": all(
            control[field]["point_delta"] >= SAFETY_DROP_MARGIN for field in CANDIDATE_SAFETY_FIELDS
        ),
    }
    if require_risk_safety_gain:
        gates["risk_any_safety_positive"] = any(
            risk[field]["point_delta"] > 0.0 for field in CANDIDATE_SAFETY_FIELDS
        )
    direction_pass = all(gates.values())
    ci_supported = (
        risk["pdms_scaled"]["ci_lower"] > 0.0
        and risk["strict_clear"]["ci_lower"] > 0.0
    )
    if direction_pass and ci_supported:
        status = "STATISTICALLY_SUPPORTED_IMPROVEMENT"
    elif direction_pass:
        status = "DIRECTIONAL_EXPLORATORY_PASS"
    else:
        status = "NO_IMPROVEMENT_GATE"
    return {
        "status": status,
        "gates": gates,
        "all_direction_gates_pass": direction_pass,
        "risk_primary_both_ci_lower_gt_zero": ci_supported,
    }


def parse_run_env(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if line)


def load_dev_run(run_dir: Path, model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (run_dir / "COMPLETE").exists() or (run_dir / "exit_code").read_text().strip() != "0":
        raise ValueError(f"Dev run is not COMPLETE/0: {run_dir}")
    if (run_dir / "source_status.txt").read_text(encoding="utf-8"):
        raise ValueError(f"Dev source is dirty: {run_dir}")
    scene_rows = read_csv(run_dir / "results" / "scene_metrics.csv")
    rollout_rows = read_jsonl(run_dir / "rollouts.jsonl")
    if len(scene_rows) != EXPECTED_DEV_SCENES or len(rollout_rows) != EXPECTED_DEV_SCENES:
        raise ValueError(f"Dev run does not contain 416 rows: {run_dir}")
    scenes = {row["token"]: row for row in scene_rows}
    rollouts = {str(row["token"]): row for row in rollout_rows}
    if len(scenes) != EXPECTED_DEV_SCENES or len(rollouts) != EXPECTED_DEV_SCENES or set(scenes) != set(rollouts):
        raise ValueError(f"Dev token coverage mismatch: {run_dir}")
    merged = []
    for token in sorted(scenes):
        scene = scenes[token]
        rollout = rollouts[token]
        candidate_metrics = {field: float(rollout[field]) for field in CANDIDATE_SAFETY_FIELDS}
        if any(not math.isfinite(value) for value in candidate_metrics.values()):
            raise ValueError(f"Non-finite candidate safety metric for {model}/{token}")
        merged.append(
            {
                **scene,
                **{field: float(scene[field]) for field in METRIC_FIELDS},
                **candidate_metrics,
                "strict_clear": truthy(scene["strict_clear"]),
                "model": model,
            }
        )
    metadata = {
        "run_dir": str(run_dir),
        "source_commit": (run_dir / "source_commit.txt").read_text().strip(),
        "run_env": parse_run_env(run_dir / "run.env"),
        "input_sha256": (run_dir / "input_sha256.txt").read_text(),
        "model_sha256": (run_dir / "model_sha256.txt").read_text(),
        "result_sha256": (run_dir / "result_sha256.txt").read_text(),
    }
    return merged, metadata


def compare_dev(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    dev_scene_rows = read_csv(args.dev_scene_labels)
    original_tier_rows = read_csv(args.original_dev_tiers)
    if len(dev_scene_rows) != EXPECTED_DEV_SCENES or len(original_tier_rows) != EXPECTED_DEV_SCENES:
        raise ValueError("V4 Dev labels must each contain 416 rows")
    revised_labels = revised_dev_tiers(dev_scene_rows, original_tier_rows)
    labels = {row["token"]: row for row in revised_labels}

    model_rows: dict[str, list[dict[str, Any]]] = {}
    run_metadata: dict[str, dict[str, Any]] = {}
    for model, run_dir in (("RR", args.rr_run), ("GPU-A", args.gpu_a_run), ("GPU-B", args.gpu_b_run)):
        rows, metadata = load_dev_run(run_dir, model)
        if {row["token"] for row in rows} != set(labels):
            raise ValueError(f"{model} Dev tokens differ from frozen V4 labels")
        model_rows[model] = [{**row, **labels[row["token"]]} for row in rows]
        run_metadata[model] = metadata

    sources = {metadata["source_commit"] for metadata in run_metadata.values()}
    input_hashes = {metadata["input_sha256"] for metadata in run_metadata.values()}
    model_hashes = {metadata["model_sha256"] for metadata in run_metadata.values()}
    fixed_env_keys = ("evaluation_seed", "rollout_n", "temperature", "top_p", "max_response_length", "model")
    fixed_envs = {
        tuple((key, metadata["run_env"].get(key)) for key in fixed_env_keys)
        for metadata in run_metadata.values()
    }
    if len(sources) != 1 or len(input_hashes) != 1 or len(model_hashes) != 1 or len(fixed_envs) != 1:
        raise ValueError("RR/GPU-A/GPU-B Dev protocols are not identical")

    slice_metrics_rows = []
    model_summaries = {}
    for model, rows in model_rows.items():
        summaries = {}
        for slice_name in slice_name_membership(rows[0]):
            selected = [row for row in rows if slice_name_membership(row)[slice_name]]
            summary = summarize_rows(selected)
            summaries[slice_name] = summary
            slice_metrics_rows.append(
                {"model": model, "slice": slice_name, "scenes": summary["scenes"], **summary["metrics"]}
            )
        model_summaries[model] = summaries

    contrasts = {}
    delta_rows = []
    for contrast, baseline_name, candidate_name, require_safety_gain, seed_offset in (
        ("GPU-A_minus_RR", "RR", "GPU-A", False, 0),
        ("GPU-B_minus_GPU-A", "GPU-A", "GPU-B", True, 100),
    ):
        baseline = {row["token"]: row for row in model_rows[baseline_name]}
        candidate = {row["token"]: row for row in model_rows[candidate_name]}
        slice_deltas = {}
        for slice_index, slice_name in enumerate(slice_name_membership(next(iter(candidate.values())))):
            pairs = [
                (baseline[token], candidate[token])
                for token in sorted(candidate)
                if slice_name_membership(candidate[token])[slice_name]
            ]
            deltas = _paired_bootstrap(
                pairs,
                list(COMPARISON_METRICS),
                args.bootstrap_resamples,
                args.seed + seed_offset + slice_index,
            )
            slice_deltas[slice_name] = deltas
            for metric, values in deltas.items():
                delta_rows.append({"contrast": contrast, "slice": slice_name, "metric": metric, **values})
        contrasts[contrast] = {
            **classify_contrast(slice_deltas, require_risk_safety_gain=require_safety_gain),
            "baseline": baseline_name,
            "candidate": candidate_name,
            "slices": slice_deltas,
        }

    report = {
        "status": "V4_POST_TRAINING_SCIENCE_COMPLETE",
        "models": model_summaries,
        "contrasts": contrasts,
        "slice_counts": {
            name: sum(slice_name_membership(row)[name] for row in model_rows["RR"])
            for name in slice_name_membership(model_rows["RR"][0])
        },
        "gate_thresholds": {
            "risk_pdms_scaled_point_min_exclusive": 0.0,
            "risk_strict_clear_point_min_exclusive": 0.0,
            "all_dev_pdms_scaled_point_min": 0.0,
            "control_pdms_scaled_point_min": CONTROL_PDMS_MARGIN,
            "safety_component_point_min": SAFETY_DROP_MARGIN,
            "statistical_support": "risk PDMS-scaled and StrictClear paired 95% CI lower bounds both > 0",
        },
        "bootstrap": {"resamples": args.bootstrap_resamples, "cluster": "log_name", "paired": True, "seed": args.seed},
        "protocol": {
            "source_commit": next(iter(sources)),
            "fixed_run_env": dict(next(iter(fixed_envs))),
            "run_metadata": run_metadata,
        },
        "limitations": {
            "training_seeds": 1,
            "dev_previously_accessed": True,
            "final_accessed": False,
            "claim_boundary": "matched exploratory Dev evidence; no training-seed stability or independent-final claim",
        },
        "input_sha256": {
            "dev_scene_labels": sha256_file(args.dev_scene_labels),
            "original_dev_tiers": sha256_file(args.original_dev_tiers),
        },
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "model_slice_metrics.csv", slice_metrics_rows)
    write_csv(args.output_dir / "paired_deltas.csv", delta_rows)
    (args.output_dir / "v4_post_training_science_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def recover_monitor(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.m0_protocol.read_text(encoding="utf-8"))
    train_tokens = set(read_manifest(args.train_manifest))
    monitor_tokens = set(read_manifest(args.monitor_manifest))
    original_rows = read_jsonl(args.original_rollouts)
    recovery_rows = read_jsonl(args.recovery_rollouts)

    train_rows = [row for row in original_rows if row["evidence_phase"] == "train"]
    train_by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        train_by_token[str(row["token"])].append(row)
    group_size = int(protocol["optimization"]["group_size"])
    train_complete = set(train_by_token) == train_tokens and all(
        len(rows) == group_size for rows in train_by_token.values()
    )

    original_monitor = [row for row in original_rows if row["evidence_phase"] == "train_monitor"]
    original_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in original_monitor:
        original_by_step[int(row["evidence_step"])].append(row)
    missing_by_step = {
        str(step): sorted(monitor_tokens - {str(row["token"]) for row in original_by_step[step]})
        for step in EXPECTED_MONITOR_STEPS
    }
    missing_sets = {tuple(tokens) for tokens in missing_by_step.values()}
    original_gap_consistent = (
        set(original_by_step) == set(EXPECTED_MONITOR_STEPS)
        and len(missing_sets) == 1
        and all(len(original_by_step[step]) + len(missing_by_step[str(step)]) == len(monitor_tokens) for step in EXPECTED_MONITOR_STEPS)
    )

    error_lines = [
        line
        for line in args.original_run_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if "min_distance_to_actors must be finite and non-negative: inf" in line
    ]
    expected_error_count = len(next(iter(missing_sets))) * len(EXPECTED_MONITOR_STEPS) if missing_sets else 0

    if len(recovery_rows) != len(monitor_tokens) or {str(row["token"]) for row in recovery_rows} != monitor_tokens:
        raise ValueError("Recovered final Monitor does not exactly cover 256 tokens")
    if any(row["evidence_phase"] != "train_monitor_recovery" or int(row["evidence_step"]) != 500 for row in recovery_rows):
        raise ValueError("Recovered Monitor evidence phase/step mismatch")
    if any(not math.isfinite(float(row[field])) for row in recovery_rows for field in METRIC_FIELDS):
        raise ValueError("Recovered Monitor contains non-finite evaluator metrics")
    tiers = [
        _classify_tier(bool(row["parsed_ok"]), {field: float(row[field]) for field in METRIC_FIELDS})
        for row in recovery_rows
    ]
    recovery_summary = {
        **{field: statistics.fmean(float(row[field]) for row in recovery_rows) for field in METRIC_FIELDS},
        "parse_rate": statistics.fmean(bool(row["parsed_ok"]) for row in recovery_rows),
        "clip_rate": statistics.fmean(int(row["response_length"]) >= 512 for row in recovery_rows),
        "strict_clear_rate": statistics.fmean(tier == "L3" for tier in tiers),
        "tier_rates": {tier: statistics.fmean(value == tier for value in tiers) for tier in TIERS},
    }
    recovery_health = {
        "parse_rate": recovery_summary["parse_rate"] >= protocol["resolved_config_gate"]["parse_rate_min"],
        "clip_rate": recovery_summary["clip_rate"] <= protocol["resolved_config_gate"]["clip_rate_max"],
        "nonfinite": True,
    }

    log_rows = read_jsonl(args.experiment_log)
    step_rows = [row for row in log_rows if int(row["step"]) > 0]
    step_complete = {int(row["step"]) for row in step_rows} == set(range(1, int(protocol["optimization"]["max_steps"]) + 1))
    log_finite = True
    for row in step_rows:
        flat: dict[str, Any] = {}
        _flatten("", row, flat)
        log_finite &= all(math.isfinite(float(value)) for key, value in flat.items() if key != "step" and isinstance(value, (int, float)))

    checks = {
        "training_rollouts_complete": train_complete,
        "training_steps_complete": step_complete,
        "training_log_finite": log_finite,
        "original_monitor_gap_same_tokens_all_steps": original_gap_consistent,
        "original_gap_matches_reward_errors": len(error_lines) == expected_error_count,
        "final_monitor_recovery_exact_256": True,
        "final_monitor_health": all(recovery_health.values()),
    }
    status = "CHECKPOINT_USABLE_FOR_EXPLORATORY_DEV" if all(checks.values()) else "RECOVERY_GATE_FAILED"
    report = {
        "status": status,
        "original_formal_run_status": "FAILED",
        "original_formal_exit_code": 1,
        "historical_six_point_monitor_curve_recovered": False,
        "training": {
            "updates": len(step_rows),
            "groups": len(train_by_token),
            "rollouts": len(train_rows),
        },
        "original_monitor": {
            "rows_by_step": {str(step): len(original_by_step[step]) for step in EXPECTED_MONITOR_STEPS},
            "missing_tokens_by_step": missing_by_step,
            "reward_error_count": len(error_lines),
            "root_cause": "JSON null no-actor distance normalized to +inf, then rejected by safety reward",
        },
        "recovered_final_monitor": recovery_summary,
        "recovered_final_monitor_health": recovery_health,
        "checks": checks,
        "claim_boundary": "final checkpoint may enter matched exploratory Dev; original formal FAILED marker and incomplete historical Monitor curve remain",
        "dev_accessed": False,
        "final_accessed": False,
        "input_sha256": {
            "m0_protocol": sha256_file(args.m0_protocol),
            "train_manifest": sha256_file(args.train_manifest),
            "monitor_manifest": sha256_file(args.monitor_manifest),
            "original_rollouts": sha256_file(args.original_rollouts),
            "recovery_rollouts": sha256_file(args.recovery_rollouts),
            "experiment_log": sha256_file(args.experiment_log),
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "CHECKPOINT_USABLE_FOR_EXPLORATORY_DEV":
        raise ValueError(f"GPU-B recovery gate failed: {checks}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    monitor = commands.add_parser("recover-monitor")
    monitor.add_argument("--m0-protocol", type=Path, required=True)
    monitor.add_argument("--train-manifest", type=Path, required=True)
    monitor.add_argument("--monitor-manifest", type=Path, required=True)
    monitor.add_argument("--original-rollouts", type=Path, required=True)
    monitor.add_argument("--recovery-rollouts", type=Path, required=True)
    monitor.add_argument("--original-run-log", type=Path, required=True)
    monitor.add_argument("--experiment-log", type=Path, required=True)
    monitor.add_argument("--output", type=Path, required=True)
    monitor.set_defaults(function=recover_monitor)

    compare = commands.add_parser("compare-dev")
    compare.add_argument("--rr-run", type=Path, required=True)
    compare.add_argument("--gpu-a-run", type=Path, required=True)
    compare.add_argument("--gpu-b-run", type=Path, required=True)
    compare.add_argument("--dev-scene-labels", type=Path, required=True)
    compare.add_argument("--original-dev-tiers", type=Path, required=True)
    compare.add_argument("--bootstrap-resamples", type=int, default=20000)
    compare.add_argument("--seed", type=int, default=20260901)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.set_defaults(function=compare_dev)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
