from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from projects.dataset_v3.data_prep import stable_key
from projects.dataset_v3.inventory import sha256_file


FAMILY_QUOTAS = {"proximity": 500, "construction": 500, "signal": 1000}
INTENT_QUOTAS = {"straight": 1333, "left": 434, "right": 233}
LOG_CAP_CANDIDATES = (2, 4, 6, 8)
EXPECTED_MODELS = {"SFT", "RR", "TR", "TC", "TC-PPO2"}
METRIC_FIELDS = (
    "no_at_fault_collisions",
    "time_to_collision_within_bound",
    "pdms_scaled",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: Any) -> bool:
    return str(value).lower() in {"1", "true"}


def exclusive_family(row: dict[str, str]) -> str:
    if truthy(row["visible_critical_proximity"]):
        return "proximity"
    if truthy(row["front_construction_response"]):
        return "construction"
    if truthy(row["current_signal_hard_response"]):
        return "signal"
    return "control"


def _deterministic_cost(seed: int, token: str) -> float:
    return int(stable_key(seed, "v4-risk-balanced-2000", token)[:13], 16) / float(16**13)


def _solve_selection(
    rows: list[dict[str, str]],
    *,
    family_quotas: dict[str, int],
    intent_quotas: dict[str, int],
    log_cap: int,
    seed: int,
) -> list[dict[str, str]] | None:
    ordered = sorted(rows, key=lambda row: stable_key(seed, "v4-milp-order", row["token"]))
    constraint_rows: list[tuple[list[int], float, float]] = []
    for family, quota in family_quotas.items():
        constraint_rows.append(
            ([index for index, row in enumerate(ordered) if row["exclusive_family"] == family], quota, quota)
        )
    for intent, quota in intent_quotas.items():
        constraint_rows.append(
            ([index for index, row in enumerate(ordered) if row["intent"] == intent], quota, quota)
        )
    by_log: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        by_log[row["log_name"]].append(index)
    for indices in by_log.values():
        constraint_rows.append((indices, -math.inf, log_cap))

    matrix_row = []
    matrix_column = []
    matrix_data = []
    lower = []
    upper = []
    for row_index, (indices, minimum, maximum) in enumerate(constraint_rows):
        matrix_row.extend([row_index] * len(indices))
        matrix_column.extend(indices)
        matrix_data.extend([1.0] * len(indices))
        lower.append(minimum)
        upper.append(maximum)
    matrix = coo_matrix(
        (matrix_data, (matrix_row, matrix_column)),
        shape=(len(constraint_rows), len(ordered)),
    ).tocsr()
    result = milp(
        c=np.asarray([_deterministic_cost(seed, row["token"]) for row in ordered]),
        integrality=np.ones(len(ordered)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 60, "mip_rel_gap": 0.0},
    )
    if not result.success:
        return None
    selected = [row for row, value in zip(ordered, result.x) if value > 0.5]
    expected = sum(family_quotas.values())
    if len(selected) != expected:
        raise ValueError(f"MILP returned {len(selected)} rows, expected {expected}")
    return sorted(selected, key=lambda row: stable_key(seed, "v4-risk-balanced-output", row["token"]))


def build_train_selection(
    scene_rows: list[dict[str, str]],
    tier_rows: list[dict[str, str]],
    *,
    family_quotas: dict[str, int] = FAMILY_QUOTAS,
    intent_quotas: dict[str, int] = INTENT_QUOTAS,
    cap_candidates: tuple[int, ...] = LOG_CAP_CANDIDATES,
    seed: int = 20260831,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    scene = {row["token"]: row for row in scene_rows}
    tiers = {row["token"]: row for row in tier_rows}
    if len(scene) != len(scene_rows) or len(tiers) != len(tier_rows) or set(scene) != set(tiers):
        raise ValueError("Train scene and tier labels must have identical unique token coverage")
    candidates = []
    for token, tier in tiers.items():
        if not truthy(tier["train_tier1"]):
            continue
        family = exclusive_family(tier)
        if family == "control":
            raise ValueError(f"Train Tier-1 token has no exclusive family: {token}")
        source = scene[token]
        candidates.append(
            {
                "token": token,
                "log_name": source["log_name"],
                "intent": source["intent"],
                "exclusive_family": family,
            }
        )

    by_log = Counter(row["log_name"] for row in candidates)
    cap_upper_bounds = {
        str(cap): sum(min(cap, count) for count in by_log.values()) for cap in cap_candidates
    }
    selected = None
    selected_cap = None
    feasibility = {}
    for cap in cap_candidates:
        if cap_upper_bounds[str(cap)] < sum(family_quotas.values()):
            feasibility[str(cap)] = "UPPER_BOUND_INSUFFICIENT"
            continue
        solution = _solve_selection(
            candidates,
            family_quotas=family_quotas,
            intent_quotas=intent_quotas,
            log_cap=cap,
            seed=seed,
        )
        feasibility[str(cap)] = "EXACT_FEASIBLE" if solution is not None else "EXACT_INFEASIBLE"
        if solution is not None:
            selected = solution
            selected_cap = cap
            break
    if selected is None or selected_cap is None:
        raise ValueError("No exact V4 2,000-token selection satisfies family/intent/log constraints")

    selected_tokens = {row["token"] for row in selected}
    labels = [{**row, "selected_provisional_2000": int(row["token"] in selected_tokens)} for row in candidates]
    selected_logs = Counter(row["log_name"] for row in selected)
    report = {
        "status": "V4_PROVISIONAL_2000_EXACT",
        "exclusive_priority": ["proximity", "construction", "signal"],
        "candidate_scenes": len(candidates),
        "candidate_unique_logs": len(by_log),
        "candidate_family_counts": dict(sorted(Counter(row["exclusive_family"] for row in candidates).items())),
        "candidate_family_intent_counts": {
            family: dict(
                sorted(Counter(row["intent"] for row in candidates if row["exclusive_family"] == family).items())
            )
            for family in family_quotas
        },
        "log_cap_upper_bounds": cap_upper_bounds,
        "log_cap_feasibility": feasibility,
        "selected_log_cap": selected_cap,
        "selected_scenes": len(selected),
        "selected_unique_logs": len(selected_logs),
        "selected_max_per_log": max(selected_logs.values()),
        "selected_family_counts": dict(sorted(Counter(row["exclusive_family"] for row in selected).items())),
        "selected_intent_counts": dict(sorted(Counter(row["intent"] for row in selected).items())),
        "family_quotas": family_quotas,
        "intent_quotas": intent_quotas,
        "seed": seed,
        "manifest_status": "PROVISIONAL_NOT_TRAINING_AUTHORIZED",
    }
    return labels, selected, report


def outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"scenes": 0}
    return {
        "scenes": len(rows),
        "strict_clear_rate": sum(truthy(row["strict_clear"]) for row in rows) / len(rows),
        "metric_means": {
            field: sum(float(row[field]) for row in rows) / len(rows) for field in METRIC_FIELDS
        },
    }


def _cluster_delta(
    rows: list[dict[str, Any]],
    left: Callable[[dict[str, Any]], bool],
    right: Callable[[dict[str, Any]], bool],
    value: Callable[[dict[str, Any]], float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    logs = sorted({row["log_name"] for row in rows})
    left_sum = np.zeros(len(logs))
    left_count = np.zeros(len(logs))
    right_sum = np.zeros(len(logs))
    right_count = np.zeros(len(logs))
    log_index = {log_name: index for index, log_name in enumerate(logs)}
    for row in rows:
        index = log_index[row["log_name"]]
        if left(row):
            left_sum[index] += value(row)
            left_count[index] += 1
        if right(row):
            right_sum[index] += value(row)
            right_count[index] += 1
    point = left_sum.sum() / left_count.sum() - right_sum.sum() / right_count.sum()
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, resamples, 1000):
        batch = min(1000, resamples - start)
        indices = rng.integers(0, len(logs), size=(batch, len(logs)))
        sampled_left_count = left_count[indices].sum(axis=1)
        sampled_right_count = right_count[indices].sum(axis=1)
        valid = (sampled_left_count > 0) & (sampled_right_count > 0)
        sampled = (
            left_sum[indices].sum(axis=1)[valid] / sampled_left_count[valid]
            - right_sum[indices].sum(axis=1)[valid] / sampled_right_count[valid]
        )
        deltas.extend(sampled.tolist())
    lower, upper = np.quantile(np.asarray(deltas), (0.025, 0.975))
    return {"point_delta": float(point), "ci_lower": float(lower), "ci_upper": float(upper)}


def build_dev_report(
    model_rows: list[dict[str, str]],
    tier_rows: list[dict[str, str]],
    *,
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tiers = {row["token"]: row for row in tier_rows}
    if len(tiers) != len(tier_rows):
        raise ValueError("Dev tier labels contain duplicate tokens")
    enriched = []
    for row in model_rows:
        if row["token"] not in tiers:
            raise ValueError(f"Model outcome token is outside V4 Dev labels: {row['token']}")
        enriched.append({**row, **tiers[row["token"]]})
    models = sorted({row["model"] for row in enriched})
    for model in models:
        model_tokens = {row["token"] for row in enriched if row["model"] == model}
        if model_tokens != set(tiers):
            raise ValueError(f"Dev model {model} does not exactly cover the 416 V4 labels")

    slices: dict[str, Callable[[dict[str, Any]], bool]] = {
        "all_dev": lambda row: True,
        "tier1": lambda row: truthy(row["eval_tier1"]),
        "control": lambda row: not truthy(row["eval_tier1"]),
        "critical_proximity": lambda row: truthy(row["critical_proximity"]),
        "response_complexity": lambda row: truthy(row["response_complexity"]),
        "current_interaction": lambda row: truthy(row["interaction_tail_flag"]),
        "current_noninteraction": lambda row: not truthy(row["interaction_tail_flag"]),
    }
    comparisons = {
        "tier1_minus_control": (slices["tier1"], slices["control"]),
        "critical_minus_control": (slices["critical_proximity"], slices["control"]),
        "response_minus_control": (slices["response_complexity"], slices["control"]),
        "current_interaction_minus_noninteraction": (
            slices["current_interaction"],
            slices["current_noninteraction"],
        ),
    }
    slice_rows = []
    model_report = {}
    for model_index, model in enumerate(models):
        rows = [row for row in enriched if row["model"] == model]
        summaries = {}
        for name, predicate in slices.items():
            summary = outcome_summary([row for row in rows if predicate(row)])
            summaries[name] = summary
            slice_rows.append(
                {
                    "model": model,
                    "slice": name,
                    "scenes": summary["scenes"],
                    "strict_clear_rate": summary.get("strict_clear_rate", ""),
                    **{
                        field: summary.get("metric_means", {}).get(field, "")
                        for field in METRIC_FIELDS
                    },
                }
            )
        deltas = {}
        for comparison_index, (name, (left, right)) in enumerate(comparisons.items()):
            deltas[name] = {
                "strict_clear_rate": _cluster_delta(
                    rows,
                    left,
                    right,
                    lambda row: float(truthy(row["strict_clear"])),
                    resamples=resamples,
                    seed=seed + model_index * 100 + comparison_index * 2,
                ),
                "pdms_scaled": _cluster_delta(
                    rows,
                    left,
                    right,
                    lambda row: float(row["pdms_scaled"]),
                    resamples=resamples,
                    seed=seed + model_index * 100 + comparison_index * 2 + 1,
                ),
            }
        model_report[model] = {"slices": summaries, "cluster_bootstrap_deltas": deltas}

    def all_negative(comparison: str, metric: str) -> bool:
        return all(
            model_report[model]["cluster_bootstrap_deltas"][comparison][metric]["point_delta"] < 0
            for model in models
        )

    checks = {
        "tier1_strict_clear_harder_all_models": all_negative("tier1_minus_control", "strict_clear_rate"),
        "tier1_pdms_scaled_harder_all_models": all_negative("tier1_minus_control", "pdms_scaled"),
        "critical_strict_clear_harder_all_models": all_negative("critical_minus_control", "strict_clear_rate"),
        "critical_pdms_scaled_harder_all_models": all_negative("critical_minus_control", "pdms_scaled"),
        "current_interaction_strict_clear_harder_all_models": all_negative(
            "current_interaction_minus_noninteraction", "strict_clear_rate"
        ),
        "current_interaction_pdms_scaled_harder_all_models": all_negative(
            "current_interaction_minus_noninteraction", "pdms_scaled"
        ),
    }
    label_gate = all(checks[key] for key in (
        "tier1_strict_clear_harder_all_models",
        "tier1_pdms_scaled_harder_all_models",
        "critical_strict_clear_harder_all_models",
        "critical_pdms_scaled_harder_all_models",
    ))
    reference_rows = [row for row in enriched if row["model"] == models[0]]
    report = {
        "status": "LABEL_DIFFICULTY_GATE_PASS" if label_gate else "LABEL_DIFFICULTY_GATE_FAILED",
        "coverage": {
            "models": models,
            "dev_tokens": len(tiers),
            "model_rows": len(enriched),
        },
        "slice_counts": {
            **{
                name: sum(predicate(row) for row in tier_rows)
                for name, predicate in slices.items()
                if name not in {"current_interaction", "current_noninteraction"}
            },
            "current_interaction": sum(slices["current_interaction"](row) for row in reference_rows),
            "current_noninteraction": sum(
                slices["current_noninteraction"](row) for row in reference_rows
            ),
        },
        "models": model_report,
        "directional_checks": checks,
        "training_ready": label_gate,
        "bootstrap": {"resamples": resamples, "cluster": "log_name", "seed": seed},
    }
    return slice_rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-scene-labels", type=Path, required=True)
    parser.add_argument("--train-tier-labels", type=Path, required=True)
    parser.add_argument("--dev-tier-labels", type=Path, required=True)
    parser.add_argument("--dev-model-outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite V4 closure output: {args.output_dir}")

    train_scene_rows = read_csv(args.train_scene_labels)
    train_tier_rows = read_csv(args.train_tier_labels)
    dev_tier_rows = read_csv(args.dev_tier_labels)
    model_rows = read_csv(args.dev_model_outcomes)
    if len(train_scene_rows) != 8000 or len(train_tier_rows) != 8000:
        raise ValueError("V4 Train inputs must each contain 8,000 rows")
    if len(dev_tier_rows) != 416 or len(model_rows) != 2080:
        raise ValueError("V4 Dev inputs must contain 416 labels and 2,080 model rows")
    if {row["model"] for row in model_rows} != EXPECTED_MODELS:
        raise ValueError("V4 Dev model set differs from the frozen five-model audit")
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")

    train_labels, selected, selection_report = build_train_selection(
        train_scene_rows,
        train_tier_rows,
        seed=args.seed,
    )
    dev_slices, dev_report = build_dev_report(
        model_rows,
        dev_tier_rows,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    report = {
        "status": "V4_CPU_CLOSURE_COMPLETE",
        "train_selection": selection_report,
        "dev_label_validation": dev_report,
        "decision": {
            "provisional_manifest_created": True,
            "gpu_training_authorized": bool(dev_report["training_ready"]),
            "next_action": (
                "freeze GPU-A inputs"
                if dev_report["training_ready"]
                else "stop before GPU and revise policy-independent risk labels using train-only geometry"
            ),
        },
        "input_sha256": {
            "train_scene_labels": sha256_file(args.train_scene_labels),
            "train_tier_labels": sha256_file(args.train_tier_labels),
            "dev_tier_labels": sha256_file(args.dev_tier_labels),
            "dev_model_outcomes": sha256_file(args.dev_model_outcomes),
        },
        "dev_accessed": True,
        "final_accessed": False,
    }

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "train_exclusive_family_labels.csv", train_labels)
    write_csv(args.output_dir / "dev_historical_model_slices.csv", dev_slices)
    (args.output_dir / "provisional_risk_balanced_2000.txt").write_text(
        "".join(f"{row['token']}\n" for row in selected), encoding="utf-8"
    )
    (args.output_dir / "v4_cpu_experiment_closure_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
