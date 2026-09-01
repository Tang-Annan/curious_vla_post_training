from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from projects.dataset_v3.data_prep import stable_key
from projects.dataset_v3.inventory import sha256_file
from projects.dataset_v3.v4_experiment_closure import INTENT_QUOTAS, read_csv, write_csv
from projects.dataset_v3.v4_reward_audit import (
    candidate_rewards,
    group_rows,
    load_reward_module,
    read_jsonl,
    trainer_metrics,
)


TOTAL_SCENES = 2000
EXPECTED_CANDIDATES = 4005
GROUP_SIZE = 4
FAMILY_QUOTAS = {"proximity": 1000, "construction": 500, "signal": 500}
LOG_CAP = 4
HEADROOM_MIN = 0.005
MASTERED_MEAN_MIN = 0.90
ANCHOR_RATIOS = (30, 25, 20)
SENSITIVITY_THRESHOLDS = (0.0025, 0.005, 0.01)
SEED = 20260901
BUCKETS = ("A", "B", "C", "D")
HASH_TIE_SCALE = 1e-6


def read_manifest(path: Path, expected: int | None = None) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Manifest contains duplicate tokens: {path}")
    if expected is not None and len(tokens) != expected:
        raise ValueError(f"Manifest must contain {expected} tokens: {path}")
    return tokens


def proportional_quotas(total: int, base: dict[str, int]) -> dict[str, int]:
    base_total = sum(base.values())
    raw = {key: total * value / base_total for key, value in base.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = {key: index for index, key in enumerate(base)}
    ranked = sorted(base, key=lambda key: (-(raw[key] - quotas[key]), order[key]))
    for key in ranked[:remainder]:
        quotas[key] += 1
    return quotas


def _group_stats(rows: list[dict[str, Any]], reward: Any) -> dict[str, Any]:
    values = []
    strict_clear = []
    reward_hard_safe = []
    for row in rows:
        scores = candidate_rewards(row, reward)
        values.append(float(scores["safety_continuous"]))
        strict_clear.append(
            int(
                reward.classify_cdt_tier(
                    bool(row["parsed_ok"]), trainer_metrics(row)
                )
                == "L3"
            )
        )
        reward_hard_safe.append(int(float(scores["hard_safe"])))
    mean = statistics.fmean(values)
    best = max(values)
    return {
        "mean": mean,
        "best": best,
        "headroom": best - mean,
        "strict_clear_count": sum(strict_clear),
        "reward_hard_safe_count": sum(reward_hard_safe),
        "safety_label_disagreements": sum(
            strict != hard for strict, hard in zip(strict_clear, reward_hard_safe)
        ),
        "parse_failures": sum(not bool(row["parsed_ok"]) for row in rows),
    }


def _bucket(stats: dict[str, Any], headroom_min: float = HEADROOM_MIN) -> str:
    strict_clear_count = int(stats["strict_clear_count"])
    headroom = float(stats["headroom"])
    if 0 < strict_clear_count < GROUP_SIZE:
        return "A"
    if (
        strict_clear_count == GROUP_SIZE
        and float(stats["mean"]) >= MASTERED_MEAN_MIN
        and headroom < headroom_min
    ):
        return "B"
    if headroom >= headroom_min:
        return "C"
    return "D"


def build_features(
    label_rows: list[dict[str, str]],
    screen_rows: list[dict[str, Any]],
    confirm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = {row["token"]: row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("Risk labels contain duplicate tokens")
    tokens = list(labels)
    screen_groups = group_rows(screen_rows, tokens, GROUP_SIZE)
    confirm_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in confirm_rows:
        token = str(row["token"])
        if token in labels:
            confirm_groups[token].append(row)
    invalid_confirm = {
        token: len(rows) for token, rows in confirm_groups.items() if len(rows) != GROUP_SIZE
    }
    if invalid_confirm:
        raise ValueError(f"Unexpected confirm group sizes: {invalid_confirm}")

    reward = load_reward_module()
    features = []
    for token in tokens:
        label = labels[token]
        screen = _group_stats(screen_groups[token], reward)
        confirm = _group_stats(confirm_groups[token], reward) if token in confirm_groups else None
        bucket = _bucket(screen)
        recovery_candidate = bool(
            bucket == "D"
            and screen["strict_clear_count"] == 0
            and confirm is not None
            and confirm["strict_clear_count"] == 0
            and confirm["headroom"] < HEADROOM_MIN
        )
        features.append(
            {
                "token": token,
                "log_name": label["log_name"],
                "intent": label["intent"],
                "exclusive_family": label["exclusive_family"],
                "bucket": bucket,
                "learnable_eligible": int(bucket in {"A", "C"}),
                "recovery_candidate": int(recovery_candidate),
                "screen_mean": screen["mean"],
                "screen_best": screen["best"],
                "screen_headroom": screen["headroom"],
                "screen_strict_clear_count": screen["strict_clear_count"],
                "screen_reward_hard_safe_count": screen["reward_hard_safe_count"],
                "screen_safety_label_disagreements": screen["safety_label_disagreements"],
                "screen_parse_failures": screen["parse_failures"],
                "confirm_available": int(confirm is not None),
                "confirm_mean": None if confirm is None else confirm["mean"],
                "confirm_best": None if confirm is None else confirm["best"],
                "confirm_headroom": None if confirm is None else confirm["headroom"],
                "confirm_strict_clear_count": (
                    None if confirm is None else confirm["strict_clear_count"]
                ),
                "confirm_reward_hard_safe_count": (
                    None if confirm is None else confirm["reward_hard_safe_count"]
                ),
                "confirm_safety_label_disagreements": (
                    None if confirm is None else confirm["safety_label_disagreements"]
                ),
                "confirm_parse_failures": None if confirm is None else confirm["parse_failures"],
            }
        )
    return features


def _tie_cost(rows: list[dict[str, Any]], namespace: str) -> np.ndarray:
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (
            stable_key(SEED, namespace, str(rows[index]["token"])),
            str(rows[index]["token"]),
        ),
    )
    costs = np.empty(len(rows))
    for rank, index in enumerate(ranked):
        costs[index] = rank / max(len(rows), 1)
    return costs


def _solve_joint(
    features: list[dict[str, Any]],
    *,
    anchor_ratio: int,
    anchor_family: dict[str, int],
    anchor_intent: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_total = TOTAL_SCENES * anchor_ratio // 100
    learnable_total = TOTAL_SCENES - anchor_total
    if sum(anchor_family.values()) != anchor_total or sum(anchor_intent.values()) != anchor_total:
        raise ValueError("Anchor family and intent quotas must sum to the anchor total")

    count = len(features)
    learnable_indices = list(range(count))
    anchor_indices = list(range(count, 2 * count))
    constraints: list[tuple[list[int], float, float]] = [
        (learnable_indices, learnable_total, learnable_total),
        (anchor_indices, anchor_total, anchor_total),
    ]
    for field, quotas in (("exclusive_family", FAMILY_QUOTAS), ("intent", INTENT_QUOTAS)):
        for value, quota in quotas.items():
            matching = [index for index, row in enumerate(features) if row[field] == value]
            constraints.append((matching + [count + index for index in matching], quota, quota))
    for field, quotas in (("exclusive_family", anchor_family), ("intent", anchor_intent)):
        for value, quota in quotas.items():
            constraints.append(
                (
                    [count + index for index, row in enumerate(features) if row[field] == value],
                    quota,
                    quota,
                )
            )
    constraints.extend(([index, count + index], -math.inf, 1) for index in range(count))
    constraints.extend(
        ([index], 1, 1) for index, row in enumerate(features) if row["bucket"] == "A"
    )
    by_log: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(features):
        by_log[str(row["log_name"])].append(index)
    constraints.extend(
        (indices + [count + index for index in indices], -math.inf, LOG_CAP)
        for indices in by_log.values()
    )

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for constraint_index, (indices, minimum, maximum) in enumerate(constraints):
        matrix_rows.extend([constraint_index] * len(indices))
        matrix_columns.extend(indices)
        lower.append(minimum)
        upper.append(maximum)
    matrix = coo_matrix(
        (np.ones(len(matrix_rows)), (matrix_rows, matrix_columns)),
        shape=(len(constraints), 2 * count),
    ).tocsr()

    learnable_hash = _tie_cost(features, f"v4-safety-learnable-{anchor_ratio}")
    anchor_hash = _tie_cost(features, f"v4-safety-anchor-{anchor_ratio}")
    cost = np.zeros(2 * count)
    for index, row in enumerate(features):
        cost[index] = (
            -float(row["screen_headroom"])
            + HASH_TIE_SCALE * learnable_hash[index]
        )
        cost[count + index] = HASH_TIE_SCALE * anchor_hash[index]

    variable_upper = np.ones(2 * count)
    for index, row in enumerate(features):
        variable_upper[index] = int(row["learnable_eligible"])
    result = milp(
        c=cost,
        integrality=np.ones(2 * count),
        bounds=Bounds(np.zeros(2 * count), variable_upper),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise ValueError(f"Joint selector MILP did not reach an exact optimum: {result.message}")
    learnable = [row for row, value in zip(features, result.x[:count]) if value > 0.5]
    anchors = [row for row, value in zip(features, result.x[count:]) if value > 0.5]
    if len(learnable) != learnable_total or len(anchors) != anchor_total:
        raise ValueError("Joint selector returned unexpected role totals")
    return learnable, anchors


def select_trial(features: list[dict[str, Any]], anchor_ratio: int) -> dict[str, Any]:
    anchor_total = TOTAL_SCENES * anchor_ratio // 100
    anchor_family = proportional_quotas(anchor_total, FAMILY_QUOTAS)
    anchor_intent = proportional_quotas(anchor_total, INTENT_QUOTAS)
    learnable, anchors = _solve_joint(
        features,
        anchor_ratio=anchor_ratio,
        anchor_family=anchor_family,
        anchor_intent=anchor_intent,
    )
    selected = sorted(
        [*learnable, *anchors],
        key=lambda row: stable_key(SEED, f"v4-safety-output-{anchor_ratio}", str(row["token"])),
    )
    selected_tokens = {str(row["token"]) for row in selected}
    if len(selected_tokens) != TOTAL_SCENES:
        raise ValueError("Learnable and anchor roles overlap")
    learnable_tokens = {str(row["token"]) for row in learnable}
    a_tokens = {str(row["token"]) for row in features if row["bucket"] == "A"}
    if not a_tokens <= learnable_tokens or any(row["bucket"] == "A" for row in anchors):
        raise ValueError("Every A token must be learnable and no A token may be an anchor")
    if any(not row["learnable_eligible"] for row in learnable):
        raise ValueError("Learnable role contains a B or D token")
    if (
        dict(Counter(str(row["exclusive_family"]) for row in anchors)) != anchor_family
        or dict(Counter(str(row["intent"]) for row in anchors)) != anchor_intent
    ):
        raise ValueError("Anchor role violates its frozen margins")
    logs = Counter(str(row["log_name"]) for row in selected)
    family_counts = Counter(str(row["exclusive_family"]) for row in selected)
    intent_counts = Counter(str(row["intent"]) for row in selected)
    if (
        dict(family_counts) != FAMILY_QUOTAS
        or dict(intent_counts) != INTENT_QUOTAS
        or max(logs.values()) > LOG_CAP
    ):
        raise ValueError("Combined selector output violates its frozen margins")
    return {
        "anchor_ratio": anchor_ratio,
        "anchors": anchors,
        "learnable": learnable,
        "selected": selected,
        "anchor_family_quotas": anchor_family,
        "anchor_intent_quotas": anchor_intent,
    }


def _bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["bucket"]) for row in rows)
    return {bucket: counts[bucket] for bucket in BUCKETS}


def _parse_induced_a(row: dict[str, Any]) -> bool:
    parse_failures = int(row["screen_parse_failures"])
    return bool(
        row["bucket"] == "A"
        and parse_failures > 0
        and int(row["screen_strict_clear_count"]) == GROUP_SIZE - parse_failures
    )


def _c_partition(rows: list[dict[str, Any]]) -> dict[str, int]:
    c_rows = [row for row in rows if row["bucket"] == "C"]
    mixed = [row for row in c_rows if 0 < int(row["screen_strict_clear_count"]) < GROUP_SIZE]
    if mixed:
        raise ValueError("C bucket unexpectedly contains strict-clear mixed groups")
    return {
        "C-safe": sum(int(row["screen_strict_clear_count"]) == GROUP_SIZE for row in c_rows),
        "C-unsafe": sum(int(row["screen_strict_clear_count"]) == 0 for row in c_rows),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    logs = Counter(str(row["log_name"]) for row in rows)
    strict_counts = Counter(int(row["screen_strict_clear_count"]) for row in rows)
    return {
        "scenes": len(rows),
        "bucket_counts": _bucket_counts(rows),
        "a_parse_affected": sum(
            row["bucket"] == "A" and int(row["screen_parse_failures"]) > 0 for row in rows
        ),
        "a_parse_induced": sum(_parse_induced_a(row) for row in rows),
        "c_partition": _c_partition(rows),
        "strict_clear_groups": {
            "all_safe": strict_counts[GROUP_SIZE],
            "mixed": sum(strict_counts[count] for count in range(1, GROUP_SIZE)),
            "all_unsafe": strict_counts[0],
        },
        "family_counts": dict(sorted(Counter(str(row["exclusive_family"]) for row in rows).items())),
        "intent_counts": dict(sorted(Counter(str(row["intent"]) for row in rows).items())),
        "unique_logs": len(logs),
        "max_per_log": max(logs.values()) if logs else 0,
        "mean_headroom": statistics.fmean(float(row["screen_headroom"]) for row in rows),
        "safety_label_disagreement_rollouts": sum(
            int(row["screen_safety_label_disagreements"]) for row in rows
        ),
        "recovery_candidates": sum(int(row["recovery_candidate"]) for row in rows),
    }


def selector_explanation_table(
    candidate_rows: list[dict[str, Any]],
    learnable_rows: list[dict[str, Any]] | None,
    anchor_rows: list[dict[str, Any]] | None,
    final_rows: list[dict[str, Any]] | None,
    risk50_tokens: set[str],
    random_tokens: set[str],
) -> list[dict[str, Any]]:
    roles: dict[str, list[dict[str, Any]] | None] = {
        "Candidate 4005": candidate_rows,
        "Learnable role": learnable_rows,
        "Anchor role": anchor_rows,
        "Final 2K": final_rows,
    }

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        bucket_counts = _bucket_counts(rows)
        c_partition = _c_partition(rows)
        tokens = {str(row["token"]) for row in rows}
        return {
            "A": bucket_counts["A"],
            "A parse-induced": sum(_parse_induced_a(row) for row in rows),
            "B": bucket_counts["B"],
            "C-safe": c_partition["C-safe"],
            "C-unsafe": c_partition["C-unsafe"],
            "D": bucket_counts["D"],
            "Mean reward": statistics.fmean(float(row["screen_mean"]) for row in rows),
            "Mean Headroom": statistics.fmean(
                float(row["screen_headroom"]) for row in rows
            ),
            "StrictClear mixed rate": statistics.fmean(
                0 < int(row["screen_strict_clear_count"]) < GROUP_SIZE for row in rows
            ),
            "unique logs": len({str(row["log_name"]) for row in rows}),
            "Overlap with Risk50": len(tokens & risk50_tokens),
            "Overlap with Random": len(tokens & random_tokens),
        }

    candidate_metrics = metrics(candidate_rows)
    by_role = {
        role: (candidate_metrics if role == "Candidate 4005" else None if rows is None else metrics(rows))
        for role, rows in roles.items()
    }
    return [
        {
            "metric": metric,
            **{
                role: None if values is None else values[metric]
                for role, values in by_role.items()
            },
        }
        for metric in candidate_metrics
    ]


def sensitivity_summary(features: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    bucket_by_token = {
        str(row["token"]): _bucket(
            {
                "strict_clear_count": row["screen_strict_clear_count"],
                "mean": row["screen_mean"],
                "headroom": row["screen_headroom"],
            },
            threshold,
        )
        for row in features
    }
    counts = Counter(bucket_by_token.values())
    c_rows = [row for row in features if bucket_by_token[str(row["token"])] == "C"]
    learnable_rows = [
        row for row in features if bucket_by_token[str(row["token"])] in {"A", "C"}
    ]
    return {
        "headroom_threshold": threshold,
        "bucket_counts": {bucket: counts[bucket] for bucket in BUCKETS},
        "c_partition": {
            "C-safe": sum(
                int(row["screen_strict_clear_count"]) == GROUP_SIZE for row in c_rows
            ),
            "C-unsafe": sum(int(row["screen_strict_clear_count"]) == 0 for row in c_rows),
        },
        "learnable_candidates": len(learnable_rows),
        "learnable_family_counts": dict(
            sorted(Counter(str(row["exclusive_family"]) for row in learnable_rows).items())
        ),
        "learnable_intent_counts": dict(
            sorted(Counter(str(row["intent"]) for row in learnable_rows).items())
        ),
    }


def selected_table(
    screen_parquet: Path, selected_tokens: list[str], data_root: Path
) -> tuple[pa.Table, dict[str, Any]]:
    screen = pq.read_table(screen_parquet)
    screen_tokens = [str(answer["token"]) for answer in screen.column("answer").to_pylist()]
    if len(screen_tokens) != len(set(screen_tokens)):
        raise ValueError("Screen parquet contains duplicate answer tokens")
    index_by_token = {token: index for index, token in enumerate(screen_tokens)}
    missing = set(selected_tokens) - set(index_by_token)
    if missing:
        raise ValueError(f"Selected manifest contains {len(missing)} tokens outside Screen parquet")
    table = screen.take(pa.array([index_by_token[token] for token in selected_tokens]))
    output_tokens = [str(answer["token"]) for answer in table.column("answer").to_pylist()]
    if output_tokens != selected_tokens:
        raise ValueError("Selected parquet order differs from its manifest")
    image_paths = [path for paths in table.column("images").to_pylist() for path in paths]
    missing_images = [path for path in image_paths if not (data_root / path).is_file()]
    if missing_images:
        raise ValueError(f"Selected parquet references {len(missing_images)} missing images")
    return table, {
        "rows": table.num_rows,
        "columns": table.column_names,
        "manifest_order_exact": True,
        "image_references": len(image_paths),
        "missing_images": 0,
    }


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite selector output: {args.output_dir}")
    labels = read_csv(args.risk_labels)
    if len(labels) != EXPECTED_CANDIDATES:
        raise ValueError(f"Risk labels must contain {EXPECTED_CANDIDATES} rows")
    features = build_features(labels, read_jsonl(args.screen_enriched), read_jsonl(args.confirm_enriched))
    risk50_tokens = set(read_manifest(args.baseline_manifest, TOTAL_SCENES))
    random_tokens = set(read_manifest(args.random_manifest, TOTAL_SCENES))
    if not risk50_tokens <= {str(row["token"]) for row in features}:
        raise ValueError("Baseline Risk50 escaped the candidate pool")

    trials: dict[int, dict[str, Any]] = {}
    trial_reports: dict[str, Any] = {}
    pool_a = sum(row["bucket"] == "A" for row in features)
    for ratio in ANCHOR_RATIOS:
        try:
            trial = select_trial(features, ratio)
        except ValueError as error:
            trial_reports[str(ratio)] = {"status": "INFEASIBLE", "reason": str(error)}
            continue
        trials[ratio] = trial
        selected_a = sum(row["bucket"] == "A" for row in trial["selected"])
        learnable_a = sum(row["bucket"] == "A" for row in trial["learnable"])
        trial_reports[str(ratio)] = {
            "status": "EXACT_FEASIBLE",
            "anchor_ratio": ratio / 100,
            "anchor_family_quotas": trial["anchor_family_quotas"],
            "anchor_intent_quotas": trial["anchor_intent_quotas"],
            "learnable": _summary(trial["learnable"]),
            "anchors": _summary(trial["anchors"]),
            "combined": _summary(trial["selected"]),
            "a_pool_coverage": selected_a / pool_a if pool_a else 1.0,
            "a_learnable_coverage": learnable_a / pool_a if pool_a else 1.0,
        }
    chosen_ratio = next(
        (
            ratio
            for ratio in ANCHOR_RATIOS
            if ratio in trials and trial_reports[str(ratio)]["a_learnable_coverage"] == 1.0
        ),
        None,
    )

    args.output_dir.mkdir(parents=True)
    membership_path = args.output_dir / "bucket_membership.csv"
    recovery_path = args.output_dir / "recovery_candidate.txt"
    sensitivity_path = args.output_dir / "train_headroom_sensitivity.csv"
    membership_rows = []
    for row in sorted(features, key=lambda item: str(item["token"])):
        membership_rows.append(
            {key: "" if value is None else value for key, value in row.items()}
        )
    write_csv(membership_path, membership_rows)
    recovery_tokens = [
        str(row["token"])
        for row in features
        if row["recovery_candidate"]
    ]
    recovery_path.write_text("".join(f"{token}\n" for token in recovery_tokens), encoding="utf-8")
    sensitivity = {
        str(threshold): sensitivity_summary(features, threshold)
        for threshold in SENSITIVITY_THRESHOLDS
    }
    sensitivity_rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        summary = sensitivity[str(threshold)]
        sensitivity_rows.append(
            {
                "headroom_threshold": threshold,
                **{f"bucket_{bucket}": summary["bucket_counts"][bucket] for bucket in BUCKETS},
                "C-safe": summary["c_partition"]["C-safe"],
                "C-unsafe": summary["c_partition"]["C-unsafe"],
                "learnable_candidates": summary["learnable_candidates"],
                **{
                    f"learnable_family_{family}": summary["learnable_family_counts"].get(family, 0)
                    for family in FAMILY_QUOTAS
                },
                **{
                    f"learnable_intent_{intent}": summary["learnable_intent_counts"].get(intent, 0)
                    for intent in INTENT_QUOTAS
                },
            }
        )
    write_csv(sensitivity_path, sensitivity_rows)
    for ratio, trial in trials.items():
        (args.output_dir / f"trial_anchor{ratio}_2000.txt").write_text(
            "".join(f"{row['token']}\n" for row in trial["selected"]), encoding="utf-8"
        )
    chosen_for_table = None if chosen_ratio is None else trials[chosen_ratio]
    explanation = selector_explanation_table(
        features,
        None if chosen_for_table is None else chosen_for_table["learnable"],
        None if chosen_for_table is None else chosen_for_table["anchors"],
        None if chosen_for_table is None else chosen_for_table["selected"],
        risk50_tokens,
        random_tokens,
    )
    explanation_path = args.output_dir / "selector_explanation_table.csv"
    write_csv(explanation_path, explanation)

    report: dict[str, Any] = {
        "status": (
            "V4_SAFETY_BUCKET_CAPACITY_GATE_FAILED"
            if chosen_ratio is None
            else "V4_SAFETY_BUCKET_DATASET_READY"
        ),
        "bucket_definition": {
            "strict_clear": "parsed L3: collision=1, drivable-area=1, TTC-within-bound=1; exactly aligned with final StrictClear",
            "reward_hard_safe": "diagnostic only: candidate collision, drivable-area, driving-direction, and traffic-light compliance; not used as safe_count",
            "A": "0 < strict_clear_count < 4; direct StrictClear contrast",
            "B": f"strict_clear_count=4, mean>={MASTERED_MEAN_MIN}, headroom<{HEADROOM_MIN}",
            "C": f"not A/B and headroom>={HEADROOM_MIN}; continuous-reward improvable",
            "D": f"remaining low-signal groups with headroom<{HEADROOM_MIN}",
            "recovery_candidate": "D plus strict_clear_count=0 and low headroom in both Screen and independent Confirm G4 blocks",
        },
        "selection_protocol": {
            "anchor_ratios_audited": list(ANCHOR_RATIOS),
            "joint_roles": "anchor and learnable binary variables are solved in one MILP; anchors are never pre-drawn",
            "anchors": "fixed stable-hash rank supplies the random term under anchor family/intent quotas",
            "learnable": "every A token is fixed into learnable; remaining learnable capacity uses C safety-continuous headroom; no composite semantic score",
            "anchor_bucket_rule": "anchor receives no forced retention from bucket identity; A cannot enter anchor",
            "decision": "try 30% anchors, then 25%, then 20%; if 20% is still infeasible, fail the capacity gate without changing the protocol",
        },
        "headroom_sensitivity": {
            "purpose": "Train-only robustness audit; the frozen selector threshold remains 0.005",
            "selection_rerun": False,
            "thresholds": sensitivity,
        },
        "candidate_pool": _summary(features),
        "confirm_coverage": sum(int(row["confirm_available"]) for row in features),
        "capacity_trials": trial_reports,
        "chosen_anchor_ratio": chosen_ratio,
        "selector_explanation_table": explanation,
        "input_sha256": {
            "risk_labels": sha256_file(args.risk_labels),
            "screen_enriched": sha256_file(args.screen_enriched),
            "confirm_enriched": sha256_file(args.confirm_enriched),
            "baseline_manifest": sha256_file(args.baseline_manifest),
            "random_manifest": sha256_file(args.random_manifest),
            "screen_parquet": sha256_file(args.screen_parquet),
        },
        "dev_accessed": False,
        "final_accessed": False,
        "gpu_used": False,
        "audit_output_sha256": {
            "membership": sha256_file(membership_path),
            "recovery_manifest": sha256_file(recovery_path),
            "headroom_sensitivity": sha256_file(sensitivity_path),
            "selector_explanation_table": sha256_file(explanation_path),
        },
    }

    if chosen_ratio is not None:
        chosen = trials[chosen_ratio]
        selected_tokens = [str(row["token"]) for row in chosen["selected"]]
        table, parquet_report = selected_table(args.screen_parquet, selected_tokens, args.data_root)
        prefix = f"risk50_learnable_anchor{chosen_ratio}_2000"
        manifest_path = args.output_dir / f"{prefix}.txt"
        parquet_path = args.output_dir / f"{prefix}.parquet"
        learnable_path = args.output_dir / f"learnable_{len(chosen['learnable'])}.txt"
        anchor_path = args.output_dir / f"random_anchor_{len(chosen['anchors'])}.txt"
        manifest_path.write_text("".join(f"{token}\n" for token in selected_tokens), encoding="utf-8")
        learnable_path.write_text(
            "".join(f"{row['token']}\n" for row in chosen["learnable"]), encoding="utf-8"
        )
        anchor_path.write_text(
            "".join(f"{row['token']}\n" for row in chosen["anchors"]), encoding="utf-8"
        )
        pq.write_table(table, parquet_path)
        overlap = len(set(selected_tokens) & risk50_tokens)
        random_overlap = len(set(selected_tokens) & random_tokens)
        report["chosen_dataset"] = {
            "summary": _summary(chosen["selected"]),
            "learnable": _summary(chosen["learnable"]),
            "anchors": _summary(chosen["anchors"]),
            "risk50_overlap": overlap,
            "risk50_jaccard": overlap / (2 * TOTAL_SCENES - overlap),
            "random_overlap": random_overlap,
            "random_jaccard": random_overlap / (2 * TOTAL_SCENES - random_overlap),
            "selector_explanation_table": explanation,
            "parquet": parquet_report,
        }
        report["output_sha256"] = {
            "manifest": sha256_file(manifest_path),
            "parquet": sha256_file(parquet_path),
            "learnable_manifest": sha256_file(learnable_path),
            "anchor_manifest": sha256_file(anchor_path),
        }
    report_path = args.output_dir / "v4_safety_bucket_selector_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-labels", type=Path, required=True)
    parser.add_argument("--screen-enriched", type=Path, required=True)
    parser.add_argument("--confirm-enriched", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--screen-parquet", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
