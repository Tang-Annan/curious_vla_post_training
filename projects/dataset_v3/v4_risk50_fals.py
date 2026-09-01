from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from projects.dataset_v3.data_prep import stable_key
from projects.dataset_v3.inventory import sha256_file
from projects.dataset_v3.v4_experiment_closure import read_csv, write_csv
from projects.dataset_v3.v4_grpo_selector import selected_table
from projects.dataset_v3.v4_reward_audit import group_rows, read_jsonl


TOTAL_SCENES = 2000
EXPECTED_CANDIDATES = 4005
GROUP_SIZE = 4
FAMILY_QUOTAS = {"proximity": 1000, "construction": 500, "signal": 500}
INTENT_QUOTAS = {"straight": 1333, "left": 434, "right": 233}
LOG_CAP = 4
SEED = 20260901
SENSITIVITY_THRESHOLDS = (0.0025, 0.005, 0.01)
ROLES = ("direct_safety_contrast", "fals_learnable", "random_anchor")
BUCKETS = ("A", "B", "C-safe", "C-unsafe", "D")


def read_manifest(path: Path, expected: int | None = None) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Manifest contains duplicate tokens: {path}")
    if expected is not None and len(tokens) != expected:
        raise ValueError(f"Manifest must contain {expected} tokens: {path}")
    return tokens


def _strict_clear(row: dict[str, Any]) -> bool:
    return bool(row["parsed_ok"]) and all(
        float(row[field]) == 1.0
        for field in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
        )
    )


def _raw_pdms(row: dict[str, Any]) -> float:
    value = 0.0 if not bool(row["parsed_ok"]) else float(row["pdms"])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"raw PDMS is outside [0, 1]: {value}")
    return value


def build_features(
    label_rows: list[dict[str, str]], screen_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labels = {row["token"]: row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("Risk labels contain duplicate tokens")
    tokens = list(labels)
    groups = group_rows(screen_rows, tokens, GROUP_SIZE)
    features = []
    for token in tokens:
        rows = groups[token]
        values = [_raw_pdms(row) for row in rows]
        mean = statistics.fmean(values)
        best = max(values)
        headroom = best - mean
        difficulty = 1.0 - mean
        fals = difficulty * headroom
        strict_clear_count = sum(_strict_clear(row) for row in rows)
        mixed = 0 < strict_clear_count < GROUP_SIZE
        parse_failures = sum(not bool(row["parsed_ok"]) for row in rows)
        parse_induced = bool(
            mixed and parse_failures and strict_clear_count == GROUP_SIZE - parse_failures
        )
        if mixed:
            role = "direct_safety_contrast"
            bucket = "A"
        elif fals > 0.0:
            role = "fals_learnable"
            bucket = "C-safe" if strict_clear_count == GROUP_SIZE else "C-unsafe"
        else:
            role = "random_anchor"
            bucket = "B" if strict_clear_count == GROUP_SIZE else "D"
        label = labels[token]
        features.append(
            {
                "token": token,
                "log_name": label["log_name"],
                "intent": label["intent"],
                "exclusive_family": label["exclusive_family"],
                "mean_raw_pdms": mean,
                "best_raw_pdms": best,
                "difficulty": difficulty,
                "headroom": headroom,
                "fals": fals,
                "strict_clear_count": strict_clear_count,
                "strict_clear_mixed": int(mixed),
                "parse_failures": parse_failures,
                "parse_induced_mixed": int(parse_induced),
                "semantic_bucket": bucket,
                "selector_role": role,
            }
        )
    return features


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["strict_clear_mixed"]),
        -int(float(row["fals"]) > 0.0),
        -float(row["fals"]),
        -float(row["headroom"]),
        stable_key(SEED, "v4-risk50-fals-rank", str(row["token"])),
        str(row["token"]),
    )


def select_ranked(features: list[dict[str, Any]], *, use_intent: bool) -> list[dict[str, Any]]:
    family_remaining = dict(FAMILY_QUOTAS)
    intent_remaining = dict(INTENT_QUOTAS) if use_intent else {}
    log_counts: Counter[str] = Counter()
    selected = []
    for row in sorted(features, key=_rank_key):
        family = str(row["exclusive_family"])
        intent = str(row["intent"])
        log_name = str(row["log_name"])
        if family not in family_remaining:
            raise ValueError(f"Unknown family: {family}")
        if use_intent and intent not in intent_remaining:
            raise ValueError(f"Unknown intent: {intent}")
        if family_remaining[family] == 0:
            continue
        if use_intent and intent_remaining[intent] == 0:
            continue
        if log_counts[log_name] == LOG_CAP:
            continue
        selected.append(row)
        family_remaining[family] -= 1
        if use_intent:
            intent_remaining[intent] -= 1
        log_counts[log_name] += 1
        if len(selected) == TOTAL_SCENES:
            break

    if len(selected) != TOTAL_SCENES or any(family_remaining.values()) or any(
        intent_remaining.values()
    ):
        raise ValueError(
            "deterministic constrained greedy did not fill exact quotas: "
            f"selected={len(selected)}, family_remaining={family_remaining}, "
            f"intent_remaining={intent_remaining}"
        )
    return selected


def select_with_intent_fallback(features: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        selected = select_ranked(features, use_intent=True)
    except ValueError as error:
        intent_trial = {"status": "INTENT_GREEDY_INFEASIBLE", "reason": str(error)}
        selected = select_ranked(features, use_intent=False)
        intent_constraint_used = False
    else:
        intent_trial = {"status": "EXACT_FEASIBLE", "reason": None}
        intent_constraint_used = True
    selected_tokens = {str(row["token"]) for row in selected}
    mixed_capacity_excluded = [
        str(row["token"])
        for row in sorted(features, key=_rank_key)
        if row["strict_clear_mixed"] and str(row["token"]) not in selected_tokens
    ]
    return {
        "selected": selected,
        "intent_trial": intent_trial,
        "intent_constraint_used": intent_constraint_used,
        "mixed_capacity_excluded": mixed_capacity_excluded,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    logs = Counter(str(row["log_name"]) for row in rows)
    strict_counts = Counter(int(row["strict_clear_count"]) for row in rows)
    learnable = sum(row["selector_role"] != "random_anchor" for row in rows)
    return {
        "scenes": len(rows),
        "family_counts": dict(sorted(Counter(str(row["exclusive_family"]) for row in rows).items())),
        "intent_counts": dict(sorted(Counter(str(row["intent"]) for row in rows).items())),
        "role_counts": {
            role: sum(row["selector_role"] == role for row in rows) for role in ROLES
        },
        "semantic_bucket_counts": {
            bucket: sum(row["semantic_bucket"] == bucket for row in rows) for bucket in BUCKETS
        },
        "learnable_candidates": learnable,
        "learnable_rate": learnable / len(rows) if rows else None,
        "fals_positive": sum(float(row["fals"]) > 0.0 for row in rows),
        "zero_headroom": sum(float(row["headroom"]) == 0.0 for row in rows),
        "strict_clear_groups": {
            "all_safe": strict_counts[GROUP_SIZE],
            "mixed": sum(strict_counts[count] for count in range(1, GROUP_SIZE)),
            "all_unsafe": strict_counts[0],
        },
        "parse_induced_mixed": sum(int(row["parse_induced_mixed"]) for row in rows),
        "mean_raw_pdms": statistics.fmean(float(row["mean_raw_pdms"]) for row in rows)
        if rows
        else None,
        "mean_headroom": statistics.fmean(float(row["headroom"]) for row in rows)
        if rows
        else None,
        "mean_fals": statistics.fmean(float(row["fals"]) for row in rows)
        if rows
        else None,
        "unique_logs": len(logs),
        "max_per_log": max(logs.values()) if logs else 0,
    }


def selector_explanation_table(
    candidate_rows: list[dict[str, Any]],
    learnable_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    risk50_tokens: set[str],
    random_tokens: set[str],
) -> list[dict[str, Any]]:
    roles = {
        "Candidate 4005": candidate_rows,
        "Learnable role": learnable_rows,
        "Anchor role": anchor_rows,
        "Final 2K": selected_rows,
    }

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        tokens = {str(row["token"]) for row in rows}
        bucket_counts = Counter(str(row["semantic_bucket"]) for row in rows)
        return {
            "A": bucket_counts["A"],
            "A parse-induced": sum(int(row["parse_induced_mixed"]) for row in rows),
            "B": bucket_counts["B"],
            "C-safe": bucket_counts["C-safe"],
            "C-unsafe": bucket_counts["C-unsafe"],
            "D": bucket_counts["D"],
            "Mean reward": statistics.fmean(float(row["mean_raw_pdms"]) for row in rows)
            if rows
            else None,
            "Mean Headroom": statistics.fmean(float(row["headroom"]) for row in rows)
            if rows
            else None,
            "StrictClear mixed rate": statistics.fmean(
                bool(row["strict_clear_mixed"]) for row in rows
            )
            if rows
            else None,
            "unique logs": len({str(row["log_name"]) for row in rows}),
            "Overlap with Risk50": len(tokens & risk50_tokens),
            "Overlap with Random": len(tokens & random_tokens),
        }

    by_role = {name: metrics(rows) for name, rows in roles.items()}
    return [
        {"metric": metric, **{name: values[metric] for name, values in by_role.items()}}
        for metric in by_role["Candidate 4005"]
    ]


def _headroom_sensitivity(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(threshold): {
            "candidates": sum(float(row["headroom"]) >= threshold for row in features),
            "family_counts": dict(
                sorted(
                    Counter(
                        str(row["exclusive_family"])
                        for row in features
                        if float(row["headroom"]) >= threshold
                    ).items()
                )
            ),
            "intent_counts": dict(
                sorted(
                    Counter(
                        str(row["intent"])
                        for row in features
                        if float(row["headroom"]) >= threshold
                    ).items()
                )
            ),
        }
        for threshold in SENSITIVITY_THRESHOLDS
    }


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite selector output: {args.output_dir}")
    labels = read_csv(args.risk_labels)
    if len(labels) != EXPECTED_CANDIDATES:
        raise ValueError(f"Risk labels must contain {EXPECTED_CANDIDATES} rows")
    features = build_features(labels, read_jsonl(args.screen_enriched))
    feature_tokens = {str(row["token"]) for row in features}
    risk50_tokens = set(read_manifest(args.baseline_manifest, TOTAL_SCENES))
    random_tokens = set(read_manifest(args.random_manifest, TOTAL_SCENES))
    if not risk50_tokens <= feature_tokens:
        raise ValueError("Baseline Risk50 escaped the candidate pool")

    trial = select_with_intent_fallback(features)
    selected_ranked = trial["selected"]
    selected = sorted(
        selected_ranked,
        key=lambda row: (
            stable_key(SEED, "v4-risk50-fals-output-order", str(row["token"])),
            str(row["token"]),
        ),
    )
    selected_tokens = [str(row["token"]) for row in selected]
    selected_token_set = set(selected_tokens)
    learnable = [row for row in selected if row["selector_role"] != "random_anchor"]
    anchors = [row for row in selected if row["selector_role"] == "random_anchor"]
    explanation = selector_explanation_table(
        features, learnable, anchors, selected, risk50_tokens, random_tokens
    )

    args.output_dir.mkdir(parents=True)
    rank_by_token = {
        str(row["token"]): rank for rank, row in enumerate(selected_ranked, start=1)
    }
    output_order = {token: index for index, token in enumerate(selected_tokens, start=1)}
    membership_rows = [
        {
            **row,
            "selected": int(str(row["token"]) in selected_token_set),
            "selection_rank": rank_by_token.get(str(row["token"]), ""),
            "output_order": output_order.get(str(row["token"]), ""),
        }
        for row in sorted(features, key=lambda item: str(item["token"]))
    ]
    membership_path = args.output_dir / "selector_membership.csv"
    explanation_path = args.output_dir / "selector_explanation_table.csv"
    manifest_path = args.output_dir / "risk50_fals_n1_2000.txt"
    parquet_path = args.output_dir / "risk50_fals_n1_2000.parquet"
    write_csv(membership_path, membership_rows)
    write_csv(explanation_path, explanation)
    manifest_path.write_text("".join(f"{token}\n" for token in selected_tokens), encoding="utf-8")
    table, parquet_report = selected_table(args.screen_parquet, selected_tokens, args.data_root)
    pq.write_table(table, parquet_path)

    risk50_overlap = len(selected_token_set & risk50_tokens)
    random_overlap = len(selected_token_set & random_tokens)
    report = {
        "status": "V4_RISK50_FALS_N1_READY",
        "dataset_name": "Risk50-NoIntent/Fallback-FALS-2K",
        "formula": {
            "mean": "mean(raw_pdms_G4)",
            "best": "max(raw_pdms_G4)",
            "headroom": "best - mean",
            "difficulty": "1 - mean",
            "fals": "difficulty * headroom",
        },
        "selection_protocol": {
            "priority": [
                "StrictClear-mixed",
                "FALS > 0",
                "FALS descending",
                "Headroom descending",
                "stable hash tie-break",
            ],
            "family_quotas": FAMILY_QUOTAS,
            "intent_first_trial_quotas": INTENT_QUOTAS,
            "max_per_log": LOG_CAP,
            "fallback": "remove only intent quotas after INTENT_GREEDY_INFEASIBLE",
            "training_order": "independent stable hash; selector rank is not used as curriculum",
        },
        "intent_trial": trial["intent_trial"],
        "intent_constraint_used": trial["intent_constraint_used"],
        "candidate_pool": _summary(features),
        "headroom_sensitivity": {
            "selection_rerun": False,
            "thresholds": _headroom_sensitivity(features),
        },
        "mixed_capacity_excluded": {
            "count": len(trial["mixed_capacity_excluded"]),
            "tokens": trial["mixed_capacity_excluded"],
        },
        "selector_explanation_table": explanation,
        "chosen_dataset": {
            "summary": _summary(selected),
            "learnable": _summary(learnable),
            "anchors": _summary(anchors),
            "risk50_overlap": risk50_overlap,
            "risk50_jaccard": risk50_overlap / (2 * TOTAL_SCENES - risk50_overlap),
            "random_overlap": random_overlap,
            "random_jaccard": random_overlap / (2 * TOTAL_SCENES - random_overlap),
            "parquet": parquet_report,
        },
        "input_sha256": {
            "risk_labels": sha256_file(args.risk_labels),
            "screen_enriched": sha256_file(args.screen_enriched),
            "baseline_manifest": sha256_file(args.baseline_manifest),
            "random_manifest": sha256_file(args.random_manifest),
            "screen_parquet": sha256_file(args.screen_parquet),
        },
        "output_sha256": {
            "manifest": sha256_file(manifest_path),
            "parquet": sha256_file(parquet_path),
            "membership": sha256_file(membership_path),
            "selector_explanation_table": sha256_file(explanation_path),
        },
        "dev_accessed": False,
        "final_accessed": False,
        "gpu_used": False,
    }
    report_path = args.output_dir / "v4_risk50_fals_n1_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-labels", type=Path, required=True)
    parser.add_argument("--screen-enriched", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--screen-parquet", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
