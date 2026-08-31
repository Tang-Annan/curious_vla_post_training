from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from projects.dataset_v3.data_prep import stable_key
from projects.dataset_v3.inventory import sha256_file


STRICT_VEHICLE_DISTANCE_M = 3.0
STRICT_VRU_DISTANCE_M = 5.0
SPAN_SCALED_RECIPE = {
    "warmup_positive": 667,
    "mixed_positive": 1000,
    "mixed_negative": 166,
    "mixed_recovery": 167,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flag(row: dict[str, str], key: str) -> bool:
    return row[key] == "1"


def distance(row: dict[str, str], key: str) -> float | None:
    return None if row[key] == "" else float(row[key])


def derive_tiers(row: dict[str, str]) -> dict[str, int | str]:
    vehicle_distance = distance(row, "horizon_vehicle_distance_m")
    vru_distance = distance(row, "horizon_vru_distance_m")
    critical_vehicle = vehicle_distance is not None and vehicle_distance <= STRICT_VEHICLE_DISTANCE_M
    critical_vru = vru_distance is not None and vru_distance <= STRICT_VRU_DISTANCE_M
    critical_proximity = critical_vehicle or critical_vru
    visible_critical_proximity = (
        critical_vehicle and flag(row, "current_vehicle_front_context")
    ) or (critical_vru and flag(row, "current_vru_front_context"))
    expert_response = any(
        flag(row, key)
        for key in ("expert_turn", "expert_lateral", "expert_braking", "expert_stop_to_go")
    )
    front_construction_response = (
        flag(row, "construction_present")
        and flag(row, "current_construction_front_context")
        and expert_response
    )
    current_signal_hard_response = (
        flag(row, "current_traffic_control")
        and (flag(row, "expert_braking") or flag(row, "expert_stop_to_go"))
    )
    response_complexity = front_construction_response or current_signal_hard_response
    eval_tier1 = critical_proximity or response_complexity
    train_tier1 = visible_critical_proximity or response_complexity
    families = []
    if critical_proximity:
        families.append("critical_proximity")
    if front_construction_response:
        families.append("front_construction_response")
    if current_signal_hard_response:
        families.append("current_signal_hard_response")
    return {
        "critical_vehicle": int(critical_vehicle),
        "critical_vru": int(critical_vru),
        "critical_proximity": int(critical_proximity),
        "visible_critical_proximity": int(visible_critical_proximity),
        "front_construction_response": int(front_construction_response),
        "current_signal_hard_response": int(current_signal_hard_response),
        "response_complexity": int(response_complexity),
        "eval_tier1": int(eval_tier1),
        "train_tier1": int(train_tier1),
        "tier1_families": "|".join(families) if families else "control",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, rows: Iterable[dict[str, Any]], seed: int, namespace: str) -> None:
    tokens = sorted((str(row["token"]) for row in rows), key=lambda token: stable_key(seed, namespace, token))
    path.write_text("".join(f"{token}\n" for token in tokens), encoding="utf-8")


def build_report(train_rows: list[dict[str, str]], dev_rows: list[dict[str, str]]) -> dict[str, Any]:
    train = [{**row, **derive_tiers(row)} for row in train_rows]
    dev = [{**row, **derive_tiers(row)} for row in dev_rows]
    learnable = [row for row in train if row["train_tier1"]]
    positive = [row for row in learnable if flag(row, "positive_supported")]
    negative = [row for row in learnable if flag(row, "stable_policy_negative")]
    recovery = [row for row in learnable if flag(row, "confirmed_paired_recovery")]
    capacity = {
        "positive_required": SPAN_SCALED_RECIPE["warmup_positive"] + SPAN_SCALED_RECIPE["mixed_positive"],
        "positive_available": len(positive),
        "negative_required": SPAN_SCALED_RECIPE["mixed_negative"],
        "negative_available": len(negative),
        "recovery_required": SPAN_SCALED_RECIPE["mixed_recovery"],
        "recovery_available": len(recovery),
        "negative_recovery_overlap": len({row["token"] for row in negative} & {row["token"] for row in recovery}),
    }
    feasible = all(
        capacity[f"{family}_available"] >= capacity[f"{family}_required"]
        for family in ("positive", "negative", "recovery")
    )
    tier_keys = (
        "critical_proximity",
        "visible_critical_proximity",
        "front_construction_response",
        "current_signal_hard_response",
        "response_complexity",
        "eval_tier1",
        "train_tier1",
    )
    return {
        "status": "V4_SPAN_RISK_DECISION_COMPLETE",
        "semantic_boundary": {
            "critical_proximity": "GT expert-path minimum separation: vehicle <= 3m or VRU <= 5m",
            "response_complexity": "current-front construction plus expert response, or current traffic control plus braking/stop-to-go",
            "evaluation_tier1": "critical proximity union response complexity; policy-independent",
            "training_tier1": "evaluation Tier-1 with matching current-input support for critical proximity",
            "negative_recovery": "policy-derived rollout stability proxy; not real-world takeover/correction data",
        },
        "thresholds": {
            "strict_vehicle_distance_m": STRICT_VEHICLE_DISTANCE_M,
            "strict_vru_distance_m": STRICT_VRU_DISTANCE_M,
        },
        "coverage": {"train": len(train), "dev": len(dev)},
        "train_tier_counts": {key: sum(bool(row[key]) for row in train) for key in tier_keys},
        "dev_tier_counts": {key: sum(bool(row[key]) for row in dev) for key in tier_keys},
        "dev_by_frozen_split": {
            split: {
                "scenes": sum(row["split"] == split for row in dev),
                **{
                    key: sum(row["split"] == split and bool(row[key]) for row in dev)
                    for key in tier_keys
                },
            }
            for split in ("dev_tail", "dev_natural")
        },
        "train_tier1_family_counts": dict(
            sorted(Counter(row["tier1_families"] for row in learnable).items())
        ),
        "scaled_span_recipe": SPAN_SCALED_RECIPE,
        "recipe_capacity": capacity,
        "recipe_status": "FEASIBLE_TIGHT_MARGIN" if feasible else "INSUFFICIENT_CONFIRMED_CAPACITY",
        "decision": {
            "broad_event_is_context_only": True,
            "freeze_tier1_for_manifest_construction": True,
            "launch_training": False,
            "next_gate": "freeze deterministic role manifests and family quotas before any GPU training",
        },
        "dev_accessed": True,
        "final_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--dev-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite V4 decision output: {args.output_dir}")

    train_rows = read_csv(args.train_labels)
    dev_rows = read_csv(args.dev_labels)
    if len(train_rows) != 8000 or len(dev_rows) != 416:
        raise ValueError("V4 decision inputs have unexpected coverage")
    train_tokens = {row["token"] for row in train_rows}
    dev_tokens = {row["token"] for row in dev_rows}
    if len(train_tokens) != len(train_rows) or len(dev_tokens) != len(dev_rows) or train_tokens & dev_tokens:
        raise ValueError("V4 decision inputs contain duplicates or train/dev overlap")

    train = [{**row, **derive_tiers(row)} for row in train_rows]
    dev = [{**row, **derive_tiers(row)} for row in dev_rows]
    train_labels = [
        {"token": row["token"], "split": row["split"], **derive_tiers(row)} for row in train_rows
    ]
    dev_labels = [
        {"token": row["token"], "split": row["split"], **derive_tiers(row)} for row in dev_rows
    ]
    report = build_report(train_rows, dev_rows)
    report["input_sha256"] = {
        "train_scene_labels": sha256_file(args.train_labels),
        "dev_scene_labels": sha256_file(args.dev_labels),
    }

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "train_v4_tier_labels.csv", train_labels)
    write_csv(args.output_dir / "dev_v4_tier_labels.csv", dev_labels)
    write_manifest(args.output_dir / "train_v4_tier1_learnable.txt", (row for row in train if row["train_tier1"]), args.seed, "v4-tier1")
    write_manifest(args.output_dir / "train_v4_positive.txt", (row for row in train if row["train_tier1"] and flag(row, "positive_supported")), args.seed, "v4-positive")
    write_manifest(args.output_dir / "train_v4_policy_negative.txt", (row for row in train if row["train_tier1"] and flag(row, "stable_policy_negative")), args.seed, "v4-negative")
    write_manifest(args.output_dir / "train_v4_paired_recovery.txt", (row for row in train if row["train_tier1"] and flag(row, "confirmed_paired_recovery")), args.seed, "v4-recovery")
    write_manifest(args.output_dir / "dev_v4_tier1.txt", (row for row in dev if row["eval_tier1"]), args.seed, "v4-dev-tier1")
    write_manifest(args.output_dir / "dev_v4_control.txt", (row for row in dev if not row["eval_tier1"]), args.seed, "v4-dev-control")
    (args.output_dir / "v4_span_risk_decision_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
