#!/usr/bin/env python3
"""Audit SDR versus SLDR reward and GRPO advantage geometry on D0 train rollouts."""

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from verl.trainer.core_algos import compute_grpo_outcome_advantage
from verl.utils.reward_score.navsim.safety_dense_reward import REQUIRED_METRICS, compute_sldr


COMPONENT_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
)
PAIR_TOLERANCE = 1e-9
MATERIAL_DELTA = 0.10


def load_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Training manifest contains duplicate tokens.")
    return tokens


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_recover_rows(path: Path, manifest: list[str]) -> tuple[dict[str, list[dict]], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    raw_presence = Counter()
    missing = Counter()
    parsed_missing = Counter()
    unparsed_missing = Counter()
    recovered_fields = Counter()
    recovered_rows = 0

    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            token = str(row["token"])
            parsed_ok = bool(row.get("parsed_ok", True))
            row_missing = []
            for key in REQUIRED_METRICS:
                if key in row:
                    raw_presence[key] += 1
                else:
                    missing[key] += 1
                    (parsed_missing if parsed_ok else unparsed_missing)[key] += 1
                    row_missing.append(key)

            if parsed_ok and row_missing:
                raise ValueError(f"Parsed rollout {token} is missing NAVSIM metrics: {', '.join(row_missing)}")
            if not parsed_ok:
                for key in REQUIRED_METRICS:
                    if key in row and float(row[key]) != 0.0:
                        raise ValueError(f"Parse-failure rollout {token} has nonzero {key}.")
                for key in ("safe", "training_reward"):
                    if key in row and float(row[key]) != 0.0:
                        raise ValueError(f"Parse-failure rollout {token} has nonzero {key}.")
                if row_missing:
                    recovered_rows += 1
                    for key in row_missing:
                        row[key] = 0.0
                        recovered_fields[key] += 1
            groups[token].append(row)

    allowed = set(manifest)
    unknown = set(groups) - allowed
    absent = allowed - set(groups)
    if unknown:
        raise ValueError(f"Rollout file contains {len(unknown)} tokens outside the training manifest.")
    if absent:
        raise ValueError(f"Rollout file is missing {len(absent)} training-manifest tokens.")
    mismatched = [token for token in manifest if len(groups[token]) != 4]
    if mismatched:
        raise ValueError(f"Expected four rollouts per token; {len(mismatched)} groups differ.")

    total_rows = sum(len(rows) for rows in groups.values())
    coverage = {
        "raw_rows": total_rows,
        "raw_presence_counts": {key: raw_presence[key] for key in REQUIRED_METRICS},
        "raw_missing_counts": {key: missing[key] for key in REQUIRED_METRICS},
        "parsed_row_missing_counts": {key: parsed_missing[key] for key in REQUIRED_METRICS},
        "parse_failure_missing_counts": {key: unparsed_missing[key] for key in REQUIRED_METRICS},
        "parse_failure_rows": sum(not bool(row.get("parsed_ok", True)) for rows in groups.values() for row in rows),
        "recovered_parse_failure_rows": recovered_rows,
        "recovered_zero_fields": dict(recovered_fields),
        "recovery_basis": "Production navsim_grouped._zero_result returns deterministic zero scores for parse failures.",
    }
    return {token: groups[token] for token in manifest}, coverage


def production_safe(row: dict) -> bool:
    return float(row["no_at_fault_collisions"]) > 0.0 and float(row["drivable_area_compliance"]) > 0.0


def strict_safe(row: dict) -> bool:
    return math.isclose(float(row["no_at_fault_collisions"]), 1.0) and math.isclose(
        float(row["drivable_area_compliance"]), 1.0
    )


def composition(flags: list[bool]) -> str:
    safe_count = sum(flags)
    if safe_count == len(flags):
        return "all_safe"
    if safe_count == 0:
        return "all_unsafe"
    return "mixed_safety"


def grpo_advantages(rewards: list[float], indices: list[str]) -> np.ndarray:
    reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    mask = torch.ones_like(reward_tensor)
    advantages, _ = compute_grpo_outcome_advantage(reward_tensor, mask, indices)
    return advantages[:, 0].detach().cpu().numpy().astype(float)


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def geometry_summary(group_rows: list[dict]) -> dict:
    return {
        "groups": len(group_rows),
        "exact_zero_groups": sum(row["reward_std"] == 0.0 for row in group_rows),
        "exact_zero_group_ratio": float(np.mean([row["reward_std"] == 0.0 for row in group_rows])),
        "unique_rewards": distribution([row["unique_rewards"] for row in group_rows]),
        "reward_gap": distribution([row["reward_gap"] for row in group_rows]),
        "reward_std": distribution([row["reward_std"] for row in group_rows]),
        "advantage_span": distribution([row["advantage_span"] for row in group_rows]),
    }


def bootstrap_group_means(rows: list[dict], samples: int, seed: int) -> dict:
    if not rows:
        return {
            key: {"mean_difference": None, "group_bootstrap_95_ci": None}
            for key in COMPONENT_METRICS
        }
    differences = np.asarray([[row[key] for row in rows] for key in COMPONENT_METRICS], dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((len(COMPONENT_METRICS), samples), dtype=float)
    chunk_size = 1000
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, differences.shape[1], size=(stop - start, differences.shape[1]))
        bootstrap_means[:, start:stop] = differences[:, indices].mean(axis=2)
    report = {}
    for metric_index, key in enumerate(COMPONENT_METRICS):
        low, high = np.quantile(bootstrap_means[metric_index], [0.025, 0.975])
        report[key] = {
            "mean_difference": float(differences[metric_index].mean()),
            "group_bootstrap_95_ci": [float(low), float(high)],
        }
    return report


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    manifest = load_manifest(args.train_manifest)
    groups, field_coverage = load_and_recover_rows(args.d0_rollouts, manifest)

    flat_rows = [row for token in manifest for row in groups[token]]
    sdr_rewards = [float(row["pdms_scaled"]) for row in flat_rows]
    sldr_rewards = [float(compute_sldr(row)) for row in flat_rows]
    group_indices = [token for token in manifest for _ in range(4)]
    sdr_advantages = grpo_advantages(sdr_rewards, group_indices)
    sldr_advantages = grpo_advantages(sldr_rewards, group_indices)

    recorded_mismatches = sum(
        "training_reward" not in row
        or not math.isclose(float(row["training_reward"]), sdr, rel_tol=0.0, abs_tol=1e-9)
        for row, sdr in zip(flat_rows, sdr_rewards)
    )
    stored_safe_mismatches = sum(
        "safe" not in row or bool(float(row["safe"])) != production_safe(row) for row in flat_rows
    )
    strict_mislabels = [row for row in flat_rows if production_safe(row) and not strict_safe(row)]
    partial_collision_rows = [row for row in flat_rows if 0.0 < float(row["no_at_fault_collisions"]) < 1.0]

    group_geometry = []
    composition_counts = {"production": Counter(), "strict": Counter()}
    for group_index, token in enumerate(manifest):
        start = group_index * 4
        stop = start + 4
        production_class = composition([production_safe(row) for row in groups[token]])
        strict_class = composition([strict_safe(row) for row in groups[token]])
        composition_counts["production"][production_class] += 1
        composition_counts["strict"][strict_class] += 1
        sdr = np.asarray(sdr_rewards[start:stop], dtype=float)
        sldr = np.asarray(sldr_rewards[start:stop], dtype=float)
        sdr_adv = sdr_advantages[start:stop]
        sldr_adv = sldr_advantages[start:stop]
        delta = float(np.mean(np.abs(sldr_adv - sdr_adv)))
        group_geometry.append(
            {
                "token": token,
                "production_composition": production_class,
                "strict_composition": strict_class,
                "sdr_reward_std": float(sdr.std(ddof=1)),
                "sldr_reward_std": float(sldr.std(ddof=1)),
                "sdr_unique_rewards": int(np.unique(sdr).size),
                "sldr_unique_rewards": int(np.unique(sldr).size),
                "sdr_reward_gap": float(sdr.max() - sdr.min()),
                "sldr_reward_gap": float(sldr.max() - sldr.min()),
                "sdr_advantage_span": float(sdr_adv.max() - sdr_adv.min()),
                "sldr_advantage_span": float(sldr_adv.max() - sldr_adv.min()),
                "delta_a": delta,
                "material_delta": delta >= MATERIAL_DELTA,
            }
        )

    pair_rewards_sdr = []
    pair_rewards_sldr = []
    pair_indices = []
    pair_metadata = []
    for token in manifest:
        for first, second in combinations(range(4), 2):
            pair_id = f"{token}:{first}:{second}"
            pair_indices.extend((pair_id, pair_id))
            pair_rewards_sdr.extend((float(groups[token][first]["pdms_scaled"]), float(groups[token][second]["pdms_scaled"])))
            pair_rewards_sldr.extend((float(compute_sldr(groups[token][first])), float(compute_sldr(groups[token][second]))))
            pair_metadata.append((token, first, second))
    pair_sdr_advantages = grpo_advantages(pair_rewards_sdr, pair_indices)
    pair_sldr_advantages = grpo_advantages(pair_rewards_sldr, pair_indices)

    pair_counts = Counter()
    non_tie_errors = []
    preference_differences: dict[str, list[dict]] = defaultdict(list)
    expected_non_tie_magnitude = 1.0 / math.sqrt(2.0)
    for pair_index, (token, first, second) in enumerate(pair_metadata):
        offset = pair_index * 2
        sdr_difference = pair_rewards_sdr[offset + 1] - pair_rewards_sdr[offset]
        sldr_difference = pair_rewards_sldr[offset + 1] - pair_rewards_sldr[offset]
        sdr_tie = abs(sdr_difference) <= PAIR_TOLERANCE
        sldr_tie = abs(sldr_difference) <= PAIR_TOLERANCE
        pair_counts["total"] += 1
        pair_counts["sdr_ties" if sdr_tie else "sdr_non_ties"] += 1
        pair_counts["sldr_ties" if sldr_tie else "sldr_non_ties"] += 1
        if sdr_tie and not sldr_tie:
            pair_counts["sdr_tie_sldr_breaks"] += 1
        if not sdr_tie and not sldr_tie and np.sign(sdr_difference) != np.sign(sldr_difference):
            pair_counts["preference_reversals"] += 1
        for advantages, tied in ((pair_sdr_advantages[offset : offset + 2], sdr_tie), (pair_sldr_advantages[offset : offset + 2], sldr_tie)):
            if tied:
                pair_counts["zero_advantage_pairs"] += int(np.all(np.abs(advantages) <= 1e-7))
            else:
                non_tie_errors.extend(abs(abs(value) - expected_non_tie_magnitude) for value in advantages)

        new_preference = not sldr_tie and (sdr_tie or np.sign(sdr_difference) != np.sign(sldr_difference))
        if new_preference:
            winner, loser = (second, first) if sldr_difference > 0 else (first, second)
            if not strict_safe(groups[token][winner]):
                pair_counts["new_preference_unsafe_pairs"] += 1
                preference_differences[token].append(
                    {
                        key: float(groups[token][winner][key]) - float(groups[token][loser][key])
                        for key in COMPONENT_METRICS
                    }
                )

    group_preference_differences = []
    for token, rows in preference_differences.items():
        group_preference_differences.append(
            {"token": token, "pairs": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in COMPONENT_METRICS}}
        )
    bootstrap = bootstrap_group_means(group_preference_differences, args.bootstrap_samples, args.seed)

    material_groups = [row for row in group_geometry if row["material_delta"]]
    strict_mixed_material = sum(row["strict_composition"] == "mixed_safety" for row in material_groups)
    production_mixed_material = sum(row["production_composition"] == "mixed_safety" for row in material_groups)
    sdr_rows = [
        {
            "reward_std": row["sdr_reward_std"],
            "unique_rewards": row["sdr_unique_rewards"],
            "reward_gap": row["sdr_reward_gap"],
            "advantage_span": row["sdr_advantage_span"],
        }
        for row in group_geometry
    ]
    sldr_rows = [
        {
            "reward_std": row["sldr_reward_std"],
            "unique_rewards": row["sldr_unique_rewards"],
            "reward_gap": row["sldr_reward_gap"],
            "advantage_span": row["sldr_advantage_span"],
        }
        for row in group_geometry
    ]
    sdr_geometry = geometry_summary(sdr_rows)
    sldr_geometry = geometry_summary(sldr_rows)
    zero_reduction_pp = 100.0 * (
        sdr_geometry["exact_zero_group_ratio"] - sldr_geometry["exact_zero_group_ratio"]
    )
    component_gates = {
        key: bootstrap[key]["group_bootstrap_95_ci"] is not None
        and bootstrap[key]["group_bootstrap_95_ci"][1] >= 0.0
        for key in ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress")
    }
    gates = {
        "data_and_recompute_complete": recorded_mismatches == 0,
        "safe_semantics_valid": len(strict_mislabels) == 0,
        "material_delta_group_ratio_at_least_10pct": len(material_groups) / len(manifest) >= 0.10,
        "strict_mixed_share_of_material_groups_at_least_50pct": bool(material_groups)
        and strict_mixed_material / len(material_groups) >= 0.50,
        "exact_zero_group_reduction_at_least_5pp": zero_reduction_pp >= 5.0,
        "collision_bootstrap_ci_not_entirely_below_zero": component_gates["no_at_fault_collisions"],
        "dac_bootstrap_ci_not_entirely_below_zero": component_gates["drivable_area_compliance"],
        "progress_bootstrap_ci_not_entirely_below_zero": component_gates["ego_progress"],
    }
    gates["passed"] = all(gates.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "group_geometry.csv", group_geometry)
    write_csv(args.output_dir / "unsafe_preference_group_differences.csv", group_preference_differences)
    report = {
        "input": {
            "d0_rollouts": str(args.d0_rollouts),
            "d0_rollouts_sha256": file_sha256(args.d0_rollouts),
            "train_manifest": str(args.train_manifest),
            "train_manifest_sha256": file_sha256(args.train_manifest),
            "tokens": len(manifest),
            "rollouts": len(flat_rows),
            "rollouts_per_group": 4,
            "dev_or_heldout_accessed": False,
        },
        "field_coverage_and_recovery": field_coverage,
        "reward_recompute": {
            "sdr_definition": "Production compute_score_group_fast overall = NAVSIM pdms_scaled.",
            "sldr_definition": "Production safety_dense_reward.compute_sldr.",
            "recorded_training_reward_sdr_mismatch_rows": recorded_mismatches,
        },
        "safe_semantics": {
            "stored_vs_production_mismatch_rows": stored_safe_mismatches,
            "production_rule": "no_at_fault_collisions > 0 and drivable_area_compliance > 0",
            "strict_rule": "no_at_fault_collisions == 1 and drivable_area_compliance == 1",
            "partial_collision_rows": len(partial_collision_rows),
            "partial_collision_production_safe_rows": len(strict_mislabels),
            "affected_tokens": len({str(row["token"]) for row in strict_mislabels}),
            "navsim_semantics": "A no_at_fault_collisions score of 0.5 is assigned to an at-fault collision with a non-agent object by PDMScorer._calculate_no_at_fault_collision.",
            "systematic_mislabel": bool(strict_mislabels),
        },
        "group_composition": {
            scope: {key: counts[key] for key in ("all_safe", "mixed_safety", "all_unsafe")}
            for scope, counts in composition_counts.items()
        },
        "reward_and_advantage_geometry": {
            "sdr": sdr_geometry,
            "sldr": sldr_geometry,
            "exact_zero_reduction_percentage_points": zero_reduction_pp,
            "delta_a": distribution([row["delta_a"] for row in group_geometry]),
            "material_threshold": MATERIAL_DELTA,
            "material_groups": len(material_groups),
            "material_group_ratio": len(material_groups) / len(manifest),
            "strict_mixed_material_groups": strict_mixed_material,
            "strict_mixed_share_of_material_groups": strict_mixed_material / len(material_groups) if material_groups else None,
            "production_mixed_material_groups": production_mixed_material,
            "production_mixed_share_of_material_groups": production_mixed_material / len(material_groups) if material_groups else None,
            "material_groups_by_strict_composition": dict(Counter(row["strict_composition"] for row in material_groups)),
            "material_groups_by_production_composition": dict(Counter(row["production_composition"] for row in material_groups)),
        },
        "g2_pair_audit": {
            **pair_counts,
            "pairs_per_g4_group": 6,
            "expected_non_tie_advantage_magnitude": expected_non_tie_magnitude,
            "max_non_tie_magnitude_error": max(non_tie_errors) if non_tie_errors else None,
        },
        "unsafe_new_preference_bootstrap": {
            "definition": "SLDR winner minus loser when SDR tied or preferred the other trajectory; winner must be strict-unsafe.",
            "preference_pairs": pair_counts["new_preference_unsafe_pairs"],
            "independent_group_units": len(group_preference_differences),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "metrics": bootstrap,
        },
        "gates": gates,
        "decision": "allow_one_r4_sldr" if gates["passed"] else "close_all_sldr_formal_training",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "s0_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-rollouts", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    report = analyze(args)
    print(json.dumps({"gates": report["gates"], "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
