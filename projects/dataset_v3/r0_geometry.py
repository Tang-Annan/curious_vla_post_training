from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from projects.dataset_v3.s1_pipeline import candidate_tier, group_rows, read_jsonl, read_manifest, sha256_file


TIER_VALUE = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
FORMULAS = ("raw_pdms", "cdt_pdms", "cdt_task")
SELECTORS = ("random", "tailmix")
EPS = 1e-6


def _bounded(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} is outside [0,1]: {numeric}")
    return numeric


def task_quality(row: dict[str, Any]) -> float:
    progress = _bounded(row["ego_progress"], "ego_progress")
    comfort = _bounded(row["history_comfort"], "history_comfort")
    return (5.0 * progress + 2.0 * comfort) / 7.0


def reward_value(row: dict[str, Any], formula: str) -> tuple[float, float, str | None]:
    if formula not in FORMULAS:
        raise ValueError(f"Unknown reward formula: {formula}")
    pdms = _bounded(row["pdms"], "pdms")
    tier = candidate_tier(row)
    if formula == "raw_pdms":
        return pdms, pdms, tier
    quality = pdms if formula == "cdt_pdms" else task_quality(row)
    if tier is None:
        return 0.0, quality, None
    return (2.0 * TIER_VALUE[tier] + quality) / 7.0, quality, tier


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        label: ordered[round((len(ordered) - 1) * fraction)]
        for label, fraction in (
            ("q00", 0.0),
            ("q25", 0.25),
            ("q50", 0.5),
            ("q75", 0.75),
            ("q90", 0.9),
            ("q95", 0.95),
            ("q100", 1.0),
        )
    }


def summarize_cell(
    selector: str,
    formula: str,
    tokens: list[str],
    groups: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group_records = []
    tier_counts: Counter[str] = Counter()
    zero_composition: Counter[str] = Counter()
    cross_tier_pairs = 0
    cross_tier_inversions = 0
    cross_tier_ties = 0
    cross_tier_positive_gaps = []
    within_tier_quality_pairs = 0
    within_tier_quality_inversions = 0
    invalid_rollouts = 0

    for token in tokens:
        rewards, qualities, tiers = [], [], []
        for row in groups[token]:
            reward, quality, tier = reward_value(row, formula)
            if not math.isfinite(reward):
                raise ValueError(f"Non-finite reward for {token}")
            rewards.append(reward)
            qualities.append(quality)
            tiers.append(tier)
            if tier is None:
                invalid_rollouts += 1
            else:
                tier_counts[tier] += 1

        for left in range(len(rewards)):
            for right in range(left + 1, len(rewards)):
                left_tier, right_tier = tiers[left], tiers[right]
                if left_tier is None or right_tier is None:
                    continue
                left_value, right_value = TIER_VALUE[left_tier], TIER_VALUE[right_tier]
                if left_value != right_value:
                    cross_tier_pairs += 1
                    high, low = (left, right) if left_value > right_value else (right, left)
                    gap = rewards[high] - rewards[low]
                    if gap < -1e-12:
                        cross_tier_inversions += 1
                    elif abs(gap) <= 1e-12:
                        cross_tier_ties += 1
                    else:
                        cross_tier_positive_gaps.append(gap)
                elif abs(qualities[left] - qualities[right]) > 1e-12:
                    within_tier_quality_pairs += 1
                    high, low = (left, right) if qualities[left] > qualities[right] else (right, left)
                    if rewards[high] <= rewards[low] + 1e-12:
                        within_tier_quality_inversions += 1

        reward_mean = statistics.fmean(rewards)
        reward_std = statistics.stdev(rewards)
        advantages = [(value - reward_mean) / (reward_std + EPS) for value in rewards]
        valid_tiers = [tier for tier in tiers if tier is not None]
        mixed_tier = len(set(valid_tiers)) >= 2
        exact_zero = reward_std <= 1e-12
        if exact_zero:
            if not valid_tiers:
                zero_composition["all_invalid"] += 1
            elif len(valid_tiers) < 4:
                zero_composition["partial_invalid"] += 1
            elif len(set(valid_tiers)) == 1:
                zero_composition[f"all_{valid_tiers[0]}"] += 1
            else:
                zero_composition["mixed_tier_reward_tie"] += 1
        group_records.append(
            {
                "selector": selector,
                "formula": formula,
                "cell": ("RR" if selector == "random" else "TR")
                if formula == "raw_pdms"
                else ("RC" if selector == "random" else "TC"),
                "token": token,
                "tiers": "|".join(tier or "invalid" for tier in tiers),
                "valid_rollouts": len(valid_tiers),
                "mixed_tier": int(mixed_tier),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "exact_zero": int(exact_zero),
                "low_nonzero": int(1e-12 < reward_std < 0.05),
                "effective": int(any(abs(value) > 1e-12 for value in advantages)),
                "quality_mean": statistics.fmean(qualities),
                "advantages": "|".join(f"{value:.12g}" for value in advantages),
            }
        )

    stds = [float(row["reward_std"]) for row in group_records]
    exact_zero_groups = sum(bool(row["exact_zero"]) for row in group_records)
    low_nonzero_groups = sum(bool(row["low_nonzero"]) for row in group_records)
    effective_groups = sum(bool(row["effective"]) for row in group_records)
    mixed_groups = sum(bool(row["mixed_tier"]) for row in group_records)
    summary = {
        "selector": selector,
        "formula": formula,
        "groups": len(tokens),
        "rollouts": len(tokens) * 4,
        "tier_counts": {tier: tier_counts[tier] for tier in TIER_VALUE},
        "invalid_rollouts": invalid_rollouts,
        "mixed_tier_groups": mixed_groups,
        "mixed_tier_rate": mixed_groups / len(tokens),
        "exact_zero_groups": exact_zero_groups,
        "exact_zero_rate": exact_zero_groups / len(tokens),
        "low_nonzero_groups": low_nonzero_groups,
        "low_nonzero_rate": low_nonzero_groups / len(tokens),
        "effective_groups": effective_groups,
        "effective_group_rate": effective_groups / len(tokens),
        "reward_std_quantiles": _quantiles(stds),
        "mean_reward_std": statistics.fmean(stds),
        "zero_group_composition": dict(sorted(zero_composition.items())),
        "cross_tier_pairs": cross_tier_pairs,
        "cross_tier_inversions": cross_tier_inversions,
        "cross_tier_ties": cross_tier_ties,
        "minimum_positive_cross_tier_gap": min(cross_tier_positive_gaps, default=None),
        "within_tier_quality_pairs": within_tier_quality_pairs,
        "within_tier_quality_inversions_or_ties": within_tier_quality_inversions,
    }
    return summary, group_records


def build_geometry(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    screen_tokens = read_manifest(args.screen_manifest)
    groups = group_rows(read_jsonl(args.rollouts), screen_tokens)
    selector_tokens = {
        "random": read_manifest(args.random_manifest),
        "tailmix": read_manifest(args.tailmix_manifest),
    }
    for selector, tokens in selector_tokens.items():
        if len(tokens) != 2000 or not set(tokens) <= set(screen_tokens):
            raise ValueError(f"{selector} is not a frozen 2,000-token Screen subset")

    cells, group_records = {}, []
    for formula in FORMULAS:
        for selector in SELECTORS:
            summary, records = summarize_cell(selector, formula, selector_tokens[selector], groups)
            cells[f"{selector}_{formula}"] = summary
            group_records.extend(records)
    comparison = {}
    for selector in SELECTORS:
        raw = cells[f"{selector}_raw_pdms"]
        comparison[selector] = {}
        for formula in ("cdt_pdms", "cdt_task"):
            candidate = cells[f"{selector}_{formula}"]
            comparison[selector][formula] = {
                "effective_group_rate_delta_vs_raw": candidate["effective_group_rate"] - raw["effective_group_rate"],
                "exact_zero_rate_delta_vs_raw": candidate["exact_zero_rate"] - raw["exact_zero_rate"],
                "low_nonzero_rate_delta_vs_raw": candidate["low_nonzero_rate"] - raw["low_nonzero_rate"],
            }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "group_geometry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_records[0]))
        writer.writeheader()
        writer.writerows(group_records)
    report = {
        "status": "CANDIDATE_GEOMETRY_ONLY_NOT_REWARD_FREEZE",
        "raw_formula": "clip(pdms,0,1)",
        "candidate_formulas": {
            "cdt_pdms": "(2*tier_value + pdms)/7",
            "cdt_task": "(2*tier_value + Q_task)/7; Q_task=(5*ego_progress+2*history_comfort)/7",
        },
        "candidate_interval_bounds": {
            tier: [2 * value / 7, (2 * value + 1) / 7] for tier, value in TIER_VALUE.items()
        },
        "minimum_theoretical_cross_tier_gap": 1 / 7,
        "task_quality_audit": {
            "source_pdms_weighted_terms": {"ego_progress": 5, "time_to_collision_within_bound": 5, "history_comfort": 2},
            "removed_tier_terms": ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound"],
            "retained_task_terms": {"ego_progress": 5, "history_comfort": 2},
            "normalization": 7,
            "semantic_status": "COMPLETE_NON_SAFETY_REMAINDER_OF_RECORDED_PDMS_COMPONENTS",
        },
        "pdms_candidate_safety_double_count": True,
        "invalid_policy": "parse-invalid remains outside L0-L3 and receives technical zero reward",
        "grpo_std_convention": "sample std (torch.std default correction=1) plus eps=1e-6",
        "cells": cells,
        "comparison_vs_raw": comparison,
        "rollouts_sha256": sha256_file(args.rollouts),
        "screen_manifest_sha256": sha256_file(args.screen_manifest),
        "random_manifest_sha256": sha256_file(args.random_manifest),
        "tailmix_manifest_sha256": sha256_file(args.tailmix_manifest),
        "selector_report_sha256": sha256_file(args.selector_report),
        "dev_accessed": False,
        "final_accessed": False,
    }
    (args.output_dir / "r0_geometry_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "exit_code").write_text("0\n", encoding="utf-8")
    (args.output_dir / "COMPLETE").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--tailmix-manifest", type=Path, required=True)
    parser.add_argument("--selector-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    build_geometry(parser.parse_args())


if __name__ == "__main__":
    main()
