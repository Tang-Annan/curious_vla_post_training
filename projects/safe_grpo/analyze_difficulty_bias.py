#!/usr/bin/env python3
"""Diagnose FALS/GRPO scale mismatch and bounded dynamic-sampling cost."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_rollouts import load_manifest, training_reward


def load_groups(path: Path, allowed_tokens: list[str], expected_rollouts: int) -> dict[str, list[dict]]:
    allowed = set(allowed_tokens)
    groups: dict[str, list[dict]] = defaultdict(list)
    unknown = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            token = str(row["token"])
            if token not in allowed:
                unknown.add(token)
            else:
                groups[token].append(row)

    if unknown:
        raise ValueError(f"Rollouts contain {len(unknown)} tokens outside the active manifest.")
    missing = allowed - groups.keys()
    if missing:
        raise ValueError(f"Rollout coverage is missing {len(missing)} active tokens.")
    mismatched = [token for token, rows in groups.items() if len(rows) != expected_rollouts]
    if mismatched:
        raise ValueError(f"Expected {expected_rollouts} rollouts; {len(mismatched)} groups differ.")
    return groups


def summarize_groups(
    dataset: str,
    groups: dict[str, list[dict]],
    fals_tokens: set[str],
    random_tokens: set[str],
) -> list[dict]:
    summaries = []
    for token, rows in sorted(groups.items()):
        rewards = np.asarray([training_reward(row) for row in rows], dtype=float)
        mean = float(rewards.mean())
        maximum = float(rewards.max())
        safe = [float(row["safe"]) for row in rows if "safe" in row]
        summaries.append(
            {
                "dataset": dataset,
                "token": token,
                "n": len(rows),
                "in_fals": token in fals_tokens,
                "in_random": token in random_tokens,
                "reward_mean": mean,
                "reward_std": float(rewards.std(ddof=1)),
                "reward_min": float(rewards.min()),
                "reward_max": maximum,
                "reward_gap": maximum - float(rewards.min()),
                "headroom": maximum - mean,
                "difficulty": 1.0 - mean,
                "learnability": (1.0 - mean) * (maximum - mean),
                "safe_rate": float(np.mean(safe)) if safe else None,
                "parse_success_rate": float(np.mean([bool(row.get("parsed_ok", True)) for row in rows])),
            }
        )
    return summaries


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = rankdata(first)
    second_ranks = rankdata(second)
    if first_ranks.std() == 0 or second_ranks.std() == 0:
        return 0.0
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def bootstrap_mean_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> list[float]:
    estimates = np.empty(samples, dtype=float)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        estimates[start : start + size] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def quintile_report(
    rows: list[dict], key: str, bootstrap_samples: int, rng: np.random.Generator
) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (float(row[key]), str(row["token"])))
    report = []
    for quintile in range(5):
        selected = [row for index, row in enumerate(ordered) if index * 5 // len(ordered) == quintile]
        gap = np.asarray([row["reward_gap"] for row in selected], dtype=float)
        std = np.asarray([row["reward_std"] for row in selected], dtype=float)
        report.append(
            {
                "quintile": quintile + 1,
                "count": len(selected),
                f"{key}_mean": float(np.mean([row[key] for row in selected])),
                "reward_gap_mean": float(gap.mean()),
                "reward_gap_mean_ci": bootstrap_mean_ci(gap, bootstrap_samples, rng),
                "reward_std_mean": float(std.mean()),
                "reward_std_mean_ci": bootstrap_mean_ci(std, bootstrap_samples, rng),
            }
        )
    return report


def subset_report(rows: list[dict]) -> dict:
    std = np.asarray([row["reward_std"] for row in rows], dtype=float)
    return {
        "groups": len(rows),
        "reward_mean": float(np.mean([row["reward_mean"] for row in rows])),
        "reward_std_mean": float(std.mean()),
        "reward_gap_mean": float(np.mean([row["reward_gap"] for row in rows])),
        "headroom_mean": float(np.mean([row["headroom"] for row in rows])),
        "exact_zero_std_ratio": float(np.mean(std == 0.0)),
    }


def advantage_rows(dataset: str, groups: dict[str, list[dict]], eps: float = 1e-6) -> list[dict]:
    rows = []
    for token, group in sorted(groups.items()):
        rewards = np.asarray([training_reward(row) for row in group], dtype=float)
        centered = rewards - rewards.mean()
        std = rewards.std(ddof=1)
        grpo = centered / (std + eps)
        rows.append(
            {
                "dataset": dataset,
                "token": token,
                "reward_gap": float(rewards.max() - rewards.min()),
                "reward_std": float(std),
                "grpo_abs_mean": float(np.abs(grpo).mean()),
                "dr_grpo_abs_mean": float(np.abs(centered).mean()),
            }
        )
    return rows


def simulate_sampling_caps(
    informative_ratio: float,
    target_groups: int,
    steps: int,
    maximum: int,
    trials: int,
    rng: np.random.Generator,
) -> list[dict]:
    reports = []
    for cap in range(2, maximum + 1):
        draws = rng.random((trials, cap, target_groups)) < informative_ratio
        cumulative = draws.sum(axis=2).cumsum(axis=1)
        reached = cumulative >= target_groups
        succeeded = reached.any(axis=1)
        attempts = np.where(succeeded, reached.argmax(axis=1) + 1, cap)
        step_failure = float(np.mean(~succeeded))
        reports.append(
            {
                "max_generation_batches": cap,
                "step_fill_failure_probability": step_failure,
                "run_fill_failure_probability": float(1.0 - (1.0 - step_failure) ** steps),
                "mean_raw_rollout_overhead": float(attempts.mean()),
            }
        )
    return reports


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, d0_rows: list[dict], advantage: list[dict]) -> None:
    width, height = 1200, 520
    panels = ((70, 55, 500, 400), (660, 55, 500, 400))

    def scale(value: float, low: float, high: float, start: float, length: float, invert: bool = False) -> float:
        ratio = 0.5 if high == low else (value - low) / (high - low)
        return start + length * (1.0 - ratio if invert else ratio)

    difficulty = np.asarray([row["difficulty"] for row in d0_rows], dtype=float)
    std = np.asarray([row["reward_std"] for row in d0_rows], dtype=float)
    e2_advantage = [row for row in advantage if row["dataset"] == "e2"]
    gap = np.asarray([row["reward_gap"] for row in e2_advantage], dtype=float)
    dr_abs = np.asarray([row["dr_grpo_abs_mean"] for row in e2_advantage], dtype=float)
    grpo_abs = np.asarray([row["grpo_abs_mean"] for row in e2_advantage], dtype=float)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.label{font-size:15px}.title{font-size:18px;font-weight:bold}</style>',
    ]
    for left, top, panel_width, panel_height in panels:
        elements.append(f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="none" class="axis"/>')

    left, top, panel_width, panel_height = panels[0]
    for x_value, y_value in zip(difficulty, std):
        x = scale(float(x_value), float(difficulty.min()), float(difficulty.max()), left, panel_width)
        y = scale(float(y_value), 0.0, float(std.max()), top, panel_height, invert=True)
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.5" fill="#2878b5" fill-opacity="0.28"/>')
    elements.extend(
        [
            f'<text x="{left}" y="30" class="title">D0 difficulty vs. reward variance</text>',
            f'<text x="{left + panel_width / 2}" y="490" text-anchor="middle" class="label">Difficulty (1 - mean reward)</text>',
            f'<text x="20" y="{top + panel_height / 2}" transform="rotate(-90 20 {top + panel_height / 2})" text-anchor="middle" class="label">Sample std</text>',
        ]
    )

    left, top, panel_width, panel_height = panels[1]
    y_max = max(float(dr_abs.max()), float(grpo_abs.max()))
    for x_value, dr_value, grpo_value in zip(gap, dr_abs, grpo_abs):
        x = scale(float(x_value), 0.0, float(gap.max()), left, panel_width)
        y_dr = scale(float(dr_value), 0.0, y_max, top, panel_height, invert=True)
        y_grpo = scale(float(grpo_value), 0.0, y_max, top, panel_height, invert=True)
        elements.append(f'<circle cx="{x:.2f}" cy="{y_dr:.2f}" r="2" fill="#d95319" fill-opacity="0.45"/>')
        elements.append(f'<circle cx="{x:.2f}" cy="{y_grpo:.2f}" r="1.5" fill="#2878b5" fill-opacity="0.28"/>')
    elements.extend(
        [
            f'<text x="{left}" y="30" class="title">E2 reward gap vs. advantage magnitude</text>',
            f'<text x="{left + panel_width / 2}" y="490" text-anchor="middle" class="label">Reward gap</text>',
            f'<text x="610" y="{top + panel_height / 2}" transform="rotate(-90 610 {top + panel_height / 2})" text-anchor="middle" class="label">Mean |advantage|</text>',
            f'<circle cx="{left + 20}" cy="{top + 20}" r="4" fill="#d95319"/><text x="{left + 30}" y="{top + 25}" class="label">Dr.GRPO</text>',
            f'<circle cx="{left + 125}" cy="{top + 20}" r="4" fill="#2878b5"/><text x="{left + 135}" y="{top + 25}" class="label">GRPO</text>',
        ]
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict:
    train_tokens = load_manifest(args.train_manifest)
    fals_tokens = load_manifest(args.fals_manifest)
    random_tokens = load_manifest(args.random_manifest)
    if len(fals_tokens) != args.expected_selection_size or len(random_tokens) != args.expected_selection_size:
        raise ValueError(f"FALS and random manifests must each contain {args.expected_selection_size} tokens.")
    train_set = set(train_tokens)
    if not set(fals_tokens) <= train_set or not set(random_tokens) <= train_set:
        raise ValueError("Selection manifests must be subsets of the frozen train manifest.")

    d0_groups = load_groups(args.d0_rollouts, train_tokens, 4)
    e2_groups = load_groups(args.e2_rollouts, fals_tokens, 2)
    fals_set, random_set = set(fals_tokens), set(random_tokens)
    d0_rows = summarize_groups("d0", d0_groups, fals_set, random_set)
    e2_rows = summarize_groups("e2", e2_groups, fals_set, random_set)
    rng = np.random.default_rng(args.seed)

    d0_difficulty = np.asarray([row["difficulty"] for row in d0_rows], dtype=float)
    d0_std = np.asarray([row["reward_std"] for row in d0_rows], dtype=float)
    d0_gap = np.asarray([row["reward_gap"] for row in d0_rows], dtype=float)
    e2_std = np.asarray([row["reward_std"] for row in e2_rows], dtype=float)
    e2_gap = np.asarray([row["reward_gap"] for row in e2_rows], dtype=float)
    nonzero_gap = e2_gap[e2_gap > 0]
    informative_ratio = float(np.mean(e2_std > 0))
    gap_q25, gap_q75 = np.quantile(nonzero_gap, [0.25, 0.75])

    sampling = simulate_sampling_caps(
        informative_ratio,
        args.target_groups,
        args.steps,
        args.max_generation_batches,
        args.monte_carlo_trials,
        rng,
    )
    selected = next((row for row in sampling if row["run_fill_failure_probability"] < 0.01), None)
    r1_gate = informative_ratio >= 0.50 and float(gap_q75 - gap_q25) >= 0.10
    r2_gate = (
        float(np.mean(e2_std == 0.0)) >= 0.25
        and selected is not None
        and selected["mean_raw_rollout_overhead"] <= 2.0
    )
    report = {
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "monte_carlo_trials": args.monte_carlo_trials,
        "d0": {
            "groups": len(d0_rows),
            "rollouts": sum(len(rows) for rows in d0_groups.values()),
            "all": subset_report(d0_rows),
            "fals_top": subset_report([row for row in d0_rows if row["in_fals"]]),
            "random": subset_report([row for row in d0_rows if row["in_random"]]),
            "spearman_difficulty_reward_std": spearman(d0_difficulty, d0_std),
            "spearman_difficulty_reward_gap": spearman(d0_difficulty, d0_gap),
            "difficulty_quintiles": quintile_report(d0_rows, "difficulty", args.bootstrap_samples, rng),
            "headroom_quintiles": quintile_report(d0_rows, "headroom", args.bootstrap_samples, rng),
        },
        "e2": {
            "groups": len(e2_rows),
            "rollouts": sum(len(rows) for rows in e2_groups.values()),
            "exact_zero_std_ratio": float(np.mean(e2_std == 0.0)),
            "informative_group_ratio": informative_ratio,
            "nonzero_reward_gap_q25": float(gap_q25),
            "nonzero_reward_gap_q75": float(gap_q75),
            "nonzero_reward_gap_iqr": float(gap_q75 - gap_q25),
        },
        "dynamic_sampling": {"selected_cap": selected, "caps": sampling},
        "gates": {
            "r1": {
                "informative_ratio_at_least_0_50": informative_ratio >= 0.50,
                "nonzero_gap_iqr_at_least_0_10": float(gap_q75 - gap_q25) >= 0.10,
                "passed": r1_gate,
            },
            "r2": {
                "exact_zero_ratio_at_least_0_25": float(np.mean(e2_std == 0.0)) >= 0.25,
                "run_failure_below_0_01_with_cap_at_most_8": selected is not None,
                "mean_overhead_at_most_2_0": selected is not None and selected["mean_raw_rollout_overhead"] <= 2.0,
                "passed": r2_gate,
            },
        },
        "next_stage": "r1" if r1_gate else ("r2_g" if r2_gate else "f0"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "group_metrics.csv", d0_rows + e2_rows)
    advantage = advantage_rows("d0", d0_groups) + advantage_rows("e2", e2_groups)
    write_csv(args.output_dir / "advantage_scale.csv", advantage)
    (args.output_dir / "r0_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_plot(args.output_dir / "difficulty_bias.svg", d0_rows, advantage)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-rollouts", type=Path, required=True)
    parser.add_argument("--e2-rollouts", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--fals-manifest", type=Path, required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-selection-size", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--monte-carlo-trials", type=int, default=100000)
    parser.add_argument("--target-groups", type=int, default=4)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--max-generation-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    report = analyze(args)
    print(json.dumps({"gates": report["gates"], "next_stage": report["next_stage"]}, indent=2))


if __name__ == "__main__":
    main()
