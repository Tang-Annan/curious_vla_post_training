#!/usr/bin/env python3
"""Rank frozen training tokens by Failure-Aware Learnability Sampling."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Training manifest contains duplicate tokens.")
    return tokens


def build_ranking(rollouts: Path, tokens: list[str], expected_rollouts: int) -> list[dict[str, float | str | int]]:
    allowed = set(tokens)
    rewards: dict[str, list[float]] = defaultdict(list)
    with rollouts.open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            token = str(row["token"])
            if token in allowed:
                reward = row["pdms_scaled"] if "pdms_scaled" in row else row["overall"]
                rewards[token].append(float(reward))

    missing = [token for token in tokens if len(rewards[token]) != expected_rollouts]
    if missing:
        raise ValueError(
            f"Expected {expected_rollouts} rollouts for every training token; "
            f"{len(missing)} tokens have different coverage."
        )

    rows = []
    for token in tokens:
        values = rewards[token]
        mean_reward = statistics.fmean(values)
        max_reward = max(values)
        difficulty = 1.0 - mean_reward
        headroom = max_reward - mean_reward
        rows.append(
            {
                "token": token,
                "n": len(values),
                "reward_mean": mean_reward,
                "reward_std": statistics.stdev(values),
                "reward_max": max_reward,
                "difficulty": difficulty,
                "headroom": headroom,
                "learnability": difficulty * headroom,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["learnability"]), str(row["token"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, action="append", required=True)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    args = parser.parse_args()

    tokens = load_manifest(args.train_manifest)
    ranking = build_ranking(args.rollouts, tokens, args.expected_rollouts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fals_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)

    for budget in args.budget:
        if budget <= 0 or budget > len(ranking):
            raise ValueError(f"Budget must be in [1, {len(ranking)}], got {budget}.")
        selected = "\n".join(str(row["token"]) for row in ranking[:budget]) + "\n"
        (args.output_dir / f"fals_top_{budget}.txt").write_text(selected, encoding="utf-8")


if __name__ == "__main__":
    main()
