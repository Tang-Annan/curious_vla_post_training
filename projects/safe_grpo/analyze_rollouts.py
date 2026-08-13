#!/usr/bin/env python3
"""Summarize grouped Curious-VLA rollout JSONL logs."""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def pairwise_distance(rows: list[dict], point_index: int | None = None) -> float | None:
    valid = [np.asarray(row["poses"], dtype=float) for row in rows if row.get("parsed_ok", True)]
    distances = []
    for first, second in combinations(valid, 2):
        if first.shape != second.shape or first.ndim != 2 or first.shape[1] < 2:
            continue
        if point_index is None:
            distances.append(float(np.linalg.norm(first[:, :2] - second[:, :2], axis=1).mean()))
        else:
            distances.append(float(np.linalg.norm(first[point_index, :2] - second[point_index, :2])))
    return float(np.mean(distances)) if distances else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--std-threshold", type=float, default=0.05)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = defaultdict(list)
    with args.jsonl.open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            groups[row["token"]].append(row)

    summaries = []
    for token, rows in groups.items():
        rewards = np.asarray(
            [row["pdms_scaled"] if "pdms_scaled" in row else row["overall"] for row in rows], dtype=float
        )
        summaries.append(
            {
                "token": token,
                "n": len(rows),
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "reward_min": float(rewards.min()),
                "reward_max": float(rewards.max()),
                "headroom": float(rewards.max() - rewards.mean()),
                "parse_success_rate": float(np.mean([bool(row.get("parsed_ok", True)) for row in rows])),
                "pairwise_ade": pairwise_distance(rows),
                "pairwise_fde": pairwise_distance(rows, -1),
            }
        )

    stds = np.asarray([row["reward_std"] for row in summaries], dtype=float)
    report = {
        "groups": len(summaries),
        "rollouts": sum(row["n"] for row in summaries),
        "zero_signal_ratio": float(np.mean(stds < args.std_threshold)) if len(stds) else None,
        "std_threshold": args.std_threshold,
        "response_length_percentiles": {
            f"p{percentile}": float(np.percentile(lengths, percentile))
            for percentile in (50, 90, 95, 99)
        }
        if len(lengths := np.asarray([
            row["response_length"] for rows in groups.values() for row in rows if "response_length" in row
        ], dtype=float))
        else None,
        "parse_success_rate": float(
            np.mean([bool(row.get("parsed_ok", True)) for rows in groups.values() for row in rows])
        )
        if groups
        else None,
        "group_metrics": summaries,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
