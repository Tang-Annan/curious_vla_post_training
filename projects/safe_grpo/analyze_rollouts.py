#!/usr/bin/env python3
"""Summarize grouped Curious-VLA rollout JSONL logs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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
        rewards = np.asarray([row["overall"] for row in rows], dtype=float)
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
            }
        )

    stds = np.asarray([row["reward_std"] for row in summaries], dtype=float)
    report = {
        "groups": len(summaries),
        "rollouts": sum(row["n"] for row in summaries),
        "zero_signal_ratio": float(np.mean(stds < args.std_threshold)) if len(stds) else None,
        "std_threshold": args.std_threshold,
        "group_metrics": summaries,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
