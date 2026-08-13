#!/usr/bin/env python3
"""Summarize grouped Curious-VLA rollout JSONL logs."""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


AGGREGATE_METRICS = (
    "pdms_scaled",
    "pdms",
    "safe",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
    "reward_latency_ms",
)


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


def load_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Manifest contains duplicate tokens.")
    return tokens


def analyze(
    jsonl: Path,
    std_threshold: float,
    manifest: Path | None = None,
    expected_rollouts: int | None = None,
    max_response_length: int = 512,
) -> dict:
    allowed = set(load_manifest(manifest)) if manifest is not None else None
    groups: dict[str, list[dict]] = defaultdict(list)
    unknown_rows = 0
    with jsonl.open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            token = str(row["token"])
            if allowed is not None and token not in allowed:
                unknown_rows += 1
                continue
            groups[token].append(row)

    if allowed is not None:
        missing = allowed - groups.keys()
        if missing:
            raise ValueError(f"Manifest coverage is missing {len(missing)} tokens.")
        if unknown_rows:
            raise ValueError(f"Rollout file contains {unknown_rows} rows outside the manifest.")

    if expected_rollouts is not None:
        mismatched = [token for token, rows in groups.items() if len(rows) != expected_rollouts]
        if mismatched:
            raise ValueError(
                f"Expected {expected_rollouts} rollouts per token; {len(mismatched)} tokens have different coverage."
            )

    summaries = []
    for token, rows in groups.items():
        rewards = np.asarray(
            [row["pdms_scaled"] if "pdms_scaled" in row else row["overall"] for row in rows], dtype=float
        )
        safe_values = [float(row["safe"]) for row in rows if "safe" in row]
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
                "safe_rate": float(np.mean(safe_values)) if safe_values else None,
                "pairwise_ade": pairwise_distance(rows),
                "pairwise_fde": pairwise_distance(rows, -1),
            }
        )

    stds = np.asarray([row["reward_std"] for row in summaries], dtype=float)
    all_rows = [row for rows in groups.values() for row in rows]
    rewards = np.asarray(
        [
            row["pdms_scaled"] if "pdms_scaled" in row else row["overall"]
            for row in all_rows
        ],
        dtype=float,
    )
    headrooms = np.asarray([row["headroom"] for row in summaries], dtype=float)
    pairwise_ades = np.asarray(
        [row["pairwise_ade"] for row in summaries if row["pairwise_ade"] is not None], dtype=float
    )
    pairwise_fdes = np.asarray(
        [row["pairwise_fde"] for row in summaries if row["pairwise_fde"] is not None], dtype=float
    )
    metric_means = {
        key: float(np.mean([float(row[key]) for row in all_rows if key in row]))
        for key in AGGREGATE_METRICS
        if any(key in row for row in all_rows)
    }
    lengths = np.asarray(
        [row["response_length"] for row in all_rows if "response_length" in row],
        dtype=float,
    )
    return {
        "groups": len(summaries),
        "rollouts": sum(row["n"] for row in summaries),
        "exact_zero_std_ratio": float(np.mean(stds == 0.0)) if len(stds) else None,
        "low_nonzero_std_ratio": float(np.mean((stds > 0.0) & (stds < std_threshold))) if len(stds) else None,
        "below_threshold_std_ratio": float(np.mean(stds < std_threshold)) if len(stds) else None,
        "std_threshold": std_threshold,
        "reward_mean": float(rewards.mean()) if len(rewards) else None,
        "reward_std": float(rewards.std()) if len(rewards) else None,
        "headroom_mean": float(headrooms.mean()) if len(headrooms) else None,
        "pairwise_ade_mean": float(pairwise_ades.mean()) if len(pairwise_ades) else None,
        "pairwise_fde_mean": float(pairwise_fdes.mean()) if len(pairwise_fdes) else None,
        "response_length_percentiles": {
            f"p{percentile}": float(np.percentile(lengths, percentile)) for percentile in (50, 90, 95, 99)
        }
        if len(lengths)
        else None,
        "response_length_mean": float(lengths.mean()) if len(lengths) else None,
        "clipped_responses": int(np.sum(lengths >= max_response_length)) if len(lengths) else None,
        "parse_success_rate": float(
            np.mean([bool(row.get("parsed_ok", True)) for row in all_rows])
        )
        if groups
        else None,
        "safe_rate": metric_means.get("safe"),
        "metric_means": metric_means,
        "group_metrics": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--std-threshold", type=float, default=0.05)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-rollouts", type=int)
    parser.add_argument("--max-response-length", type=int, default=512)
    args = parser.parse_args()
    report = analyze(
        args.jsonl,
        args.std_threshold,
        args.manifest,
        args.expected_rollouts,
        args.max_response_length,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
