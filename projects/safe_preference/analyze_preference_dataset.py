"""Compute the preregistered P1-M pair-capacity gate."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def read_manifest(path: Path) -> set[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Manifest contains duplicate tokens: {path}")
    return set(tokens)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_valid(row: dict) -> bool:
    return bool(row.get("parsed_ok")) and np.asarray(row.get("poses", [])).shape == (8, 3)


def is_safe(row: dict) -> bool:
    return (
        is_valid(row)
        and row.get("no_at_fault_collisions") == 1
        and row.get("drivable_area_compliance") == 1
        and row.get("time_to_collision_within_bound") == 1
    )


def largest_balanced_size(max_pairs: int, pdms_pairs: int, tier_a: int, tier_b: int) -> int:
    for size in range(max_pairs - max_pairs % 5, -1, -5):
        if size <= pdms_pairs and 3 * size // 5 <= tier_a and 2 * size // 5 <= tier_b:
            return size
    return 0


def analyze_pair_capacity(
    rows: list[dict],
    *,
    gap_quantile: float,
    max_pairs: int,
    min_pairs: int,
    expected_rollouts: int,
) -> dict:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row.get("token")].append(row)
    if any(len(group) != expected_rollouts for group in groups.values()):
        raise ValueError("D0 rollout coverage is not exactly four rows per token")

    fully_valid = {
        token: group for token, group in groups.items() if all(is_valid(row) for row in group)
    }
    gaps = {
        token: max(row["pdms_scaled"] for row in group) - min(row["pdms_scaled"] for row in group)
        for token, group in fully_valid.items()
    }
    if not gaps:
        raise ValueError("No fully valid scene is available for the gap quantile")
    gap_values = np.asarray(list(gaps.values()), dtype=float)
    delta = float(np.quantile(gap_values, gap_quantile))
    pdms_pairs = sum(gap >= delta for gap in gaps.values())

    categories = collections.Counter()
    for token, group in groups.items():
        if token not in fully_valid:
            categories["parse_or_shape_failure"] += 1
            continue
        safe_flags = [is_safe(row) for row in group]
        if any(safe_flags) and not all(safe_flags):
            categories["tier_a"] += 1
        elif all(safe_flags) and gaps[token] >= delta:
            categories["tier_b"] += 1
        elif all(safe_flags):
            categories["all_safe_below_delta"] += 1
        else:
            categories["all_unsafe"] += 1

    tier_a = categories["tier_a"]
    tier_b = categories["tier_b"]
    tier_c = len(groups) - tier_a - tier_b
    balanced_size = largest_balanced_size(max_pairs, pdms_pairs, tier_a, tier_b)
    passed = balanced_size >= min_pairs
    return {
        "status": "capacity_passed" if passed else "insufficient_pairs",
        "gate_passed": passed,
        "gap_quantile": gap_quantile,
        "delta": delta,
        "gap_quantiles": {
            f"q{int(quantile * 100):02d}": float(np.quantile(gap_values, quantile))
            for quantile in (0.0, 0.25, 0.5, gap_quantile, 0.75, 1.0)
        },
        "scenes": len(groups),
        "fully_valid_scenes": len(fully_valid),
        "N_P": pdms_pairs,
        "N_A": tier_a,
        "N_B": tier_b,
        "N_C": tier_c,
        "B": balanced_size,
        "tier_a_required": 3 * balanced_size // 5,
        "tier_b_required": 2 * balanced_size // 5,
        "max_pairs": max_pairs,
        "min_pairs": min_pairs,
        "category_counts": dict(sorted(categories.items())),
        "filter_counts": {
            "pdms_gap_pass": pdms_pairs,
            "pdms_gap_below_delta": len(fully_valid) - pdms_pairs,
            "parse_or_shape_failure": len(groups) - len(fully_valid),
        },
        "decision": "build_frozen_datasets" if passed else "close_offline_preference_route",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--p1-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gap-quantile", type=float, default=0.60)
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--min-pairs", type=int, default=500)
    parser.add_argument("--expected-rows", type=int, default=18100)
    parser.add_argument("--expected-scenes", type=int, default=4525)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    source_status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_status:
        raise SystemExit("Source checkout is not clean")
    source_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    inputs = [
        args.rollouts,
        args.train_manifest,
        args.dev_manifest,
        args.heldout_manifest,
        args.p1_audit,
    ]
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")
    if not args.output_dir.is_dir():
        raise SystemExit(f"P1 output directory does not exist: {args.output_dir}")
    outputs = [
        args.output_dir / "preference_stats.json",
        args.output_dir / "p1_m_input_sha256.txt",
        args.output_dir / "p1_m_source_commit.txt",
    ]
    if any(path.exists() for path in outputs):
        raise SystemExit("P1-M output already exists")

    p1_audit = json.loads(args.p1_audit.read_text(encoding="utf-8"))
    if not p1_audit.get("all_gates_passed"):
        raise SystemExit("P1-S audit did not pass")
    rows = read_jsonl(args.rollouts)
    train_tokens = read_manifest(args.train_manifest)
    dev_tokens = read_manifest(args.dev_manifest)
    heldout_tokens = read_manifest(args.heldout_manifest)
    row_tokens = {row.get("token") for row in rows}
    if len(rows) != args.expected_rows or len(row_tokens) != args.expected_scenes:
        raise SystemExit("D0 row or scene count does not match the preregistration")
    if row_tokens != train_tokens or row_tokens & dev_tokens or row_tokens & heldout_tokens:
        raise SystemExit("D0 manifest boundary failed")

    report = analyze_pair_capacity(
        rows,
        gap_quantile=args.gap_quantile,
        max_pairs=args.max_pairs,
        min_pairs=args.min_pairs,
        expected_rollouts=args.expected_rollouts,
    )
    report["data_boundary"] = {
        "train_manifest_exact_match": True,
        "dev_overlap": 0,
        "heldout_overlap": 0,
    }
    (args.output_dir / "preference_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "p1_m_input_sha256.txt").write_text(
        "".join(f"{sha256(path)}  {path.resolve()}\n" for path in inputs), encoding="utf-8"
    )
    (args.output_dir / "p1_m_source_commit.txt").write_text(source_commit + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
