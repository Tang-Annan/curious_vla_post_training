#!/usr/bin/env python3
"""Freeze train-only proxy candidates for the R3 persistent-failure gate."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Manifest contains duplicate tokens: {path}")
    return tokens


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(rollouts: Path, manifest: Path, expected_rollouts: int = 2) -> tuple[list[str], dict]:
    tokens = load_manifest(manifest)
    allowed = set(tokens)
    groups: dict[str, list[dict]] = defaultdict(list)
    with rollouts.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            token = str(row["token"])
            if token not in allowed:
                raise ValueError(f"Rollout token is outside the frozen manifest: {token}")
            groups[token].append(row)

    mismatched = [token for token in tokens if len(groups[token]) != expected_rollouts]
    if mismatched:
        raise ValueError(f"Expected {expected_rollouts} rollouts per token; {len(mismatched)} tokens differ.")

    all_unsafe = {
        token for token in tokens if all(float(row.get("safe", 0.0)) == 0.0 for row in groups[token])
    }
    max_pdms_zero = {
        token for token in tokens if max(float(row["pdms_scaled"]) for row in groups[token]) == 0.0
    }
    candidates = [token for token in tokens if token in all_unsafe or token in max_pdms_zero]
    report = {
        "source": "e2_training_rollouts_proxy_only",
        "manifest_tokens": len(tokens),
        "expected_rollouts_per_token": expected_rollouts,
        "rollout_rows": sum(len(rows) for rows in groups.values()),
        "all_unsafe_tokens": len(all_unsafe),
        "max_pdms_scaled_zero_tokens": len(max_pdms_zero),
        "proxy_candidates": len(candidates),
        "proxy_ratio": len(candidates) / len(tokens),
        "input_sha256": {
            "rollouts": file_sha256(rollouts),
            "manifest": file_sha256(manifest),
        },
    }
    return candidates, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=2)
    parser.add_argument("--expected-candidates", type=int)
    args = parser.parse_args()

    candidates, report = prepare(args.rollouts, args.manifest, args.expected_rollouts)
    if args.expected_candidates is not None and len(candidates) != args.expected_candidates:
        raise ValueError(f"Expected {args.expected_candidates} proxy candidates, got {len(candidates)}.")
    args.output_manifest.write_text("".join(f"{token}\n" for token in candidates), encoding="utf-8")
    report["output_manifest_sha256"] = file_sha256(args.output_manifest)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
