#!/usr/bin/env python3
"""Analyze frozen-checkpoint four-rollout persistent failures for R3."""

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(
    rollouts: Path,
    proxy_manifest: Path,
    full_manifest: Path,
    expected_rollouts: int = 4,
    minimum_persistent: int = 100,
    selection_limit: int = 200,
) -> tuple[list[str], list[str], dict]:
    proxy_tokens = load_manifest(proxy_manifest)
    full_tokens = load_manifest(full_manifest)
    if not set(proxy_tokens) <= set(full_tokens):
        raise ValueError("Proxy manifest contains tokens outside the frozen full manifest.")

    allowed = set(proxy_tokens)
    groups: dict[str, list[dict]] = defaultdict(list)
    with rollouts.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            token = str(row["token"])
            if token not in allowed:
                raise ValueError(f"Rollout token is outside the proxy manifest: {token}")
            groups[token].append(row)

    mismatched = [token for token in proxy_tokens if len(groups[token]) != expected_rollouts]
    if mismatched:
        raise ValueError(f"Expected {expected_rollouts} rollouts per token; {len(mismatched)} tokens differ.")

    all_unsafe = {
        token for token in proxy_tokens if all(float(row.get("safe", 0.0)) == 0.0 for row in groups[token])
    }
    max_pdms_zero = {
        token for token in proxy_tokens if max(float(row["pdms_scaled"]) for row in groups[token]) == 0.0
    }
    persistent = [token for token in proxy_tokens if token in all_unsafe or token in max_pdms_zero]
    selected = persistent[:selection_limit]
    report = {
        "proxy_tokens": len(proxy_tokens),
        "full_manifest_tokens": len(full_tokens),
        "expected_rollouts_per_token": expected_rollouts,
        "rollout_rows": sum(len(rows) for rows in groups.values()),
        "all_unsafe_tokens": len(all_unsafe),
        "max_pdms_scaled_zero_tokens": len(max_pdms_zero),
        "persistent_failure_tokens": len(persistent),
        "persistent_failure_lower_bound_full_manifest": len(persistent) / len(full_tokens),
        "minimum_persistent_tokens": minimum_persistent,
        "selected_tokens": len(selected),
        "selection_limit": selection_limit,
        "gate_passed": len(persistent) >= minimum_persistent,
        "input_sha256": {
            "rollouts": file_sha256(rollouts),
            "proxy_manifest": file_sha256(proxy_manifest),
            "full_manifest": file_sha256(full_manifest),
        },
    }
    return persistent, selected, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--proxy-manifest", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--persistent-output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--minimum-persistent", type=int, default=100)
    parser.add_argument("--selection-limit", type=int, default=200)
    args = parser.parse_args()

    persistent, selected, report = analyze(
        args.rollouts,
        args.proxy_manifest,
        args.full_manifest,
        args.expected_rollouts,
        args.minimum_persistent,
        args.selection_limit,
    )
    args.persistent_output.write_text("".join(f"{token}\n" for token in persistent), encoding="utf-8")
    args.selected_output.write_text("".join(f"{token}\n" for token in selected), encoding="utf-8")
    report["output_sha256"] = {
        "persistent_manifest": file_sha256(args.persistent_output),
        "selected_manifest": file_sha256(args.selected_output),
    }
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
