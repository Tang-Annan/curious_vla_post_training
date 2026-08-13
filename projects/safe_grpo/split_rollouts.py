#!/usr/bin/env python3
"""Split a mixed training/final-validation rollout log by frozen manifests."""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Manifest contains duplicate tokens: {path}")
    return tokens


def split_rollouts(
    source: Path,
    train_manifest: Path,
    dev_manifest: Path,
    expected_train_rollouts: int,
    expected_dev_rollouts: int,
) -> tuple[list[dict], list[dict]]:
    train_tokens = set(load_manifest(train_manifest))
    dev_tokens = set(load_manifest(dev_manifest))
    overlap = train_tokens & dev_tokens
    if overlap:
        raise ValueError(f"Train/dev manifests overlap on {len(overlap)} tokens.")

    train_rows = []
    dev_rows = []
    unknown_tokens = set()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            token = row.get("token")
            if token in train_tokens:
                train_rows.append(row)
            elif token in dev_tokens:
                dev_rows.append(row)
            else:
                unknown_tokens.add(str(token))

    if unknown_tokens:
        raise ValueError(f"Rollout log contains {len(unknown_tokens)} tokens outside both manifests.")

    for label, rows, tokens, expected in (
        ("train", train_rows, train_tokens, expected_train_rollouts),
        ("dev", dev_rows, dev_tokens, expected_dev_rollouts),
    ):
        counts = Counter(row["token"] for row in rows)
        mismatched = [token for token in tokens if counts[token] != expected]
        if mismatched:
            raise ValueError(
                f"Expected {expected} {label} rollouts per token; "
                f"{len(mismatched)} of {len(tokens)} tokens have different coverage."
            )

    return train_rows, dev_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--expected-train-rollouts", type=int, required=True)
    parser.add_argument("--expected-dev-rollouts", type=int, required=True)
    args = parser.parse_args()

    train_rows, dev_rows = split_rollouts(
        args.source,
        args.train_manifest,
        args.dev_manifest,
        args.expected_train_rollouts,
        args.expected_dev_rollouts,
    )
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.dev_output, dev_rows)
    print(json.dumps({"train_rows": len(train_rows), "dev_rows": len(dev_rows)}, indent=2))


if __name__ == "__main__":
    main()
