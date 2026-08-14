#!/usr/bin/env python3
"""Build a deterministic final-set manifest from previously unused RL rows."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


def load_tokens(path: Path) -> set[str]:
    tokens = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not tokens:
        raise ValueError(f"Manifest is empty: {path}")
    return tokens


def log_name_from_image(image: str) -> str:
    parts = Path(image).parts
    if len(parts) < 3 or parts[-2] != "CAM_F0":
        raise ValueError(f"Unexpected single-view image path: {image}")
    return parts[-3]


def select_rows(
    rows: Iterable[dict], excluded: set[str], count: int, salt: str
) -> list[dict[str, str]]:
    candidates: dict[str, dict[str, str]] = {}
    for row in rows:
        token = str(row["answer"]["token"])
        if token in excluded:
            continue
        if token in candidates:
            raise ValueError(f"Duplicate token in source data: {token}")
        images = row["images"]
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError(f"Expected one image path for token {token}")
        candidates[token] = {
            "token": token,
            "log_name": log_name_from_image(images[0]),
            "image": images[0],
        }

    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible tokens for requested count {count}")
    return sorted(
        candidates.values(),
        key=lambda row: (hashlib.sha256(f"{salt}\0{row['token']}".encode()).hexdigest(), row["token"]),
    )[:count]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-data", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=566)
    parser.add_argument("--salt", default="safe_preference_final_v2_20260815")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")

    import pyarrow.parquet as pq

    excluded = set().union(*(load_tokens(path) for path in args.exclude_manifest))
    table = pq.read_table(args.rl_data, columns=["images", "answer"])
    selected = select_rows(table.to_pylist(), excluded, args.count, args.salt)

    for row in selected:
        image_path = args.image_root / row["image"]
        log_path = args.logs_root / f"{row['log_name']}.pkl"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for token {row['token']}: {image_path}")
        if not log_path.is_file():
            raise FileNotFoundError(f"Missing log for token {row['token']}: {log_path}")

    args.output_dir.mkdir(parents=True)
    token_path = args.output_dir / "final_v2_tokens.txt"
    csv_path = args.output_dir / "final_v2_log_tokens.csv"
    token_path.write_text("".join(f"{row['token']}\n" for row in selected), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["log_name", "token"])
        writer.writeheader()
        writer.writerows({"log_name": row["log_name"], "token": row["token"]} for row in selected)

    report = {
        "source_rows": table.num_rows,
        "excluded_tokens": len(excluded),
        "eligible_tokens": table.num_rows - len(excluded),
        "selected_tokens": len(selected),
        "unique_logs": len({row["log_name"] for row in selected}),
        "salt": args.salt,
        "overlap_with_excluded": len({row["token"] for row in selected} & excluded),
        "input_sha256": {
            "rl_data": sha256(args.rl_data),
            **{str(path): sha256(path) for path in args.exclude_manifest},
        },
        "output_sha256": {
            token_path.name: sha256(token_path),
            csv_path.name: sha256(csv_path),
        },
    }
    (args.output_dir / "final_v2_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
