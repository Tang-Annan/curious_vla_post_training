#!/usr/bin/env python3
"""Build the parquet-only, log-disjoint Dataset V2 inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


INTENTS = ("straight", "left", "right")
INTENT_PATTERN = re.compile(r"Current high-level intent \(string\):\s*([^\n]+)", re.IGNORECASE)
SOURCE_IMAGE_PREFIX = "navsim/trainval_sensor_blobs/trainval/"
HORIZON_5 = "optimal future 5-second trajectory"
HORIZON_4 = "optimal future 4-second trajectory"


@dataclass(frozen=True)
class Row:
    index: int
    token: str
    log_name: str
    intent: str
    source_image: str
    v2_image: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode("utf-8")).hexdigest()


def normalize_intent(text: str) -> str:
    match = INTENT_PATTERN.search(text)
    if not match:
        raise ValueError("Prompt is missing the current high-level intent")
    intent = match.group(1).strip().lower()
    if "straight" in intent:
        return "straight"
    if "left" in intent:
        return "left"
    if "right" in intent:
        return "right"
    raise ValueError(f"Unsupported navigation intent: {intent!r}")


def log_from_image(image: str) -> str:
    marker = SOURCE_IMAGE_PREFIX
    if not image.startswith(marker):
        raise ValueError(f"Unexpected image path prefix: {image!r}")
    remainder = image[len(marker) :]
    log_name = remainder.split("/", 1)[0]
    if not log_name:
        raise ValueError(f"Image path has no log name: {image!r}")
    return log_name


def v2_image_path(image: str, version: str) -> str:
    return f"{version}/sensor_blobs/trainval/{image[len(SOURCE_IMAGE_PREFIX):]}"


def load_rows(input_parquet: Path, logs_root: Path, version: str) -> tuple[list[Row], int]:
    parquet = pq.ParquetFile(input_parquet)
    schema_columns = set(parquet.schema_arrow.names)
    required = {"images", "problem", "answer"}
    missing = required - schema_columns
    if missing:
        raise ValueError(f"Input parquet is missing columns: {sorted(missing)}")

    rows: list[Row] = []
    tokens: set[str] = set()
    horizon_count = 0
    index = 0
    for batch in parquet.iter_batches(columns=["images", "problem", "answer"], batch_size=1024):
        for raw in batch.to_pylist():
            images = raw["images"]
            answer = raw["answer"]
            if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
                raise ValueError(f"Row {index} must contain exactly one image path")
            if not isinstance(answer, dict) or not answer.get("token"):
                raise ValueError(f"Row {index} has no answer.token")
            token = str(answer["token"])
            if token in tokens:
                raise ValueError(f"Duplicate token in input parquet: {token}")
            tokens.add(token)
            source_image = images[0]
            log_name = log_from_image(source_image)
            if not (logs_root / f"{log_name}.pkl").is_file():
                raise ValueError(f"Source log is missing for token {token}: {log_name}")
            problem = str(raw["problem"])
            if HORIZON_5 not in problem:
                raise ValueError(f"Row {index} does not contain the expected 5-second prompt")
            horizon_count += 1
            rows.append(
                Row(
                    index=index,
                    token=token,
                    log_name=log_name,
                    intent=normalize_intent(problem),
                    source_image=source_image,
                    v2_image=v2_image_path(source_image, version),
                )
            )
            index += 1
    return rows, horizon_count


def load_tokens(paths: Iterable[Path]) -> set[str]:
    tokens: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            token = line.strip()
            if token:
                tokens.add(token)
    return tokens


def largest_remainder(counts: Counter[str], total: int) -> dict[str, int]:
    source_total = sum(counts.values())
    if source_total <= 0:
        raise ValueError("Cannot allocate quotas from an empty intent distribution")
    raw = {intent: counts[intent] * total / source_total for intent in INTENTS}
    quota = {intent: math.floor(raw[intent]) for intent in INTENTS}
    remaining = total - sum(quota.values())
    order = sorted(INTENTS, key=lambda intent: (raw[intent] - quota[intent], intent), reverse=True)
    for intent in order[:remaining]:
        quota[intent] += 1
    return quota


def select_rows(
    rows: list[Row],
    *,
    total: int,
    quotas: dict[str, int],
    seed: int,
    namespace: str,
    max_per_log: int,
    min_logs: int,
) -> list[Row]:
    if total != sum(quotas.values()):
        raise ValueError(f"Quota sum {sum(quotas.values())} does not equal requested total {total}")
    ranked_by_intent = {
        intent: sorted(
            (row for row in rows if row.intent == intent),
            key=lambda row: stable_key(seed, namespace, row.token),
        )
        for intent in INTENTS
    }
    selected: list[Row] = []
    counts = Counter[str]()
    per_log = Counter[str]()
    positions = Counter[str]()
    while len(selected) < total:
        progress = False
        order = sorted(INTENTS, key=lambda intent: (-quotas[intent] + counts[intent], intent))
        for intent in order:
            if counts[intent] >= quotas[intent]:
                continue
            candidates = ranked_by_intent[intent]
            while positions[intent] < len(candidates) and per_log[candidates[positions[intent]].log_name] >= max_per_log:
                positions[intent] += 1
            if positions[intent] >= len(candidates):
                continue
            row = candidates[positions[intent]]
            positions[intent] += 1
            selected.append(row)
            counts[intent] += 1
            per_log[row.log_name] += 1
            progress = True
        if not progress:
            break
    if len(selected) != total:
        available = Counter(row.intent for row in rows)
        raise ValueError(
            f"Unable to satisfy {namespace} quotas: selected={len(selected)} requested={total} "
            f"selected_by_intent={dict(counts)} requested_by_intent={quotas} available_by_intent={dict(available)}"
        )
    if len(per_log) < min_logs:
        raise ValueError(f"{namespace} covers only {len(per_log)} logs; minimum is {min_logs}")
    if max(per_log.values(), default=0) > max_per_log:
        raise AssertionError(f"{namespace} exceeded per-log cap")
    return sorted(selected, key=lambda row: row.index)


def split_new_logs(
    rows: list[Row],
    seed: int,
    dev_quotas: dict[str, int],
    final_quotas: dict[str, int],
) -> tuple[set[str], set[str]]:
    all_logs = {row.log_name for row in rows}
    ranked_logs = sorted(all_logs, key=lambda value: stable_key(seed, "dev-final-log", value))
    for dev_count in range(150, len(ranked_logs) - 79):
        dev_logs = set(ranked_logs[:dev_count])
        final_logs = set(ranked_logs[dev_count:])
        dev_pool = [row for row in rows if row.log_name in dev_logs]
        final_pool = [row for row in rows if row.log_name in final_logs]
        try:
            select_rows(
                dev_pool,
                total=sum(dev_quotas.values()),
                quotas=dev_quotas,
                seed=seed,
                namespace="dev",
                max_per_log=15,
                min_logs=150,
            )
            select_rows(
                final_pool,
                total=sum(final_quotas.values()),
                quotas=final_quotas,
                seed=seed,
                namespace="final",
                max_per_log=10,
                min_logs=80,
            )
        except ValueError:
            continue
        return dev_logs, final_logs
    raise ValueError("Unable to find a deterministic dev/final log partition satisfying all gates")


def write_tokens(path: Path, rows: list[Row]) -> None:
    path.write_text("".join(f"{row.token}\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], records: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_parquet(input_parquet: Path, rows: list[Row], path: Path) -> None:
    wanted = {row.index: row for row in rows}
    records: list[dict[str, object]] = []
    index = 0
    parquet = pq.ParquetFile(input_parquet)
    for batch in parquet.iter_batches(columns=["problem"], batch_size=1024):
        for raw in batch.to_pylist():
            row = wanted.get(index)
            if row is not None:
                problem = str(raw["problem"])
                records.append(
                    {
                        "images": [row.v2_image],
                        "problem": problem.replace(HORIZON_5, HORIZON_4),
                        "answer": {"gt": [], "token": row.token},
                    }
                )
            index += 1
    if len(records) != len(rows):
        raise ValueError(f"Selected parquet rows missing: expected={len(rows)} actual={len(records)}")
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


def overlap_report(named_rows: dict[str, list[Row]]) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    names = list(named_rows)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_tokens = {row.token for row in named_rows[left]}
            right_tokens = {row.token for row in named_rows[right]}
            left_logs = {row.log_name for row in named_rows[left]}
            right_logs = {row.log_name for row in named_rows[right]}
            report[f"{left}__{right}"] = {
                "token_overlap": len(left_tokens & right_tokens),
                "log_overlap": len(left_logs & right_logs),
            }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--version", default="dataset_v2_20260825")
    parser.add_argument("--legacy-manifest", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--candidate-size", type=int, default=8000)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--final-size", type=int, default=1000)
    parser.add_argument("--phase1-size", type=int, default=6000)
    parser.add_argument("--random-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase1_size > args.candidate_size:
        raise ValueError("phase1-size cannot exceed candidate-size")
    if args.random_size > args.phase1_size:
        raise ValueError("random-size cannot exceed phase1-size")
    if args.output_data.exists() or args.output_manifest.exists():
        raise SystemExit("Output directory already exists; use a new Dataset V2 version")
    args.output_data.mkdir(parents=True)
    hf_dir = args.output_data / "hf"
    hf_dir.mkdir()
    args.output_manifest.mkdir(parents=True)

    rows, source_horizon_count = load_rows(args.input_parquet, args.logs_root, args.version)
    by_token = {row.token: row for row in rows}
    legacy_tokens = load_tokens(args.legacy_manifest)
    missing_legacy = legacy_tokens - set(by_token)
    if missing_legacy:
        raise ValueError(f"Legacy manifest contains {len(missing_legacy)} tokens absent from source parquet")
    legacy_log_set = {by_token[token].log_name for token in legacy_tokens}
    source_intents = Counter(row.intent for row in rows)
    candidate_pool = [row for row in rows if row.log_name in legacy_log_set and row.token not in legacy_tokens]
    new_pool = [row for row in rows if row.log_name not in legacy_log_set]
    if len(candidate_pool) < args.candidate_size:
        raise ValueError(f"Candidate pool has only {len(candidate_pool)} rows")
    if len(new_pool) < args.dev_size + args.final_size:
        raise ValueError(f"New-log pool has only {len(new_pool)} rows")

    candidate = select_rows(
        candidate_pool,
        total=args.candidate_size,
        quotas=largest_remainder(source_intents, args.candidate_size),
        seed=args.seed,
        namespace="candidate",
        max_per_log=25,
        min_logs=500,
    )
    dev_logs, final_logs = split_new_logs(
        new_pool,
        args.seed,
        largest_remainder(source_intents, args.dev_size),
        largest_remainder(source_intents, args.final_size),
    )
    dev_pool = [row for row in new_pool if row.log_name in dev_logs]
    final_pool = [row for row in new_pool if row.log_name in final_logs]
    dev = select_rows(
        dev_pool,
        total=args.dev_size,
        quotas=largest_remainder(source_intents, args.dev_size),
        seed=args.seed,
        namespace="dev",
        max_per_log=15,
        min_logs=150,
    )
    final = select_rows(
        final_pool,
        total=args.final_size,
        quotas=largest_remainder(source_intents, args.final_size),
        seed=args.seed,
        namespace="final",
        max_per_log=10,
        min_logs=80,
    )
    phase1 = sorted(candidate, key=lambda row: stable_key(args.seed, "phase1", row.token))[: args.phase1_size]
    phase1_tokens = {row.token for row in phase1}
    extension = sorted((row for row in candidate if row.token not in phase1_tokens), key=lambda row: row.index)
    random_rows = select_rows(
        phase1,
        total=args.random_size,
        quotas=largest_remainder(source_intents, args.random_size),
        seed=args.seed,
        namespace="random",
        max_per_log=5,
        min_logs=200,
    )
    named_rows = {"candidate": candidate, "dev": dev, "final_reserve": final}
    overlaps = overlap_report(named_rows)
    random_tokens = {row.token for row in random_rows}
    random_logs = {row.log_name for row in random_rows}
    candidate_tokens = {row.token for row in candidate}
    candidate_logs = {row.log_name for row in candidate}
    if not random_tokens <= candidate_tokens or not random_logs <= candidate_logs:
        raise AssertionError("Random manifest is not contained in the candidate pool")
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise AssertionError(f"Dataset V2 split overlap detected: {overlaps}")

    write_parquet(args.input_parquet, candidate, hf_dir / "train.parquet")
    write_parquet(args.input_parquet, dev, hf_dir / "test.parquet")
    write_tokens(args.output_manifest / "legacy_5656_tokens.txt", [by_token[token] for token in sorted(legacy_tokens)])
    write_tokens(args.output_manifest / "selector_pool_8000.txt", candidate)
    write_tokens(args.output_manifest / "selector_pool_phase1_6000.txt", phase1)
    write_tokens(args.output_manifest / "selector_extension_2000.txt", extension)
    write_tokens(args.output_manifest / "dev_2000.txt", dev)
    write_tokens(args.output_manifest / "final_reserve_1000.txt", final)
    write_tokens(args.output_manifest / "random_1k.txt", random_rows)

    cache_rows = []
    for split, split_rows in (("candidate", candidate), ("dev", dev)):
        cache_rows.extend(
            {"token": row.token, "log_name": row.log_name, "split": split, "image_path": row.v2_image}
            for row in split_rows
        )
    write_csv(args.output_manifest / "cache_10000.csv", ["token", "log_name", "split", "image_path"], cache_rows)
    master_rows = []
    selected_tokens = {row.token: split for split, split_rows in (("candidate", candidate), ("dev", dev), ("final_reserve", final)) for row in split_rows}
    for row in rows:
        master_rows.append(
            {
                "token": row.token,
                "log_name": row.log_name,
                "intent": row.intent,
                "source_image": row.source_image,
                "v2_image": row.v2_image,
                "source_row": row.index,
                "legacy_token": int(row.token in legacy_tokens),
                "legacy_log": int(row.log_name in legacy_log_set),
                "split": selected_tokens.get(row.token, "unused"),
            }
        )
    write_csv(
        args.output_manifest / "master_index.csv",
        ["token", "log_name", "intent", "source_image", "v2_image", "source_row", "legacy_token", "legacy_log", "split"],
        master_rows,
    )

    output_files = sorted(path for path in args.output_data.rglob("*") if path.is_file())
    output_files += sorted(path for path in args.output_manifest.rglob("*") if path.is_file())
    report = {
        "version": args.version,
        "seed": args.seed,
        "builder_sha256": sha256(Path(__file__).resolve()),
        "source": {
            "input_parquet": str(args.input_parquet.resolve()),
            "input_sha256": sha256(args.input_parquet),
            "rows": len(rows),
            "unique_tokens": len(by_token),
            "intent_counts": dict(source_intents),
            "legacy_tokens": len(legacy_tokens),
            "legacy_logs": len(legacy_log_set),
            "candidate_pool_rows": len(candidate_pool),
            "new_log_pool_rows": len(new_pool),
            "candidate_pool_logs": len({row.log_name for row in candidate_pool}),
            "new_log_pool_logs": len({row.log_name for row in new_pool}),
        },
        "splits": {
            name: {
                "rows": len(split_rows),
                "logs": len({row.log_name for row in split_rows}),
                "intent_counts": dict(Counter(row.intent for row in split_rows)),
                "max_rows_per_log": max(Counter(row.log_name for row in split_rows).values()),
            }
            for name, split_rows in {**named_rows, "random": random_rows}.items()
        },
        "overlap_gates": overlaps,
        "prompt_fix": {
            "source_rows_with_5_second_phrase": source_horizon_count,
            "selected_rows_with_4_second_phrase": len(candidate) + len(dev),
            "source_5s_phrase_replaced_in_outputs": True,
        },
        "image_stage_deferred": True,
        "cache_stage_deferred": True,
        "gates": {
            "source_tokens_unique": len(by_token) == len(rows),
            "legacy_tokens_present": not missing_legacy,
            "candidate_dev_final_token_disjoint": all(value["token_overlap"] == 0 for value in overlaps.values()),
            "candidate_dev_final_log_disjoint": all(value["log_overlap"] == 0 for value in overlaps.values()),
            "candidate_size": len(candidate) == args.candidate_size,
            "dev_size": len(dev) == args.dev_size,
            "final_size": len(final) == args.final_size,
            "random_size": len(random_rows) == args.random_size,
            "random_subset_candidate": random_tokens <= candidate_tokens and random_logs <= candidate_logs,
            "prompt_fixed": source_horizon_count == len(rows) and len(candidate) + len(dev) == args.candidate_size + args.dev_size,
            "source_logs_exist": True,
        },
    }
    report["all_gates_passed"] = all(report["gates"].values())
    card = {
        "dataset_version": args.version,
        "seed": args.seed,
        "source": report["source"],
        "splits": report["splits"],
        "artifacts": {
            "train_parquet": f"{args.version}/hf/train.parquet",
            "dev_parquet": f"{args.version}/hf/test.parquet",
            "image_stage": "deferred",
            "metric_cache_stage": "deferred",
            "selector_manifests": ["selector_pool_8000.txt", "selector_pool_phase1_6000.txt", "random_1k.txt"],
            "deferred_selector_manifests": ["adas_1k.txt", "fals_1k.txt"],
        },
        "gates": report["gates"],
        "overlap_gates": report["overlap_gates"],
    }
    (args.output_data / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_manifest / "acceptance_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    output_files = sorted(path for path in args.output_data.rglob("*") if path.is_file()) + sorted(path for path in args.output_manifest.rglob("*") if path.is_file())
    hash_lines = [f"{sha256(path)}  {path.relative_to(args.output_manifest.parent.parent)}" for path in output_files if path.name != "sha256sum.txt"]
    (args.output_manifest / "sha256sum.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    if not report["all_gates_passed"]:
        raise SystemExit(2)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
