from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import pickle
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow.parquet as pq


NUM_HISTORY_FRAMES = 4
NUM_FUTURE_FRAMES = 10
NUM_FRAMES = NUM_HISTORY_FRAMES + NUM_FUTURE_FRAMES
ALLOWED_PICKLE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class NavsimUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")
        return getattr(importlib.import_module(module), name)


def load_navsim_log(handle: BinaryIO) -> list[dict[str, Any]]:
    frames = NavsimUnpickler(handle).load()
    if not isinstance(frames, list) or any(not isinstance(frame, dict) for frame in frames):
        raise ValueError("NAVSIM log must contain a list of frame dictionaries")
    return frames


def eligible_centers(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    centers = []
    for start in range(0, len(frames), NUM_FRAMES):
        window = frames[start : start + NUM_FRAMES]
        if len(window) < NUM_FRAMES:
            continue
        center = window[NUM_HISTORY_FRAMES - 1]
        if not center.get("roadblock_ids"):
            continue
        if not isinstance(center.get("token"), str):
            raise ValueError("Eligible NAVSIM frame is missing a string token")
        centers.append(center)
    return centers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0, "mean": 0.0}
    return {
        "min": ordered[0],
        "p25": percentile(ordered, 0.25),
        "median": statistics.median(ordered),
        "p75": percentile(ordered, 0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set, str, bytes)):
        return len(value) > 0
    return True


def parse_model_hashes(path: Path) -> dict[str, str]:
    hashes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        hashes[Path(filename).name] = digest
    return hashes


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    table = pq.read_table(args.sft_parquet, columns=["answer"])
    parquet_tokens = [answer["token"] for answer in table.column("answer").to_pylist()]
    parquet_token_set = set(parquet_tokens)

    master_tokens = []
    master_logs = []
    intent_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    with args.master_index.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"token", "log_name", "intent", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Master index is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            master_tokens.append(row["token"])
            master_logs.append(row["log_name"])
            intent_counts[row["intent"]] += 1
            split_counts[row["split"]] += 1

    master_token_set = set(master_tokens)
    sft_log_set = set(master_logs)
    raw_log_paths = sorted(args.navsim_logs.rglob("*.pkl"))
    raw_log_names = [path.stem for path in raw_log_paths]
    raw_log_set = set(raw_log_names)
    unseen_paths = [path for path in raw_log_paths if path.stem not in sft_log_set]

    unseen_tokens: list[str] = []
    scenes_per_log = []
    frames_per_log = []
    top_level_coverage: Counter[str] = Counter()
    nested_presence: dict[str, Counter[str]] = {}
    nested_nonempty: dict[str, Counter[str]] = {}
    month_counts: Counter[str] = Counter()
    location_counts: Counter[str] = Counter()
    unreadable_logs = 0

    for path in unseen_paths:
        try:
            with path.open("rb") as handle:
                frames = load_navsim_log(handle)
        except (OSError, EOFError, pickle.UnpicklingError, ValueError):
            unreadable_logs += 1
            continue

        centers = eligible_centers(frames)
        frames_per_log.append(len(frames))
        scenes_per_log.append(len(centers))
        for center in centers:
            unseen_tokens.append(center["token"])
            for key, value in center.items():
                if value is not None:
                    top_level_coverage[key] += 1
                if isinstance(value, dict):
                    presence = nested_presence.setdefault(key, Counter())
                    nonempty = nested_nonempty.setdefault(key, Counter())
                    for nested_key, nested_value in value.items():
                        presence[nested_key] += 1
                        if has_content(nested_value):
                            nonempty[nested_key] += 1

            log_name = center.get("log_name")
            if isinstance(log_name, str):
                match = re.match(r"^(\d{4}\.\d{2})", log_name)
                if match:
                    month_counts[match.group(1)] += 1
            for key in ("map_location", "map_name", "location"):
                value = center.get(key)
                if isinstance(value, str) and value:
                    location_counts[value] += 1

    unseen_token_set = set(unseen_tokens)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "sft_token_blacklist.txt").write_text(
        "".join(f"{token}\n" for token in sorted(parquet_token_set)), encoding="utf-8"
    )
    (output_dir / "sft_log_blacklist.txt").write_text(
        "".join(f"{log_name}\n" for log_name in sorted(sft_log_set)), encoding="utf-8"
    )

    report = {
        "version": "dataset_v3_sft_unseen_d0i",
        "source_commit": args.source_commit,
        "scene_filter": {
            "num_history_frames": NUM_HISTORY_FRAMES,
            "num_future_frames": NUM_FUTURE_FRAMES,
            "frame_interval": NUM_FRAMES,
            "has_route": True,
        },
        "inputs": {
            "sft_parquet_sha256": sha256_file(args.sft_parquet),
            "master_index_sha256": sha256_file(args.master_index),
            "model_hash_record_sha256": sha256_file(args.model_hash_record),
            "model_hashes": parse_model_hashes(args.model_hash_record),
        },
        "sft_provenance": {
            "parquet_rows": len(parquet_tokens),
            "parquet_unique_tokens": len(parquet_token_set),
            "parquet_duplicate_tokens": len(parquet_tokens) - len(parquet_token_set),
            "master_rows": len(master_tokens),
            "master_unique_tokens": len(master_token_set),
            "master_unique_logs": len(sft_log_set),
            "tokens_missing_from_master": len(parquet_token_set - master_token_set),
            "tokens_extra_in_master": len(master_token_set - parquet_token_set),
            "intent_counts": dict(sorted(intent_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
        },
        "raw_logs": {
            "files": len(raw_log_paths),
            "unique_names": len(raw_log_set),
            "duplicate_names": len(raw_log_names) - len(raw_log_set),
            "sft_logs_missing_from_raw": len(sft_log_set - raw_log_set),
            "sft_unseen_logs": len(raw_log_set - sft_log_set),
            "unreadable_sft_unseen_logs": unreadable_logs,
        },
        "sft_unseen_capacity": {
            "eligible_scenes": len(unseen_tokens),
            "unique_tokens": len(unseen_token_set),
            "duplicate_tokens": len(unseen_tokens) - len(unseen_token_set),
            "sft_token_overlap": len(unseen_token_set & parquet_token_set),
            "frames_per_log": distribution(frames_per_log),
            "eligible_scenes_per_log": distribution(scenes_per_log),
            "month_counts": dict(sorted(month_counts.items())),
            "location_counts": dict(sorted(location_counts.items())),
        },
        "tail_field_audit": {
            "top_level_non_null": dict(sorted(top_level_coverage.items())),
            "nested_present": {
                key: dict(sorted(values.items())) for key, values in sorted(nested_presence.items())
            },
            "nested_nonempty": {
                key: dict(sorted(values.items())) for key, values in sorted(nested_nonempty.items())
            },
        },
        "assets": {
            "sensor_files": sum(1 for path in args.sensor_root.rglob("*") if path.is_file()),
            "metric_cache_files": sum(1 for path in args.metric_cache_root.rglob("*") if path.is_file()),
        },
        "outputs": {
            "sft_token_blacklist": "sft_token_blacklist.txt",
            "sft_log_blacklist": "sft_log_blacklist.txt",
        },
    }
    (output_dir / "inventory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "COMPLETE").touch()
    (output_dir / "exit_code").write_text("0\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the aggregate Dataset V3 D0I inventory.")
    parser.add_argument("--sft-parquet", type=Path, required=True)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--navsim-logs", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--metric-cache-root", type=Path, required=True)
    parser.add_argument("--model-hash-record", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = build_inventory(parse_args())
    print(
        json.dumps(
            {
                "sft_unique_logs": report["sft_provenance"]["master_unique_logs"],
                "sft_unseen_logs": report["raw_logs"]["sft_unseen_logs"],
                "sft_unseen_unique_tokens": report["sft_unseen_capacity"]["unique_tokens"],
                "sensor_files": report["assets"]["sensor_files"],
                "metric_cache_files": report["assets"]["metric_cache_files"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
