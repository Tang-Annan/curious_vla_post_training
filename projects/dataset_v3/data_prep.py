from __future__ import annotations

import argparse
import binascii
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tarfile
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from projects.dataset_v3.inventory import (
    NUM_FRAMES,
    NUM_FUTURE_FRAMES,
    NUM_HISTORY_FRAMES,
    load_navsim_log,
    parse_model_hashes,
    sha256_file,
)


VERSION = "dataset_v3_controlled_overlap"
PROMPT_VERSION = "navsim_front_4s_v1"
SEED = 20260827
SCREEN_SIZE = 8000
MONITOR_SIZE = 256
RANDOM_SIZE = 2000
SCREEN_PER_LOG_CAP = 8
MONITOR_PER_LOG_CAP = 2
SOURCE_IMAGE_PREFIX = "navsim/trainval_sensor_blobs/trainval/"
V3_IMAGE_PREFIX = f"{VERSION}/sensor_blobs/trainval/"
INTENTS = ("straight", "left", "right")
VEHICLE_DISTANCE_M = 5.0
VRU_DISTANCE_M = 10.0
SELECTIVE_ZIP_REVISION = "7707301e13828b4599b3a0f834b44efed57df90e"
SELECTIVE_ZIP_URL = (
    f"https://hf-mirror.com/datasets/richardyann/navsim-select/resolve/{SELECTIVE_ZIP_REVISION}"
    "/sensor_blobs/trainval.zip"
)
MIRROR_OVERLAP_CHECKS = 64
SELECTIVE_ZIP_SIZE = 148_230_424_017
SELECTIVE_ZIP_DIRECTORY_OFFSET = 148_109_563_035
SELECTIVE_ZIP_WORKERS = 8
ZIP_LOCAL_EXTRA_LIMIT = 65_535


@dataclass(frozen=True)
class SftRow:
    source_row: int
    token: str
    log_name: str
    intent: str
    source_image: str


@dataclass(frozen=True)
class EvalRow:
    token: str
    log_name: str
    intent: str
    source_image: str
    v3_image: str
    problem: str
    map_location: str
    month: str
    min_vehicle_distance_m: float | None
    min_vru_distance_m: float | None
    interaction_tail_flag: bool


def stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def largest_remainder(counts: Counter[str], total: int) -> dict[str, int]:
    source_total = sum(counts.values())
    raw = {intent: counts[intent] * total / source_total for intent in INTENTS}
    quota = {intent: math.floor(raw[intent]) for intent in INTENTS}
    order = sorted(INTENTS, key=lambda intent: (raw[intent] - quota[intent], intent), reverse=True)
    for intent in order[: total - sum(quota.values())]:
        quota[intent] += 1
    return quota


def intent_matched(rows: list[SftRow], size: int, seed: int, namespace: str) -> list[SftRow]:
    quota = largest_remainder(Counter(row.intent for row in rows), size)
    selected = []
    for intent in INTENTS:
        candidates = sorted(
            (row for row in rows if row.intent == intent),
            key=lambda row: stable_key(seed, namespace, row.token),
        )
        if len(candidates) < quota[intent]:
            raise ValueError(f"Insufficient {intent} rows for {namespace}: {len(candidates)} < {quota[intent]}")
        selected.extend(candidates[: quota[intent]])
    return sorted(selected, key=lambda row: stable_key(seed, f"{namespace}-order", row.token))


def load_sft_rows(path: Path) -> list[SftRow]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"token", "log_name", "intent", "source_image", "source_row"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"SFT master index is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for raw in reader:
            rows.append(
                SftRow(
                    source_row=int(raw["source_row"]),
                    token=raw["token"],
                    log_name=raw["log_name"],
                    intent=raw["intent"],
                    source_image=raw["source_image"],
                )
            )
    if len({row.token for row in rows}) != len(rows):
        raise ValueError("SFT master index contains duplicate tokens")
    if any(row.intent not in INTENTS for row in rows):
        raise ValueError("SFT master index contains unsupported intents")
    return rows


def choose_training_rows(
    rows: list[SftRow],
    *,
    seed: int = SEED,
    screen_size: int = SCREEN_SIZE,
    monitor_size: int = MONITOR_SIZE,
    random_size: int = RANDOM_SIZE,
) -> tuple[list[SftRow], list[SftRow], list[SftRow]]:
    by_log: dict[str, list[SftRow]] = defaultdict(list)
    for row in rows:
        by_log[row.log_name].append(row)
    for log_rows in by_log.values():
        log_rows.sort(key=lambda row: stable_key(seed, "within-log", row.token))

    monitor = []
    monitor_logs = set()
    for log_name in sorted(by_log, key=lambda value: stable_key(seed, "monitor-log", value)):
        take = min(MONITOR_PER_LOG_CAP, monitor_size - len(monitor), len(by_log[log_name]))
        if take:
            monitor.extend(by_log[log_name][:take])
            monitor_logs.add(log_name)
        if len(monitor) == monitor_size:
            break
    if len(monitor) != monitor_size:
        raise ValueError(f"Could not allocate {monitor_size} train-monitor rows")

    screen_candidates = [
        row
        for log_name, log_rows in by_log.items()
        if log_name not in monitor_logs
        for row in log_rows[:SCREEN_PER_LOG_CAP]
    ]
    if len(screen_candidates) < screen_size:
        raise ValueError(f"Controlled train pool capacity {len(screen_candidates)} < {screen_size}")
    screen = intent_matched(screen_candidates, screen_size, seed, "screen")
    random_rows = intent_matched(screen, random_size, seed, "random")
    return screen, monitor, random_rows


def eligible_windows(frames: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    windows = []
    for start in range(0, len(frames), NUM_FRAMES):
        window = frames[start : start + NUM_FRAMES]
        if len(window) == NUM_FRAMES and window[NUM_HISTORY_FRAMES - 1].get("roadblock_ids"):
            windows.append(window)
    return windows


def quaternion_yaw(value: Any) -> float:
    w, x, y, z = (float(item) for item in value)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def format_number(value: float) -> str:
    if abs(round(value, 2)) <= 0.01:
        return "0.0"
    return f"{value:+.2f}"


def build_problem(window: list[dict[str, Any]], template: str) -> str:
    center = window[NUM_HISTORY_FRAMES - 1]
    command_index = int(np.argmax(center["driving_command"]))
    commands = ("turn left", "go straight", "turn right", "unknown")
    command = commands[command_index]
    origin_translation = center["ego2global_translation"]
    origin_yaw = quaternion_yaw(center["ego2global_rotation"])
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    statuses = []
    for index, frame in enumerate(window[: NUM_HISTORY_FRAMES - 1]):
        translation = frame["ego2global_translation"]
        dx = float(translation[0] - origin_translation[0])
        dy = float(translation[1] - origin_translation[1])
        x = cos_yaw * dx + sin_yaw * dy
        y = -sin_yaw * dx + cos_yaw * dy
        heading = wrap_angle(quaternion_yaw(frame["ego2global_rotation"]) - origin_yaw)
        statuses.append(
            f"   - t-{NUM_HISTORY_FRAMES - index - 1}: ({format_number(x)}, {format_number(y)}, {format_number(heading)})"
        )
    statuses.append("   - t-0: (0.0, 0.0, 0.0)")
    marker = "Each trajectory point format: (x:float, y:float, heading:float)"
    if marker not in template:
        raise ValueError("Source problem template is missing the trajectory marker")
    suffix = marker + template.split(marker, 1)[1]
    prefix = f"""Suppose you are driving. Let's complete the following tasks step by step.
Input:
- 1 frame of front-view image collected from the ego-vehicle at the present timestep
Picture 1: <image> the front view of the ego-vehicle
- Current high-level intent (string): {command}
- 1.5-second past trajectory(3 steps at 2 Hz): {' '.join(statuses)}
"""
    return (prefix + suffix).replace("5-second", "4-second")


def scene_risk(center: dict[str, Any]) -> tuple[float | None, float | None, bool]:
    vehicle_distances = []
    vru_distances = []
    for name, box in zip(center["anns"]["gt_names"], center["anns"]["gt_boxes"]):
        distance = float(np.hypot(float(box[0]), float(box[1])))
        if str(name) == "vehicle":
            vehicle_distances.append(distance)
        elif str(name) in {"pedestrian", "bicycle"}:
            vru_distances.append(distance)
    vehicle = min(vehicle_distances, default=None)
    vru = min(vru_distances, default=None)
    flag = (vehicle is not None and vehicle <= VEHICLE_DISTANCE_M) or (vru is not None and vru <= VRU_DISTANCE_M)
    return vehicle, vru, flag


def build_eval_rows(log_paths: list[Path], template: str) -> tuple[list[EvalRow], dict[str, dict[str, Any]]]:
    rows = []
    log_summary = {}
    for path in sorted(log_paths):
        with path.open("rb") as handle:
            frames = load_navsim_log(handle)
        log_rows = []
        for window in eligible_windows(frames):
            center = window[NUM_HISTORY_FRAMES - 1]
            command_index = int(np.argmax(center["driving_command"]))
            intent = ("left", "straight", "right", "unknown")[command_index]
            source_image = center["cams"]["CAM_F0"]["data_path"]
            vehicle, vru, flag = scene_risk(center)
            row = EvalRow(
                token=center["token"],
                log_name=path.stem,
                intent=intent,
                source_image=source_image,
                v3_image=V3_IMAGE_PREFIX + source_image,
                problem=build_problem(window, template),
                map_location=center["map_location"],
                month=path.stem[:7],
                min_vehicle_distance_m=vehicle,
                min_vru_distance_m=vru,
                interaction_tail_flag=flag,
            )
            rows.append(row)
            log_rows.append(row)
        minimum = min(
            (
                distance
                for row in log_rows
                for distance in (row.min_vehicle_distance_m, row.min_vru_distance_m)
                if distance is not None
            ),
            default=None,
        )
        risk_count = sum(row.interaction_tail_flag for row in log_rows)
        log_summary[path.stem] = {
            "eligible_scenes": len(log_rows),
            "interaction_scenes": risk_count,
            "interaction_rate": risk_count / len(log_rows) if log_rows else 0.0,
            "minimum_actor_distance_m": minimum,
        }
    if len({row.token for row in rows}) != len(rows):
        raise ValueError("Evaluation reserve contains duplicate tokens")
    return rows, log_summary


def assign_eval_logs(log_summary: dict[str, dict[str, Any]], seed: int = SEED) -> dict[str, str]:
    nonzero = [log_name for log_name, values in log_summary.items() if values["eligible_scenes"]]
    ordered = sorted(
        nonzero,
        key=lambda log_name: (
            -log_summary[log_name]["interaction_rate"],
            -log_summary[log_name]["interaction_scenes"],
            log_summary[log_name]["minimum_actor_distance_m"] or math.inf,
            stable_key(seed, "tail-rank", log_name),
        ),
    )
    tail_logs = set(ordered[: len(ordered) // 2])
    families = {
        "tail": [log_name for log_name in nonzero if log_name in tail_logs],
        "natural": [log_name for log_name in nonzero if log_name not in tail_logs],
    }
    assignment = {}
    bin_state = {}
    for family, logs in families.items():
        state = {"dev": [0, 0], "final": [0, 0]}
        for log_name in sorted(
            logs,
            key=lambda value: (-log_summary[value]["eligible_scenes"], stable_key(seed, f"{family}-size", value)),
        ):
            destination = min(
                ("dev", "final"),
                key=lambda split: (
                    state[split][0],
                    state[split][1],
                    stable_key(seed, f"{family}-tie-{log_name}", split),
                ),
            )
            assignment[log_name] = f"{destination}_{family}"
            state[destination][0] += log_summary[log_name]["eligible_scenes"]
            state[destination][1] += 1
        bin_state[family] = state

    zero_logs = [log_name for log_name, values in log_summary.items() if not values["eligible_scenes"]]
    natural_state = bin_state["natural"]
    for log_name in sorted(zero_logs, key=lambda value: stable_key(seed, "zero-log", value)):
        destination = min(
            ("dev", "final"),
            key=lambda split: (natural_state[split][1], stable_key(seed, f"zero-tie-{log_name}", split)),
        )
        assignment[log_name] = f"{destination}_natural"
        natural_state[destination][1] += 1
    return assignment


def update_source_row(raw: dict[str, Any]) -> dict[str, Any]:
    image = raw["images"][0]
    if not image.startswith(SOURCE_IMAGE_PREFIX):
        raise ValueError(f"Unexpected source image path: {image}")
    return {
        "images": [V3_IMAGE_PREFIX + image[len(SOURCE_IMAGE_PREFIX) :]],
        "problem": raw["problem"].replace("5-second", "4-second"),
        "answer": raw["answer"],
    }


def write_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def split_counts(master_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for split in sorted({row["split"] for row in master_rows}):
        selected = [row for row in master_rows if row["split"] == split]
        result[split] = {
            "tokens": len(selected),
            "logs": len({row["log_name"] for row in selected}),
            "intent_counts": dict(sorted(Counter(row["intent"] for row in selected).items())),
        }
    return result


def interaction_summary(
    eval_rows: list[EvalRow], eval_assignment: dict[str, str]
) -> dict[str, dict[str, float | int]]:
    result = {}
    for family in ("natural", "tail"):
        selected = [row for row in eval_rows if eval_assignment[row.log_name].endswith(family)]
        interactions = sum(row.interaction_tail_flag for row in selected)
        result[family] = {
            "tokens": len(selected),
            "interaction_scenes": interactions,
            "interaction_rate": interactions / len(selected),
        }
    return result


def prepare(args: argparse.Namespace) -> None:
    if args.output_data.exists() or args.output_manifests.exists() or args.output_report.exists():
        raise FileExistsError("D0S outputs must not already exist")
    sft_rows = load_sft_rows(args.sft_master_index)
    screen, monitor, random_rows = choose_training_rows(sft_rows)
    screen_tokens = {row.token for row in screen}
    monitor_tokens = {row.token for row in monitor}
    random_tokens = {row.token for row in random_rows}
    sft_logs = {row.log_name for row in sft_rows}

    parquet = pq.ParquetFile(args.sft_parquet)
    template_batch = next(parquet.iter_batches(columns=["problem"], batch_size=1))
    template = template_batch.column("problem")[0].as_py()
    raw_log_paths = sorted(args.navsim_logs.rglob("*.pkl"))
    unseen_paths = [path for path in raw_log_paths if path.stem not in sft_logs]
    eval_rows, log_summary = build_eval_rows(unseen_paths, template)
    eval_assignment = assign_eval_logs(log_summary)
    eval_by_split: dict[str, list[EvalRow]] = defaultdict(list)
    for row in eval_rows:
        eval_by_split[eval_assignment[row.log_name]].append(row)

    selected_source: dict[str, dict[str, Any]] = {}
    selected_tokens = screen_tokens | monitor_tokens
    for batch in parquet.iter_batches(columns=["images", "problem", "answer"], batch_size=2048):
        for raw in batch.to_pylist():
            token = raw["answer"]["token"]
            if token in selected_tokens:
                selected_source[token] = update_source_row(raw)
    if set(selected_source) != selected_tokens:
        raise ValueError("Selected source tokens do not exactly match SFT parquet")

    schema = parquet.schema_arrow
    write_parquet(args.output_data / "hf/grpo_screen.parquet", [selected_source[row.token] for row in screen], schema)
    write_parquet(args.output_data / "hf/train_monitor.parquet", [selected_source[row.token] for row in monitor], schema)
    dev_rows = sorted(eval_by_split["dev_natural"] + eval_by_split["dev_tail"], key=lambda row: row.token)
    final_rows = sorted(eval_by_split["final_natural"] + eval_by_split["final_tail"], key=lambda row: row.token)
    eval_payload = lambda row: {"images": [row.v3_image], "problem": row.problem, "answer": {"gt": [], "token": row.token}}
    write_parquet(args.output_data / "hf/dev.parquet", [eval_payload(row) for row in dev_rows], schema)
    write_parquet(args.output_data / "frozen_state/final/final.parquet", [eval_payload(row) for row in final_rows], schema)

    write_lines(args.output_manifests / "grpo_screen_8000.txt", (row.token for row in screen))
    write_lines(args.output_manifests / "train_monitor_256.txt", (row.token for row in monitor))
    write_lines(args.output_manifests / "random_train_2000.txt", (row.token for row in random_rows))
    for split in ("dev_natural", "dev_tail", "final_natural", "final_tail"):
        destination = args.output_data / "frozen_state/final" if split.startswith("final") else args.output_manifests
        write_lines(destination / f"{split}.txt", (row.token for row in sorted(eval_by_split[split], key=lambda value: value.token)))
    write_lines(args.output_data / "frozen_state/final/final_logs.txt", sorted(log for log, split in eval_assignment.items() if split.startswith("final")))

    eval_token_to_row = {row.token: row for row in eval_rows}
    eval_split_by_token = {row.token: split for split, rows in eval_by_split.items() for row in rows}
    master_rows = []
    for row in sft_rows:
        split = "train_monitor" if row.token in monitor_tokens else "grpo_screen" if row.token in screen_tokens else "sft_seen_unused"
        master_rows.append(
            {
                "token": row.token,
                "log_name": row.log_name,
                "source_universe": "sft_seen",
                "split": split,
                "intent": row.intent,
                "source_image": row.source_image,
                "v3_image": V3_IMAGE_PREFIX + row.source_image[len(SOURCE_IMAGE_PREFIX) :],
                "source_row": row.source_row,
                "map_location": "",
                "month": row.log_name[:7],
                "min_vehicle_distance_m": "",
                "min_vru_distance_m": "",
                "interaction_tail_flag": "",
                "optimizer_random": int(row.token in random_tokens),
                "prompt_version": PROMPT_VERSION,
                "sft_overlap": 1,
                "data_status": "source_unused" if split == "sft_seen_unused" else "active",
            }
        )
    for token, row in eval_token_to_row.items():
        master_rows.append(
            {
                "token": token,
                "log_name": row.log_name,
                "source_universe": "sft_unseen",
                "split": eval_split_by_token[token],
                "intent": row.intent,
                "source_image": row.source_image,
                "v3_image": row.v3_image,
                "source_row": "",
                "map_location": row.map_location,
                "month": row.month,
                "min_vehicle_distance_m": row.min_vehicle_distance_m,
                "min_vru_distance_m": row.min_vru_distance_m,
                "interaction_tail_flag": int(row.interaction_tail_flag),
                "optimizer_random": 0,
                "prompt_version": PROMPT_VERSION,
                "sft_overlap": 0,
                "data_status": "active",
            }
        )
    master_fields = list(master_rows[0])
    write_csv(args.output_manifests / "master_index.csv", master_fields, master_rows)
    write_csv(
        args.output_manifests / "eval_log_assignment.csv",
        ["log_name", "split", "eligible_scenes", "interaction_scenes", "interaction_rate", "minimum_actor_distance_m"],
        (
            {"log_name": log_name, "split": eval_assignment[log_name], **log_summary[log_name]}
            for log_name in sorted(eval_assignment)
        ),
    )

    active_rows = [row for row in master_rows if row["split"] != "sft_seen_unused"]
    write_csv(
        args.output_manifests / "active_assets.csv",
        ["token", "log_name", "split", "v3_image", "archive_relative"],
        (
            {
                "token": row["token"],
                "log_name": row["log_name"],
                "split": row["split"],
                "v3_image": row["v3_image"],
                "archive_relative": row["v3_image"][len(V3_IMAGE_PREFIX) :],
            }
            for row in active_rows
        ),
    )

    split_report = split_counts(master_rows)
    split_token_sets = {split: {row["token"] for row in master_rows if row["split"] == split} for split in split_report}
    split_log_sets = {split: {row["log_name"] for row in master_rows if row["split"] == split} for split in split_report}
    eval_splits = ("dev_natural", "dev_tail", "final_natural", "final_tail")
    overlap = {
        "train_eval_token": len((screen_tokens | monitor_tokens) & set(eval_token_to_row)),
        "train_eval_log": len((set(row.log_name for row in screen + monitor)) & set(eval_assignment)),
        "screen_monitor_token": len(screen_tokens & monitor_tokens),
        "screen_monitor_log": len(set(row.log_name for row in screen) & set(row.log_name for row in monitor)),
        "eval_pairwise": {
            f"{left}__{right}": {
                "token": len(split_token_sets[left] & split_token_sets[right]),
                "log": len(split_log_sets[left] & split_log_sets[right]),
            }
            for index, left in enumerate(eval_splits)
            for right in eval_splits[index + 1 :]
        },
    }
    risk_summary = interaction_summary(eval_rows, eval_assignment)
    all_gates = {
        "sft_rows_exact": len(sft_rows) == 103288,
        "sft_logs_exact": len(sft_logs) == 1192,
        "unseen_logs_exact": len(unseen_paths) == 118,
        "unseen_scenes_exact": len(eval_rows) == 835,
        "screen_exact": len(screen) == SCREEN_SIZE,
        "monitor_exact": len(monitor) == MONITOR_SIZE,
        "random_exact": len(random_rows) == RANDOM_SIZE,
        "random_subset_screen": random_tokens <= screen_tokens,
        "train_eval_disjoint": overlap["train_eval_token"] == overlap["train_eval_log"] == 0,
        "screen_monitor_disjoint": overlap["screen_monitor_token"] == overlap["screen_monitor_log"] == 0,
        "eval_pairwise_disjoint": all(not values["token"] and not values["log"] for values in overlap["eval_pairwise"].values()),
        "all_unseen_logs_assigned": set(eval_assignment) == {path.stem for path in unseen_paths},
        "tail_policy_independent": True,
        "tail_interaction_enriched": risk_summary["tail"]["interaction_rate"] > risk_summary["natural"]["interaction_rate"],
        "prompt_version_consistent": {row["prompt_version"] for row in master_rows} == {PROMPT_VERSION},
    }
    if not all(all_gates.values()):
        raise ValueError(f"D0S gate failure: {all_gates}")

    write_json(
        args.output_report / "sft_provenance_report.json",
        {
            "route": "REUSE_SFT_CONTROLLED_GRPO_OVERLAP",
            "sft_parquet_sha256": sha256_file(args.sft_parquet),
            "sft_master_index_sha256": sha256_file(args.sft_master_index),
            "model_hash_record_sha256": sha256_file(args.model_hash_record),
            "sft_tokens": len(sft_rows),
            "sft_logs": len(sft_logs),
            "screen_sft_token_reuse": len(screen_tokens),
            "screen_sft_log_reuse": len({row.log_name for row in screen}),
            "monitor_sft_token_reuse": len(monitor_tokens),
            "monitor_sft_log_reuse": len({row.log_name for row in monitor}),
            "screen_source_selection_rate": len(screen_tokens) / len(sft_rows),
            "screen_per_log_cap": SCREEN_PER_LOG_CAP,
            "monitor_per_log_cap": MONITOR_PER_LOG_CAP,
            "optimizer_manifest_tokens": RANDOM_SIZE,
        },
    )
    write_json(args.output_report / "overlap_report.json", overlap)
    write_json(args.output_report / "distribution_report.json", {"splits": split_report})
    write_json(
        args.output_report / "tail_definition_report.json",
        {
            "policy_independent": True,
            "unit": "log",
            "scene_flag": f"vehicle_distance<={VEHICLE_DISTANCE_M}m OR pedestrian/bicycle_distance<={VRU_DISTANCE_M}m",
            "log_rank": "interaction_scene_rate DESC, interaction_scene_count DESC, minimum_actor_distance ASC, stable_hash",
            "tail_logs": len({log for log, split in eval_assignment.items() if split.endswith("tail")}),
            "natural_logs": len({log for log, split in eval_assignment.items() if split.endswith("natural")}),
            "interaction_summary": risk_summary,
            "log_summary": log_summary,
        },
    )
    write_json(
        args.output_report / "d0r2_decision_report.json",
        {
            "status": "FROZEN",
            "seed": SEED,
            "sft_route": "REUSE_SFT_CONTROLLED_GRPO_OVERLAP",
            "evaluation": {
                "strict_unseen_logs": len(eval_assignment),
                "strict_unseen_tokens": len(eval_rows),
                "split_unit": "log",
                "tail_route": "POLICY_INDEPENDENT_GT_ACTOR_PROXIMITY",
                "tail_scene_rule": f"vehicle_distance<={VEHICLE_DISTANCE_M}m OR pedestrian/bicycle_distance<={VRU_DISTANCE_M}m",
                "tail_log_rule": "top half of nonempty logs by interaction rate/count/distance/stable hash",
                "zero_eligible_logs": "natural reserve only",
                "dev_final_allocation": "greedy scene-count balance within Natural and Tail",
                "unknown_intent_policy": "retain to preserve the complete strict-unseen reserve",
            },
            "training": {
                "screen_tokens": SCREEN_SIZE,
                "screen_per_log_cap": SCREEN_PER_LOG_CAP,
                "monitor_tokens": MONITOR_SIZE,
                "monitor_per_log_cap": MONITOR_PER_LOG_CAP,
                "optimizer_tokens_per_cell": RANDOM_SIZE,
                "random_tailmix_intent_quota": dict(sorted(Counter(row.intent for row in random_rows).items())),
                "tailmix_manifest_stage": "V3-S1 after the shared SFT rollout bank",
            },
            "matrix_priority": ["V3-RR", "V3-TC", "V3-TR", "V3-RC"],
        },
    )
    write_json(
        args.output_report / "d0s_acceptance_report.json",
        {
            "version": VERSION,
            "seed": SEED,
            "source_commit": args.source_commit,
            "counts": {"master_rows": len(master_rows), "active_assets": len(active_rows)},
            "gates": all_gates,
            "all_gates_passed": True,
        },
    )
    (args.output_report / "D0S_COMPLETE").touch()
    (args.output_report / "exit_code").write_text("0\n", encoding="utf-8")


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def verify_replay(args: argparse.Namespace) -> None:
    roots = {
        "data": (args.reference_data, args.replay_data),
        "manifests": (args.reference_manifests, args.replay_manifests),
        "reports": (args.reference_report, args.replay_report),
    }
    comparisons = {}
    for kind, (reference, replay) in roots.items():
        reference_hashes = tree_hashes(reference)
        replay_hashes = tree_hashes(replay)
        comparisons[kind] = {
            "files": len(reference_hashes),
            "reference_tree_sha256": hashlib.sha256(
                json.dumps(reference_hashes, sort_keys=True).encode()
            ).hexdigest(),
            "replay_tree_sha256": hashlib.sha256(
                json.dumps(replay_hashes, sort_keys=True).encode()
            ).hexdigest(),
            "match": reference_hashes == replay_hashes,
        }
    report = {"all_files_match": all(item["match"] for item in comparisons.values()), "trees": comparisons}
    if not report["all_files_match"]:
        raise ValueError(f"D0S replay mismatch: {report}")
    write_json(args.output, report)


def read_active_assets(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["token"] for row in rows}) != len(rows):
        raise ValueError("active_assets.csv contains duplicate tokens")
    return rows


def fetch_images(args: argparse.Namespace) -> None:
    assets = read_active_assets(args.active_assets)
    targets = {row["archive_relative"] for row in assets}
    manifest_hash = sha256_file(args.active_assets)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    for shard in range(1, args.shards + 1):
        marker = args.state_dir / f"current_{shard:02d}.json"
        if marker.exists() and json.loads(marker.read_text())["active_assets_sha256"] == manifest_hash:
            continue
        archive = args.state_dir / f"navtrain_current_{shard}.tgz"
        url = args.url_template.format(shard=shard)
        subprocess.run(["wget", "-c", "--progress=dot:giga", "-O", str(archive), url], check=True)
        extracted = []
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or len(parts) < 2:
                    continue
                relative = PurePosixPath(*parts[1:]).as_posix()
                if relative not in targets:
                    continue
                source = handle.extractfile(member)
                if source is None:
                    raise ValueError(f"Could not read archive member {member.name}")
                destination = args.output_root / Path(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append(relative)
        archive.unlink()
        write_json(marker, {"active_assets_sha256": manifest_hash, "shard": shard, "extracted": len(extracted)})

    missing = sorted(relative for relative in targets if not (args.output_root / Path(relative)).is_file())
    report = {
        "expected": len(targets),
        "present": len(targets) - len(missing),
        "missing": len(missing),
        "active_assets_sha256": manifest_hash,
    }
    write_json(args.state_dir / "image_coverage_report.json", report)
    if missing:
        raise ValueError(f"Missing {len(missing)} active CAM_F0 images")
    (args.state_dir / "D0A_IMAGES_COMPLETE").touch()


def decode_zip_member(info: zipfile.ZipInfo, payload: bytes) -> bytes:
    if len(payload) < 30:
        raise ValueError(f"Truncated ZIP local header for {info.filename}")
    signature, _, flags, compression, _, _, _, _, _, name_length, extra_length = struct.unpack_from(
        "<4s5H3I2H", payload
    )
    if signature != b"PK\x03\x04" or flags & 1:
        raise ValueError(f"Invalid or encrypted ZIP member: {info.filename}")
    if compression != info.compress_type:
        raise ValueError(f"ZIP compression mismatch for {info.filename}")
    data_start = 30 + name_length + extra_length
    compressed = payload[data_start : data_start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise ValueError(f"Truncated ZIP member data for {info.filename}")
    if compression == zipfile.ZIP_STORED:
        data = compressed
    elif compression == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported ZIP compression {compression} for {info.filename}")
    if len(data) != info.file_size or binascii.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"ZIP integrity check failed for {info.filename}")
    return data


def download_zip_member(url: str, info: zipfile.ZipInfo, relative: str, staging: Path) -> dict[str, Any]:
    destination = staging / relative
    if destination.is_file():
        data = destination.read_bytes()
        if len(data) == info.file_size and binascii.crc32(data) & 0xFFFFFFFF == info.CRC:
            return {"path": relative, "crc32": f"{info.CRC:08x}", "compressed_bytes": info.compress_size}
        destination.unlink()

    range_size = 30 + len(info.filename.encode("utf-8")) + ZIP_LOCAL_EXTRA_LIMIT + info.compress_size
    end = min(info.header_offset + range_size - 1, SELECTIVE_ZIP_SIZE - 1)
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-filesize",
            str(range_size),
            "--range",
            f"{info.header_offset}-{end}",
            url,
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    data = decode_zip_member(info, result.stdout)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return {"path": relative, "crc32": f"{info.CRC:08x}", "compressed_bytes": info.compress_size}


def fetch_full_front_images(args: argparse.Namespace) -> None:
    assets = read_active_assets(args.active_assets)
    expected = {row["archive_relative"] for row in assets}
    missing_rows = [row for row in assets if not (args.output_root / row["archive_relative"]).is_file()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)

    overlap_rows = sorted(
        (row for row in assets if (args.output_root / row["archive_relative"]).is_file()), key=lambda row: row["token"]
    )[:MIRROR_OVERLAP_CHECKS]
    selected_rows = missing_rows + overlap_rows
    desired = {f"trainval/{row['archive_relative']}": row["archive_relative"] for row in selected_rows}
    staging = args.state_dir / "selective_zip_staging"
    staging.mkdir(parents=True, exist_ok=True)
    path_list = args.state_dir / "selective_zip_paths.txt"
    path_list.write_text("\n".join(sorted(desired)) + "\n", encoding="utf-8")

    resolved = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--range",
            "0-0",
            "--output",
            os.devnull,
            "--write-out",
            "%{url_effective}",
            SELECTIVE_ZIP_URL,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    directory_tail = args.state_dir / "selective_zip_directory.bin"
    directory_size = SELECTIVE_ZIP_SIZE - SELECTIVE_ZIP_DIRECTORY_OFFSET
    if not directory_tail.is_file() or directory_tail.stat().st_size != directory_size:
        directory_tail.unlink(missing_ok=True)
        print(f"selective_zip_directory_start bytes={directory_size}", flush=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--max-filesize",
                str(directory_size),
                "--range",
                f"{SELECTIVE_ZIP_DIRECTORY_OFFSET}-{SELECTIVE_ZIP_SIZE - 1}",
                "--output",
                str(directory_tail),
                resolved,
            ],
            check=True,
        )
        print("selective_zip_directory_complete", flush=True)

    sparse_index = args.state_dir / "selective_zip_index.zip"
    with sparse_index.open("wb") as index_handle:
        index_handle.truncate(SELECTIVE_ZIP_SIZE)
        index_handle.seek(SELECTIVE_ZIP_DIRECTORY_OFFSET)
        with directory_tail.open("rb") as tail_handle:
            shutil.copyfileobj(tail_handle, index_handle)
    try:
        with zipfile.ZipFile(sparse_index) as archive:
            archive_infos = archive.infolist()
            matched = {info.filename: info for info in archive_infos if info.filename in desired}
    finally:
        sparse_index.unlink(missing_ok=True)
    absent = sorted(set(desired) - set(matched))
    if absent:
        raise ValueError(f"Selective ZIP is missing {len(absent)} selected images")

    print(f"selective_zip_members_start count={len(matched)} workers={SELECTIVE_ZIP_WORKERS}", flush=True)
    member_records = []
    with ThreadPoolExecutor(max_workers=SELECTIVE_ZIP_WORKERS) as executor:
        futures = {
            executor.submit(download_zip_member, resolved, info, desired[member], staging): member
            for member, info in matched.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            member_records.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                print(f"selective_zip_members_progress={completed}/{len(futures)}", flush=True)

    for row in overlap_rows:
        relative = row["archive_relative"]
        mirror_path = staging / relative
        if sha256_file(mirror_path) != sha256_file(args.output_root / relative):
            raise ValueError(f"Selective ZIP overlap mismatch for {relative}")
    for row in missing_rows:
        relative = row["archive_relative"]
        source = staging / relative
        destination = args.output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    write_json(
        args.state_dir / "selective_zip_report.json",
        {
            "repository": "richardyann/navsim-select",
            "revision": SELECTIVE_ZIP_REVISION,
            "archive_path": "sensor_blobs/trainval.zip",
            "archive_bytes": SELECTIVE_ZIP_SIZE,
            "central_directory_sha256": sha256_file(directory_tail),
            "archive_entries": len(archive_infos),
            "missing_images_filled": len(missing_rows),
            "official_navtrain_overlap_verified": len(overlap_rows),
            "selected_compressed_bytes": sum(record["compressed_bytes"] for record in member_records),
            "members": sorted(member_records, key=lambda record: record["path"]),
            "path_list_sha256": sha256_file(path_list),
        },
    )
    shutil.rmtree(staging)
    directory_tail.unlink()

    present = {row["archive_relative"] for row in assets if (args.output_root / row["archive_relative"]).is_file()}
    report = {
        "expected": len(expected),
        "present": len(present),
        "missing": len(expected - present),
        "active_assets_sha256": sha256_file(args.active_assets),
        "sources": ["NAVSIM navtrain current", "verified selective ZIP ranges"],
    }
    write_json(args.state_dir / "image_coverage_report.json", report)
    if report["missing"]:
        raise ValueError(f"Missing {report['missing']} active CAM_F0 images")
    (args.state_dir / "D0A_IMAGES_COMPLETE").touch()


def build_cache_log(
    log_name: str,
    log_path: Path,
    tokens: set[str],
    output_root: Path,
    map_root: str,
) -> tuple[str, int]:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    processor = MetricCacheProcessor(
        cache_path=str(output_root),
        force_feature_computation=False,
        proposal_sampling=TrajectorySampling(num_poses=40, interval_length=0.1),
    )
    with log_path.open("rb") as handle:
        frames = load_navsim_log(handle)
    found = set()
    for start in range(0, len(frames) - NUM_FRAMES + 1):
        window = frames[start : start + NUM_FRAMES]
        center = window[NUM_HISTORY_FRAMES - 1]
        token = center.get("token")
        if token not in tokens:
            continue
        scene = Scene.from_scene_dict_list(
            window,
            None,
            num_history_frames=NUM_HISTORY_FRAMES,
            num_future_frames=NUM_FUTURE_FRAMES,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        scenario = NavSimScenario(scene, map_root=map_root, map_version="nuplan-maps-v1.0")
        if processor.compute_and_save_metric_cache(scenario) is None:
            raise RuntimeError(f"Metric cache failed for {token}")
        found.add(token)
        del scenario, scene
        gc.collect()
    if found != tokens:
        raise ValueError(f"Cache token mismatch for {log_name}: expected {len(tokens)}, found {len(found)}")
    return log_name, len(found)


def build_cache(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    assets = read_active_assets(args.active_assets)
    by_log: dict[str, set[str]] = defaultdict(set)
    for row in assets:
        by_log[row["log_name"]].add(row["token"])
    log_paths = {path.stem: path for path in args.navsim_logs.rglob("*.pkl")}
    missing_logs = set(by_log) - set(log_paths)
    if missing_logs:
        raise ValueError(f"Missing {len(missing_logs)} active NAVSIM logs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        (log_name, log_paths[log_name], by_log[log_name], args.output_root, os.environ["NUPLAN_MAPS_ROOT"])
        for log_name in sorted(by_log)
        if not (args.state_dir / f"{log_name}.complete").exists()
    ]
    completed = len(by_log) - len(pending)
    if args.workers == 1:
        results = (build_cache_log(*job) for job in pending)
        for log_name, token_count in results:
            completed += 1
            (args.state_dir / f"{log_name}.complete").write_text(f"{token_count}\n", encoding="utf-8")
            print(f"cache_log={completed}/{len(by_log)} tokens={token_count} log={log_name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(build_cache_log, *job) for job in pending]
            for future in as_completed(futures):
                log_name, token_count = future.result()
                completed += 1
                (args.state_dir / f"{log_name}.complete").write_text(f"{token_count}\n", encoding="utf-8")
                print(f"cache_log={completed}/{len(by_log)} tokens={token_count} log={log_name}", flush=True)

    expected = {row["token"] for row in assets}
    present = {path.parent.name for path in args.output_root.rglob("metric_cache.pkl")}
    report = {"expected": len(expected), "present": len(present & expected), "missing": len(expected - present), "unexpected": len(present - expected)}
    write_json(args.state_dir / "metric_cache_coverage_report.json", report)
    if report["missing"] or report["unexpected"]:
        raise ValueError(f"Metric cache coverage failed: {report}")
    (args.state_dir / "D0A_CACHE_COMPLETE").touch()


def file_hash_rows(root: Path, paths: Iterable[Path], kind: str) -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def json_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(json_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(json_finite(item) for item in value)
    return True


def prompt_is_consistent(path: Path) -> bool:
    table = pq.read_table(path, columns=["problem"])
    return all(
        isinstance(problem, str) and "optimal future 4-second trajectory" in problem and "5-second" not in problem
        for problem in table.column("problem").to_pylist()
    )


def freeze(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    assets = read_active_assets(args.active_assets)
    expected = {row["token"]: row for row in assets}
    image_paths = [args.image_root / Path(row["archive_relative"]) for row in assets]
    missing_images = [path for path in image_paths if not path.is_file()]
    invalid_images = []
    for path in image_paths:
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, SyntaxError):
            invalid_images.append(path)
    cache_paths = list(args.cache_root.rglob("metric_cache.pkl"))
    cache_tokens = {path.parent.name for path in cache_paths}

    required_reports = {
        name: args.report_root / name
        for name in (
            "d0r2_decision_report.json",
            "d0s_acceptance_report.json",
            "distribution_report.json",
            "overlap_report.json",
            "reproducibility_report.json",
            "sft_provenance_report.json",
            "tail_definition_report.json",
        )
    }
    reports_present = all(path.is_file() for path in required_reports.values())
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in required_reports.items()
        if path.is_file()
    }
    source_head = subprocess.run(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    source_status = subprocess.run(
        ["git", "-C", str(args.source_root), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    expected_model_hashes = parse_model_hashes(args.model_hash_record) if args.model_hash_record.is_file() else {}
    model_hashes_match = bool(expected_model_hashes) and all(
        (args.model_root / filename).is_file()
        and sha256_file(args.model_root / filename) == expected_hash
        for filename, expected_hash in expected_model_hashes.items()
    )
    parquet_paths = sorted(args.data_root.rglob("*.parquet"))
    prompts_consistent = bool(parquet_paths) and all(prompt_is_consistent(path) for path in parquet_paths)

    with args.master_index.open(encoding="utf-8-sig", newline="") as handle:
        master = list(csv.DictReader(handle))
    with args.eval_log_assignment.open(encoding="utf-8-sig", newline="") as handle:
        eval_log_assignment = list(csv.DictReader(handle))
    active_master = [row for row in master if row["split"] != "sft_seen_unused"]
    split_tokens: dict[str, set[str]] = defaultdict(set)
    split_logs: dict[str, set[str]] = defaultdict(set)
    for row in active_master:
        split_tokens[row["split"]].add(row["token"])
        split_logs[row["split"]].add(row["log_name"])
    eval_splits = ("dev_natural", "dev_tail", "final_natural", "final_tail")
    train_tokens = split_tokens["grpo_screen"] | split_tokens["train_monitor"]
    train_logs = split_logs["grpo_screen"] | split_logs["train_monitor"]
    eval_tokens = set().union(*(split_tokens[split] for split in eval_splits))
    all_eval_logs = {row["log_name"] for row in eval_log_assignment}
    gates = {
        "active_master_exact": len(active_master) == len(expected) == 9091,
        "active_tokens_exact": set(expected) == {row["token"] for row in active_master},
        "images_exact": not missing_images and not invalid_images,
        "metric_cache_exact": cache_tokens == set(expected),
        "train_eval_token_disjoint": not train_tokens & eval_tokens,
        "train_eval_log_disjoint": not train_logs & all_eval_logs,
        "eval_pairwise_token_disjoint": all(
            not split_tokens[left] & split_tokens[right]
            for index, left in enumerate(eval_splits)
            for right in eval_splits[index + 1 :]
        ),
        "eval_pairwise_log_disjoint": all(
            not split_logs[left] & split_logs[right]
            for index, left in enumerate(eval_splits)
            for right in eval_splits[index + 1 :]
        ),
        "strict_unseen_complete": len(eval_tokens) == 835 and len(eval_log_assignment) == 118,
        "strict_unseen_log_assignments_unique": len({row["log_name"] for row in eval_log_assignment}) == 118,
        "master_prompt_version_consistent": {row.get("prompt_version") for row in master} == {PROMPT_VERSION},
        "master_sft_overlap_consistent": all(
            row.get("sft_overlap") == ("1" if row["source_universe"] == "sft_seen" else "0") for row in master
        ),
        "master_data_status_consistent": all(
            row.get("data_status") == ("source_unused" if row["split"] == "sft_seen_unused" else "active")
            for row in master
        ),
        "parquet_prompt_consistent": prompts_consistent,
        "d0a_markers_present": (args.image_state_dir / "D0A_IMAGES_COMPLETE").is_file()
        and (args.cache_state_dir / "D0A_CACHE_COMPLETE").is_file(),
        "required_reports_present": reports_present,
        "reports_finite": reports_present and all(json_finite(report) for report in reports.values()),
        "d0r2_frozen": reports.get("d0r2_decision_report.json", {}).get("status") == "FROZEN",
        "d0s_gates_passed": reports.get("d0s_acceptance_report.json", {}).get("all_gates_passed") is True,
        "tail_policy_independent": reports.get("tail_definition_report.json", {}).get("policy_independent") is True,
        "d0s_reproducible": reports.get("reproducibility_report.json", {}).get("all_files_match") is True,
        "source_commit_exact": source_head == args.source_commit,
        "source_clean": not source_status,
        "model_hash_record_present": args.model_hash_record.is_file(),
        "model_files_match_record": model_hashes_match,
    }
    if not all(gates.values()):
        raise ValueError(f"D0F gate failure: {gates}")

    args.output_dir.mkdir(parents=True)
    asset_hashes = file_hash_rows(args.image_root, image_paths, "cam_f0")
    asset_hashes += file_hash_rows(args.cache_root, cache_paths, "metric_cache")
    write_csv(args.output_dir / "asset_sha256.csv", ["kind", "path", "bytes", "sha256"], asset_hashes)
    coverage = {
        "expected_tokens": len(expected),
        "cam_f0": {"present": len(image_paths), "missing": 0, "invalid": 0},
        "metric_cache": {"present": len(cache_paths), "missing": 0, "unexpected": 0},
    }
    write_json(args.output_dir / "asset_coverage_report.json", coverage)
    core_files = [
        args.master_index,
        args.active_assets,
        args.eval_log_assignment,
        args.model_hash_record,
        args.image_state_dir / "image_coverage_report.json",
        args.cache_state_dir / "metric_cache_coverage_report.json",
        *sorted(args.data_root.rglob("*.parquet")),
        *sorted(args.manifest_root.glob("*.txt")),
        *sorted(args.report_root.glob("*.json")),
    ]
    core_hashes = {str(path): sha256_file(path) for path in core_files}
    card = {
        "dataset_version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "source_commit": args.source_commit,
        "route": "REUSE_SFT_CONTROLLED_GRPO_OVERLAP",
        "counts": {
            "sft_seen_source_tokens": 103288,
            "sft_seen_source_logs": 1192,
            "grpo_screen_tokens": len(split_tokens["grpo_screen"]),
            "train_monitor_tokens": len(split_tokens["train_monitor"]),
            "strict_unseen_tokens": len(eval_tokens),
            "strict_unseen_logs": len(eval_log_assignment),
            "active_assets": len(expected),
        },
        "gates": gates,
        "core_sha256": core_hashes,
        "asset_sha256_manifest": sha256_file(args.output_dir / "asset_sha256.csv"),
        "model_sha256": expected_model_hashes,
        "final_access": "LOCKED",
    }
    write_json(args.output_dir / "dataset_card.json", card)
    write_json(
        args.output_dir / "final_access_lock.json",
        {
            "status": "LOCKED",
            "allowed_before_promotion": False,
            "final_parquet_sha256": sha256_file(args.data_root / "frozen_state/final/final.parquet"),
            "final_logs_sha256": sha256_file(args.data_root / "frozen_state/final/final_logs.txt"),
        },
    )
    for path in (args.data_root / "frozen_state/final").iterdir():
        path.chmod(0o400)
    (args.output_dir / "V3_DATA_FROZEN").touch()
    (args.output_dir / "COMPLETE").touch()
    (args.output_dir / "exit_code").write_text("0\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and freeze Dataset V3 controlled-overlap assets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--sft-parquet", type=Path, required=True)
    prepare_parser.add_argument("--sft-master-index", type=Path, required=True)
    prepare_parser.add_argument("--navsim-logs", type=Path, required=True)
    prepare_parser.add_argument("--model-hash-record", type=Path, required=True)
    prepare_parser.add_argument("--source-commit", required=True)
    prepare_parser.add_argument("--output-data", type=Path, required=True)
    prepare_parser.add_argument("--output-manifests", type=Path, required=True)
    prepare_parser.add_argument("--output-report", type=Path, required=True)
    prepare_parser.set_defaults(run=prepare)

    replay_parser = subparsers.add_parser("verify-replay")
    replay_parser.add_argument("--reference-data", type=Path, required=True)
    replay_parser.add_argument("--reference-manifests", type=Path, required=True)
    replay_parser.add_argument("--reference-report", type=Path, required=True)
    replay_parser.add_argument("--replay-data", type=Path, required=True)
    replay_parser.add_argument("--replay-manifests", type=Path, required=True)
    replay_parser.add_argument("--replay-report", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.set_defaults(run=verify_replay)

    image_parser = subparsers.add_parser("fetch-images")
    image_parser.add_argument("--active-assets", type=Path, required=True)
    image_parser.add_argument("--output-root", type=Path, required=True)
    image_parser.add_argument("--state-dir", type=Path, required=True)
    image_parser.add_argument("--shards", type=int, default=32)
    image_parser.add_argument(
        "--url-template",
        default="https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/navsim/navtrain_current_{shard}.tgz",
    )
    image_parser.set_defaults(run=fetch_images)

    full_front_parser = subparsers.add_parser("fetch-full-front-images")
    full_front_parser.add_argument("--active-assets", type=Path, required=True)
    full_front_parser.add_argument("--output-root", type=Path, required=True)
    full_front_parser.add_argument("--state-dir", type=Path, required=True)
    full_front_parser.set_defaults(run=fetch_full_front_images)

    cache_parser = subparsers.add_parser("build-cache")
    cache_parser.add_argument("--active-assets", type=Path, required=True)
    cache_parser.add_argument("--navsim-logs", type=Path, required=True)
    cache_parser.add_argument("--output-root", type=Path, required=True)
    cache_parser.add_argument("--state-dir", type=Path, required=True)
    cache_parser.add_argument("--workers", type=int, default=1)
    cache_parser.set_defaults(run=build_cache)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--master-index", type=Path, required=True)
    freeze_parser.add_argument("--eval-log-assignment", type=Path, required=True)
    freeze_parser.add_argument("--active-assets", type=Path, required=True)
    freeze_parser.add_argument("--data-root", type=Path, required=True)
    freeze_parser.add_argument("--manifest-root", type=Path, required=True)
    freeze_parser.add_argument("--report-root", type=Path, required=True)
    freeze_parser.add_argument("--image-root", type=Path, required=True)
    freeze_parser.add_argument("--cache-root", type=Path, required=True)
    freeze_parser.add_argument("--image-state-dir", type=Path, required=True)
    freeze_parser.add_argument("--cache-state-dir", type=Path, required=True)
    freeze_parser.add_argument("--model-hash-record", type=Path, required=True)
    freeze_parser.add_argument("--model-root", type=Path, required=True)
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--source-commit", required=True)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    freeze_parser.set_defaults(run=freeze)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
