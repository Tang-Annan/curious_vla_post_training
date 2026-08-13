"""Grouped NAVSIM rewards for Vanilla GRPO and SLDR experiments."""

import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import httpx

from verl.utils.reward_score.navsim.helper import denormalize, get_trajectory_parser
from verl.utils.reward_score.navsim.safety_dense_reward import REQUIRED_METRICS, compute_sldr


REWARD_NAME = "navsim_grouped"
REWARD_TYPE = "batch"

_server_url = os.environ.get("NAVSIM_REWARD_URL", "http://127.0.0.1:8901").rstrip("/")
_timeout = float(os.environ.get("NAVSIM_REWARD_TIMEOUT", "120"))
_log_dir = os.path.join("checkpoints", "debug", os.environ.get("EXP_NAME", "default_exp"))
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, f"generations_{datetime.now():%m%d%H%M}.jsonl")
_log_lock = threading.Lock()


def _zero_result(token: str, response_length: int, poses: list[list[float]]) -> tuple[dict[str, float], dict[str, Any]]:
    log_row = {
        "token": token,
        "response_length": response_length,
        "parsed_ok": False,
        "poses": poses,
        "safe": 0.0,
        "training_reward": 0.0,
        "pdms": 0.0,
        "pdms_scaled": 0.0,
        "reward_latency_ms": 0.0,
    }
    score = {
        "overall": 0.0,
        "accuracy": 0.0,
        "pdms": 0.0,
        "pdms_scaled": 0.0,
        "parsed_ok": 0.0,
        "safe": 0.0,
        "no_at_fault_collisions": 0.0,
        "drivable_area_compliance": 0.0,
        "ego_progress": 0.0,
        "time_to_collision_within_bound": 0.0,
        "history_comfort": 0.0,
        "reward_latency_ms": 0.0,
    }
    return score, log_row


def _score_groups(reward_inputs: list[dict[str, Any]], reward_mode: str) -> list[dict[str, float]]:
    if reward_mode not in {"scaled_pdms", "sldr"}:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")

    parse_fn = get_trajectory_parser()
    scores: list[dict[str, float] | None] = [None] * len(reward_inputs)
    log_rows: list[dict[str, Any] | None] = [None] * len(reward_inputs)
    groups: dict[str, list[tuple[int, int, list[list[float]]]]] = defaultdict(list)
    for index, item in enumerate(reward_inputs):
        token = item["ground_truth"]["token"]
        response_length = int(item.get("response_length", 0))
        poses = parse_fn(item["response"])
        if not poses or len(poses) != 8:
            scores[index], log_rows[index] = _zero_result(token, response_length, poses or [])
            continue
        groups[token].append((index, response_length, denormalize(poses)))

    with httpx.Client(trust_env=False, timeout=_timeout) as client:
        for token, items in groups.items():
            started = time.perf_counter()
            response = client.post(
                f"{_server_url}/score_group",
                json={"token": token, "poses": [poses for _, _, poses in items], "verbose": False},
            )
            response.raise_for_status()
            metrics_list = response.json()
            if len(metrics_list) != len(items):
                raise RuntimeError(f"score_group returned {len(metrics_list)} results for {len(items)} poses")
            latency_ms = (time.perf_counter() - started) * 1000.0

            for (index, response_length, poses), metrics in zip(items, metrics_list):
                missing = [key for key in REQUIRED_METRICS if key not in metrics]
                if missing:
                    raise KeyError(f"Missing NAVSIM metrics: {', '.join(missing)}")
                training_reward = (
                    float(metrics["pdms_scaled"]) if reward_mode == "scaled_pdms" else compute_sldr(metrics)
                )
                safe = float(
                    float(metrics["no_at_fault_collisions"]) > 0.0
                    and float(metrics["drivable_area_compliance"]) > 0.0
                )
                scores[index] = {
                    "overall": training_reward,
                    "accuracy": float(metrics["pdms_scaled"]),
                    "pdms": float(metrics["pdms"]),
                    "pdms_scaled": float(metrics["pdms_scaled"]),
                    "parsed_ok": 1.0,
                    "safe": safe,
                    "no_at_fault_collisions": float(metrics["no_at_fault_collisions"]),
                    "drivable_area_compliance": float(metrics["drivable_area_compliance"]),
                    "ego_progress": float(metrics["ego_progress"]),
                    "time_to_collision_within_bound": float(metrics["time_to_collision_within_bound"]),
                    "history_comfort": float(metrics["history_comfort"]),
                    "reward_latency_ms": latency_ms / len(items),
                }
                log_rows[index] = {
                    "token": token,
                    "response_length": response_length,
                    "parsed_ok": True,
                    "poses": poses,
                    "safe": safe,
                    "training_reward": training_reward,
                    "pdms": float(metrics["pdms"]),
                    "pdms_scaled": float(metrics["pdms_scaled"]),
                    "no_at_fault_collisions": float(metrics["no_at_fault_collisions"]),
                    "drivable_area_compliance": float(metrics["drivable_area_compliance"]),
                    "ego_progress": float(metrics["ego_progress"]),
                    "time_to_collision_within_bound": float(metrics["time_to_collision_within_bound"]),
                    "history_comfort": float(metrics["history_comfort"]),
                    "reward_latency_ms": latency_ms / len(items),
                }

    completed_scores = [score for score in scores if score is not None]
    completed_logs = [row for row in log_rows if row is not None]
    if len(completed_scores) != len(scores) or len(completed_logs) != len(log_rows):
        raise RuntimeError("A grouped reward result was not populated.")
    with _log_lock, open(_log_path, "a", encoding="utf-8") as handle:
        for row in completed_logs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return completed_scores


def compute_score_group_fast(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    return _score_groups(reward_inputs, reward_mode="scaled_pdms")


def compute_score_sldr(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    return _score_groups(reward_inputs, reward_mode="sldr")
