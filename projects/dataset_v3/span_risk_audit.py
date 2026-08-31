from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from projects.dataset_v3.data_prep import quaternion_yaw, stable_key, wrap_angle
from projects.dataset_v3.inventory import NUM_FRAMES, NUM_HISTORY_FRAMES, load_navsim_log, sha256_file


VEHICLE_DISTANCE_M = 5.0
VRU_DISTANCE_M = 10.0
CONTEXT_DISTANCE_M = 20.0
FRONT_HALF_ANGLE_RAD = math.radians(45.0)
TURN_HEADING_RAD = math.radians(20.0)
LATERAL_SHIFT_M = 1.5
BRAKE_SPEED_DROP_MPS = 2.0
STOP_SPEED_MPS = 0.5
GO_SPEED_MPS = 2.0
CONSTRUCTION_NAMES = {"barrier", "traffic_cone", "czone_sign"}
VRU_NAMES = {"pedestrian", "bicycle"}
SPAN_SCALED_RECIPE = {
    "warmup_positive": 667,
    "mixed_positive": 1000,
    "mixed_negative": 166,
    "mixed_recovery": 167,
}


def read_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Manifest contains duplicate tokens: {path}")
    return tokens


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, rows: Iterable[dict[str, Any]], seed: int, namespace: str) -> None:
    tokens = sorted((str(row["token"]) for row in rows), key=lambda token: stable_key(seed, namespace, token))
    path.write_text("".join(f"{token}\n" for token in tokens), encoding="utf-8")


def _actor_distances(
    frame: dict[str, Any],
) -> tuple[float | None, float | None, float | None, bool, bool, bool]:
    vehicle: list[float] = []
    vru: list[float] = []
    construction: list[float] = []
    front_vehicle = False
    front_vru = False
    front_construction = False
    names = frame["anns"]["gt_names"]
    boxes = frame["anns"]["gt_boxes"]
    for raw_name, box in zip(names, boxes):
        name = str(raw_name)
        x, y = float(box[0]), float(box[1])
        distance = math.hypot(x, y)
        if name == "vehicle":
            vehicle.append(distance)
            front_kind = "vehicle"
        elif name in VRU_NAMES:
            vru.append(distance)
            front_kind = "vru"
        elif name in CONSTRUCTION_NAMES:
            front_kind = "construction"
        else:
            front_kind = None
        if name in CONSTRUCTION_NAMES:
            construction.append(distance)
        in_front = (
            front_kind is not None
            and distance <= CONTEXT_DISTANCE_M
            and x > 0
            and abs(math.atan2(y, x)) <= FRONT_HALF_ANGLE_RAD
        )
        front_vehicle |= bool(in_front and front_kind == "vehicle")
        front_vru |= bool(in_front and front_kind == "vru")
        front_construction |= bool(in_front and front_kind == "construction")
    return (
        min(vehicle, default=None),
        min(vru, default=None),
        min(construction, default=None),
        front_vehicle,
        front_vru,
        front_construction,
    )


def _segment_speeds(frames: list[dict[str, Any]]) -> list[float]:
    speeds = []
    for left, right in zip(frames, frames[1:]):
        dt = (int(right["timestamp"]) - int(left["timestamp"])) / 1_000_000
        if dt <= 0:
            raise ValueError("NAVSIM timestamps must be strictly increasing")
        left_xy = left["ego2global_translation"]
        right_xy = right["ego2global_translation"]
        speeds.append(math.hypot(float(right_xy[0] - left_xy[0]), float(right_xy[1] - left_xy[1])) / dt)
    return speeds


def _final_ego_pose(window: list[dict[str, Any]]) -> tuple[float, float, float]:
    center = window[NUM_HISTORY_FRAMES - 1]
    final = window[-1]
    origin = center["ego2global_translation"]
    dx = float(final["ego2global_translation"][0] - origin[0])
    dy = float(final["ego2global_translation"][1] - origin[1])
    yaw = quaternion_yaw(center["ego2global_rotation"])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    x = cos_yaw * dx + sin_yaw * dy
    y = -sin_yaw * dx + cos_yaw * dy
    heading = wrap_angle(quaternion_yaw(final["ego2global_rotation"]) - yaw)
    return x, y, heading


def label_window(window: list[dict[str, Any]]) -> dict[str, Any]:
    center_index = NUM_HISTORY_FRAMES - 1
    center = window[center_index]
    (
        current_vehicle,
        current_vru,
        _,
        current_front_vehicle,
        current_front_vru,
        current_front_construction,
    ) = _actor_distances(center)
    horizon_vehicle: list[float] = []
    horizon_vru: list[float] = []
    construction: list[float] = []
    traffic_control = False
    for frame in window[center_index:]:
        vehicle, vru, construction_distance, _, _, _ = _actor_distances(frame)
        if vehicle is not None:
            horizon_vehicle.append(vehicle)
        if vru is not None:
            horizon_vru.append(vru)
        if construction_distance is not None:
            construction.append(construction_distance)
        traffic_control |= bool(frame.get("traffic_lights"))

    history_speeds = _segment_speeds(window[:NUM_HISTORY_FRAMES])
    future_speeds = _segment_speeds(window[center_index:])
    center_speed = history_speeds[-1]
    min_future_speed = min(future_speeds)
    max_future_speed = max(future_speeds)
    final_x, final_y, final_heading = _final_ego_pose(window)
    expert_turn = abs(final_heading) >= TURN_HEADING_RAD
    expert_lateral = abs(final_y) >= LATERAL_SHIFT_M and not expert_turn
    expert_braking = center_speed >= GO_SPEED_MPS and center_speed - min_future_speed >= BRAKE_SPEED_DROP_MPS
    expert_stop_to_go = center_speed <= STOP_SPEED_MPS and max_future_speed >= GO_SPEED_MPS

    current_interaction = (
        current_vehicle is not None and current_vehicle <= VEHICLE_DISTANCE_M
    ) or (current_vru is not None and current_vru <= VRU_DISTANCE_M)
    min_horizon_vehicle = min(horizon_vehicle, default=None)
    min_horizon_vru = min(horizon_vru, default=None)
    vehicle_interaction = min_horizon_vehicle is not None and min_horizon_vehicle <= VEHICLE_DISTANCE_M
    vru_interaction = min_horizon_vru is not None and min_horizon_vru <= VRU_DISTANCE_M
    construction_present = bool(construction) and min(construction) <= CONTEXT_DISTANCE_M
    expert_response = expert_turn or expert_lateral or expert_braking or expert_stop_to_go
    construction_response = construction_present and expert_response
    traffic_control_response = traffic_control and (expert_braking or expert_stop_to_go)
    event_risk = vehicle_interaction or vru_interaction or construction_response or traffic_control_response
    current_traffic_control = bool(center.get("traffic_lights"))
    learnable_risk = (
        (vehicle_interaction and current_front_vehicle)
        or (vru_interaction and current_front_vru)
        or (construction_response and current_front_construction)
        or (traffic_control_response and current_traffic_control)
    )

    labels = []
    if vehicle_interaction:
        labels.append("vehicle_interaction")
    if vru_interaction:
        labels.append("vru_interaction")
    if construction_response:
        labels.append("construction_response")
    if traffic_control_response:
        labels.append("traffic_control_response")
    if expert_turn:
        labels.append("expert_turn")
    if expert_lateral:
        labels.append("expert_lateral")
    if expert_braking:
        labels.append("expert_braking")
    if expert_stop_to_go:
        labels.append("expert_stop_to_go")
    return {
        "current_vehicle_distance_m": "" if current_vehicle is None else current_vehicle,
        "current_vru_distance_m": "" if current_vru is None else current_vru,
        "horizon_vehicle_distance_m": "" if min_horizon_vehicle is None else min_horizon_vehicle,
        "horizon_vru_distance_m": "" if min_horizon_vru is None else min_horizon_vru,
        "current_interaction_flag": int(current_interaction),
        "horizon_vehicle_interaction": int(vehicle_interaction),
        "horizon_vru_interaction": int(vru_interaction),
        "construction_present": int(construction_present),
        "traffic_control_present": int(traffic_control),
        "current_traffic_control": int(current_traffic_control),
        "expert_turn": int(expert_turn),
        "expert_lateral": int(expert_lateral),
        "expert_braking": int(expert_braking),
        "expert_stop_to_go": int(expert_stop_to_go),
        "event_risk_flag": int(event_risk),
        "current_vehicle_front_context": int(current_front_vehicle),
        "current_vru_front_context": int(current_front_vru),
        "current_construction_front_context": int(current_front_construction),
        "current_input_support": int(
            current_front_vehicle
            or current_front_vru
            or current_front_construction
            or current_traffic_control
        ),
        "learnable_risk_flag": int(learnable_risk),
        "event_labels": "|".join(labels) if labels else "ordinary",
        "center_speed_mps": center_speed,
        "min_future_speed_mps": min_future_speed,
        "max_future_speed_mps": max_future_speed,
        "final_forward_m": final_x,
        "final_lateral_m": final_y,
        "final_heading_rad": final_heading,
    }


def extract_log(path: Path, target_tokens: list[str]) -> list[dict[str, Any]]:
    targets = set(target_tokens)
    with path.open("rb") as handle:
        frames = load_navsim_log(handle)
    rows = []
    for start in range(0, len(frames) - NUM_FRAMES + 1):
        window = frames[start : start + NUM_FRAMES]
        token = str(window[NUM_HISTORY_FRAMES - 1]["token"])
        if token in targets:
            rows.append({"token": token, "log_name": path.stem, **label_window(window)})
    if {row["token"] for row in rows} != targets:
        missing = sorted(targets - {row["token"] for row in rows})
        raise ValueError(f"Raw log {path.stem} is missing target tokens: {missing[:5]}")
    return rows


def extract_features(raw_logs: Path, targets_by_log: dict[str, list[str]], workers: int) -> dict[str, dict[str, Any]]:
    raw_paths = {path.stem: path for path in raw_logs.rglob("*.pkl")}
    if len(raw_paths) != 1310:
        raise ValueError(f"Expected 1,310 unique NAVSIM logs, found {len(raw_paths)}")
    missing_logs = sorted(set(targets_by_log) - set(raw_paths))
    if missing_logs:
        raise ValueError(f"Missing raw logs: {missing_logs[:5]}")
    result: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_log, raw_paths[log_name], tokens): log_name
            for log_name, tokens in targets_by_log.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            for row in future.result():
                if row["token"] in result:
                    raise ValueError(f"Duplicate extracted token: {row['token']}")
                result[row["token"]] = row
            if completed % 100 == 0 or completed == len(futures):
                print(f"raw_logs={completed}/{len(futures)}", flush=True)
    return result


def add_train_role(row: dict[str, Any], stability: dict[str, str]) -> None:
    raw_flags = set(stability["raw_stability_flags"].split("|")) - {""}
    learnable_risk = bool(row["learnable_risk_flag"])
    stable_negative = learnable_risk and "stable_severe" in raw_flags
    paired_recovery = learnable_risk and "stable_mixed_recoverable" in raw_flags
    screen_mixed = learnable_risk and int(stability["screen_mixed_recoverable"]) == 1
    confirm_available = int(stability["candidate"]) == 1
    positive_supported = (
        learnable_risk
        and int(stability["screen_valid_rollouts"]) == 4
        and int(stability["screen_strict_clear_count"]) == 4
        and (
            not confirm_available
            or (
                int(stability["confirm_valid_rollouts"]) == 4
                and int(stability["confirm_strict_clear_count"]) == 4
            )
        )
    )
    if paired_recovery:
        role = "paired_recovery"
    elif stable_negative:
        role = "policy_negative"
    elif positive_supported:
        role = "positive_complex"
    elif learnable_risk:
        role = "unresolved_risk"
    else:
        role = "control"
    row.update(
        {
            "stability_category": stability["category"],
            "confirm_available": int(confirm_available),
            "screen_mixed_recoverable": int(stability["screen_mixed_recoverable"]),
            "stable_policy_negative": int(stable_negative),
            "confirmed_paired_recovery": int(paired_recovery),
            "positive_supported": int(positive_supported),
            "confirm_needed": int(screen_mixed and not confirm_available),
            "span_role": role,
        }
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    risk = [row for row in rows if bool(row["event_risk_flag"])]
    label_counts: Counter[str] = Counter()
    for row in risk:
        label_counts.update(str(row["event_labels"]).split("|"))
    result = {
        "scenes": len(rows),
        "logs": len({row["log_name"] for row in rows}),
        "event_risk_scenes": len(risk),
        "event_risk_rate": len(risk) / len(rows),
        "current_interaction_scenes": sum(bool(row["current_interaction_flag"]) for row in rows),
        "current_input_supported_risk_scenes": sum(bool(row["current_input_support"]) for row in risk),
        "learnable_risk_scenes": sum(bool(row["learnable_risk_flag"]) for row in rows),
        "event_label_counts": dict(sorted(label_counts.items())),
        "risk_by_intent": dict(sorted(Counter(row["intent"] for row in risk).items())),
        "risk_by_map": dict(sorted(Counter(row["map_location"] for row in risk).items())),
        "risk_by_month": dict(sorted(Counter(row["month"] for row in risk).items())),
    }
    if "span_role" in rows[0]:
        result["span_role_counts"] = dict(sorted(Counter(row["span_role"] for row in rows).items()))
        result["confirmed_policy_negative"] = sum(bool(row["stable_policy_negative"]) for row in rows)
        result["confirmed_paired_recovery"] = sum(bool(row["confirmed_paired_recovery"]) for row in rows)
        result["positive_supported"] = sum(bool(row["positive_supported"]) for row in rows)
        result["confirm_needed"] = sum(bool(row["confirm_needed"]) for row in rows)
        result["paired_recovery_overlap_with_policy_negative"] = sum(
            bool(row["confirmed_paired_recovery"]) and bool(row["stable_policy_negative"])
            for row in rows
        )
    return result


def build_report(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_summary = summarize(train_rows)
    dev_summary = summarize(dev_rows)
    capacity = {
        "positive_required": SPAN_SCALED_RECIPE["warmup_positive"] + SPAN_SCALED_RECIPE["mixed_positive"],
        "positive_available": train_summary["positive_supported"],
        "negative_required": SPAN_SCALED_RECIPE["mixed_negative"],
        "negative_available": train_summary["confirmed_policy_negative"],
        "recovery_required": SPAN_SCALED_RECIPE["mixed_recovery"],
        "recovery_available": train_summary["confirmed_paired_recovery"],
    }
    feasible = (
        capacity["positive_available"] >= capacity["positive_required"]
        and capacity["negative_available"] >= capacity["negative_required"]
        and capacity["recovery_available"] >= capacity["recovery_required"]
    )
    return {
        "status": "V4_SPAN_RISK_CAPACITY_AUDIT_COMPLETE",
        "semantic_boundary": {
            "paper_transfer": "SpanVLA-inspired event and negative-recovery pairing; not real-world takeover data",
            "evaluation_risk": "ground-truth expert-path horizon interaction or expert response to construction/traffic control",
            "training_behavior": "policy-conditioned stability is used only after the event-risk and current-input-support gate",
        },
        "thresholds": {
            "vehicle_distance_m": VEHICLE_DISTANCE_M,
            "vru_distance_m": VRU_DISTANCE_M,
            "construction_context_m": CONTEXT_DISTANCE_M,
            "front_half_angle_deg": math.degrees(FRONT_HALF_ANGLE_RAD),
            "turn_heading_deg": math.degrees(TURN_HEADING_RAD),
            "lateral_shift_m": LATERAL_SHIFT_M,
            "brake_speed_drop_mps": BRAKE_SPEED_DROP_MPS,
            "stop_speed_mps": STOP_SPEED_MPS,
            "go_speed_mps": GO_SPEED_MPS,
        },
        "coverage": {"train_screen": len(train_rows), "dev_all": len(dev_rows)},
        "train": train_summary,
        "dev": dev_summary,
        "scaled_span_recipe": SPAN_SCALED_RECIPE,
        "recipe_capacity": capacity,
        "recipe_status": "FEASIBLE" if feasible else "INSUFFICIENT_CONFIRMED_CAPACITY",
        "recommended_evaluation": "retain all fixed Dev tokens and report event strata plus controls; do not rebuild log-level Tail/Natural",
        "dev_accessed": True,
        "final_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-logs", type=Path, required=True)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--dev-natural", type=Path, required=True)
    parser.add_argument("--dev-tail", type=Path, required=True)
    parser.add_argument("--stability-capacity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    screen_tokens = read_manifest(args.screen_manifest)
    dev_natural = read_manifest(args.dev_natural)
    dev_tail = read_manifest(args.dev_tail)
    if len(screen_tokens) != 8000 or len(dev_natural) != 210 or len(dev_tail) != 206:
        raise ValueError("Frozen Screen/Dev manifests have unexpected coverage")
    if set(screen_tokens) & (set(dev_natural) | set(dev_tail)) or set(dev_natural) & set(dev_tail):
        raise ValueError("Screen and Dev manifests overlap")

    master_rows = read_csv(args.master_index)
    master = {row["token"]: row for row in master_rows}
    if len(master) != len(master_rows):
        raise ValueError("Master Index contains duplicate tokens")
    dev_split = {**{token: "dev_natural" for token in dev_natural}, **{token: "dev_tail" for token in dev_tail}}
    for token in screen_tokens:
        row = master[token]
        if row["source_universe"] != "sft_seen" or row["split"] != "grpo_screen":
            raise ValueError(f"Screen token escaped the frozen train universe: {token}")
    for token, split in dev_split.items():
        row = master[token]
        if row["source_universe"] != "sft_unseen" or row["split"] != split:
            raise ValueError(f"Dev token escaped the frozen Dev universe: {token}")

    stability_rows = read_csv(args.stability_capacity)
    stability = {row["token"]: row for row in stability_rows}
    if len(stability) != 8000 or set(stability) != set(screen_tokens):
        raise ValueError("Stability capacity does not exactly cover Screen")

    targets_by_log: dict[str, list[str]] = defaultdict(list)
    for token in screen_tokens + dev_natural + dev_tail:
        targets_by_log[master[token]["log_name"]].append(token)
    extracted = extract_features(args.raw_logs, targets_by_log, args.workers)
    expected = set(screen_tokens) | set(dev_natural) | set(dev_tail)
    if set(extracted) != expected:
        raise ValueError("Raw feature extraction does not exactly cover Screen and Dev")

    train_rows = []
    for token in screen_tokens:
        source = master[token]
        row = {
            **extracted[token],
            "source_universe": "sft_seen",
            "split": "grpo_screen",
            "intent": source["intent"],
            "map_location": source["map_location"],
            "month": source["log_name"][:7],
        }
        add_train_role(row, stability[token])
        train_rows.append(row)
    dev_rows = []
    for token in dev_natural + dev_tail:
        source = master[token]
        dev_rows.append(
            {
                **extracted[token],
                "source_universe": "sft_unseen",
                "split": dev_split[token],
                "intent": source["intent"],
                "map_location": source["map_location"],
                "month": source["log_name"][:7],
            }
        )

    report = build_report(train_rows, dev_rows)
    report["input_sha256"] = {
        "master_index": sha256_file(args.master_index),
        "screen_manifest": sha256_file(args.screen_manifest),
        "dev_natural": sha256_file(args.dev_natural),
        "dev_tail": sha256_file(args.dev_tail),
        "stability_capacity": sha256_file(args.stability_capacity),
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "train_scene_labels.csv", train_rows)
    write_csv(args.output_dir / "dev_scene_labels.csv", dev_rows)
    write_manifest(args.output_dir / "train_event_risk.txt", (row for row in train_rows if row["event_risk_flag"]), args.seed, "train-risk")
    write_manifest(args.output_dir / "train_learnable_risk.txt", (row for row in train_rows if row["learnable_risk_flag"]), args.seed, "train-learnable-risk")
    write_manifest(args.output_dir / "train_positive_complex.txt", (row for row in train_rows if row["positive_supported"]), args.seed, "train-positive")
    write_manifest(args.output_dir / "train_policy_negative.txt", (row for row in train_rows if row["stable_policy_negative"]), args.seed, "train-negative")
    write_manifest(args.output_dir / "train_paired_recovery.txt", (row for row in train_rows if row["confirmed_paired_recovery"]), args.seed, "train-recovery")
    write_manifest(args.output_dir / "train_confirm_needed.txt", (row for row in train_rows if row["confirm_needed"]), args.seed, "train-confirm")
    write_manifest(args.output_dir / "dev_event_risk.txt", (row for row in dev_rows if row["event_risk_flag"]), args.seed, "dev-risk")
    write_manifest(args.output_dir / "dev_event_control.txt", (row for row in dev_rows if not row["event_risk_flag"]), args.seed, "dev-control")
    (args.output_dir / "span_risk_capacity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
