from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from projects.dataset_v3.data_prep import stable_key
from projects.dataset_v3.inventory import load_navsim_log, sha256_file
from projects.dataset_v3.s1_pipeline import read_manifest
from projects.dataset_v3.v4_experiment_closure import read_csv, write_csv
from projects.dataset_v3.v4_grpo_selector import selected_table
from projects.dataset_v3.v4_reward_audit import group_rows, read_jsonl


EXPECTED_SCREEN = 8000
TOTAL_SCENES = 2000
GROUP_SIZE = 4
FAMILY_QUOTAS = {"risk": 1000, "construction": 500, "signal": 500}
INTENT_QUOTAS = {"straight": 1333, "left": 434, "right": 233}
LOG_CAP = 4
SEED = 20260904
HORIZON_S = 4.0
MIN_CLOSING_SPEED_MPS = 0.5
FRONT_HALF_ANGLE_RAD = math.radians(45.0)
IMMEDIATE_SUPPORT_M = 20.0
PROJECTED_SUPPORT_M = 40.0
IMMEDIATE_RADIUS_M = {"vehicle": 5.0, "vru": 10.0}
CONFLICT_RADIUS_M = {"vehicle": 3.0, "vru": 5.0}
VRU_NAMES = {"pedestrian", "bicycle"}
HEADROOM_THRESHOLDS = (0.0025, 0.005, 0.01)


def _actor_kind(name: str) -> str | None:
    if name == "vehicle":
        return "vehicle"
    if name in VRU_NAMES:
        return "vru"
    return None


def _finite_pair(values: Any) -> tuple[float, float] | None:
    try:
        pair = float(values[0]), float(values[1])
    except (IndexError, TypeError, ValueError):
        return None
    return pair if all(math.isfinite(value) for value in pair) else None


def _front_supported(x: float, y: float, max_distance: float) -> bool:
    return (
        x > 0.0
        and math.hypot(x, y) <= max_distance
        and abs(math.atan2(y, x)) <= FRONT_HALF_ANGLE_RAD
    )


def _time_to_radius(
    x: float, y: float, vx: float, vy: float, radius: float
) -> float | None:
    c = x * x + y * y - radius * radius
    if c <= 0.0:
        return 0.0
    a = vx * vx + vy * vy
    b = 2.0 * (x * vx + y * vy)
    if a <= 1e-8 or b >= 0.0:
        return None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    entry = (-b - math.sqrt(discriminant)) / (2.0 * a)
    return entry if entry >= 0.0 else None


def current_state_risk(frame: dict[str, Any]) -> dict[str, Any]:
    anns = frame["anns"]
    annotation_keys = ("gt_names", "gt_boxes", "gt_velocity_3d", "track_tokens")
    lengths = {key: len(anns[key]) for key in annotation_keys}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Annotation arrays are not aligned: {lengths}")
    track_tokens = [str(track) for track in anns["track_tokens"]]
    if any(not track for track in track_tokens) or len(track_tokens) != len(set(track_tokens)):
        raise ValueError("Current-frame track tokens must be non-empty and unique")
    ego_velocity = _finite_pair(frame["ego_dynamic_state"])
    if ego_velocity is None:
        raise ValueError("Current ego velocity is invalid")

    immediate_tracks: set[str] = set()
    projected_tracks: set[str] = set()
    lateral_tracks: set[str] = set()
    reason_flags = Counter()
    for raw_name, box, velocity, track in zip(
        anns["gt_names"], anns["gt_boxes"], anns["gt_velocity_3d"], track_tokens
    ):
        actor_kind = _actor_kind(str(raw_name))
        if actor_kind is None:
            continue
        position = _finite_pair(box)
        actor_velocity = _finite_pair(velocity)
        if position is None or actor_velocity is None:
            raise ValueError(f"Actor {track} has invalid current position or velocity")
        x, y = position
        vx = actor_velocity[0] - ego_velocity[0]
        vy = actor_velocity[1] - ego_velocity[1]
        distance = math.hypot(x, y)
        radial_closing = -(x * vx + y * vy) / distance if distance > 1e-8 else math.inf
        immediate = (
            _front_supported(x, y, IMMEDIATE_SUPPORT_M)
            and distance <= IMMEDIATE_RADIUS_M[actor_kind]
        )
        entry = _time_to_radius(x, y, vx, vy, CONFLICT_RADIUS_M[actor_kind])
        projected = (
            _front_supported(x, y, PROJECTED_SUPPORT_M)
            and radial_closing >= MIN_CLOSING_SPEED_MPS
            and entry is not None
            and entry <= HORIZON_S
        )
        lateral = (
            projected
            and abs(y) > CONFLICT_RADIUS_M[actor_kind]
            and y * vy < 0.0
            and abs(vy) >= MIN_CLOSING_SPEED_MPS
        )
        if immediate:
            immediate_tracks.add(track)
            reason_flags[f"immediate_{actor_kind}"] = 1
        if projected:
            projected_tracks.add(track)
            reason_flags[f"projected_{actor_kind}"] = 1
        if lateral:
            lateral_tracks.add(track)

    risk_tracks = immediate_tracks | projected_tracks
    return {
        "primary_risk": int(bool(risk_tracks)),
        "immediate_proximity": int(bool(immediate_tracks)),
        "projected_conflict": int(bool(projected_tracks)),
        "dynamic_addition": int(bool(projected_tracks) and not immediate_tracks),
        "lateral_convergence": int(bool(lateral_tracks)),
        "immediate_vehicle": reason_flags["immediate_vehicle"],
        "immediate_vru": reason_flags["immediate_vru"],
        "projected_vehicle": reason_flags["projected_vehicle"],
        "projected_vru": reason_flags["projected_vru"],
        "risk_actor_track_tokens": "|".join(sorted(risk_tracks)),
        "immediate_actor_track_tokens": "|".join(sorted(immediate_tracks)),
        "projected_actor_track_tokens": "|".join(sorted(projected_tracks)),
        "lateral_actor_track_tokens": "|".join(sorted(lateral_tracks)),
    }


def _extract_log(path: Path, target_tokens: list[str]) -> list[dict[str, Any]]:
    targets = set(target_tokens)
    with path.open("rb") as handle:
        frames = load_navsim_log(handle)
    rows = [
        {"token": str(frame["token"]), **current_state_risk(frame)}
        for frame in frames
        if str(frame.get("token", "")) in targets
    ]
    found = {row["token"] for row in rows}
    if found != targets or len(rows) != len(found):
        raise ValueError(f"Raw log {path.stem} did not uniquely cover its target tokens")
    return rows


def extract_risk_labels(
    raw_logs: Path,
    targets_by_log: dict[str, list[str]],
    workers: int,
) -> dict[str, dict[str, Any]]:
    raw_paths = {path.stem: path for path in raw_logs.rglob("*.pkl")}
    missing_logs = set(targets_by_log) - set(raw_paths)
    if missing_logs:
        raise ValueError(f"Missing raw logs: {sorted(missing_logs)[:5]}")
    result: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_extract_log, raw_paths[log_name], tokens): log_name
            for log_name, tokens in targets_by_log.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            for row in future.result():
                token = str(row["token"])
                if token in result:
                    raise ValueError(f"Duplicate extracted token: {token}")
                result[token] = row
            if completed % 100 == 0 or completed == len(futures):
                print(f"raw_logs={completed}/{len(futures)}", flush=True)
    return result


def _truthy(row: dict[str, str], key: str) -> bool:
    return row[key] == "1"


def exclusive_family(scene_label: dict[str, str], primary_risk: bool) -> str:
    if primary_risk:
        return "risk"
    expert_response = any(
        _truthy(scene_label, key)
        for key in ("expert_turn", "expert_lateral", "expert_braking", "expert_stop_to_go")
    )
    construction = (
        _truthy(scene_label, "construction_present")
        and _truthy(scene_label, "current_construction_front_context")
        and expert_response
    )
    if construction:
        return "construction"
    signal = _truthy(scene_label, "current_traffic_control") and (
        _truthy(scene_label, "expert_braking")
        or _truthy(scene_label, "expert_stop_to_go")
    )
    return "signal" if signal else "control"


def _strict_clear(row: dict[str, Any]) -> bool:
    return bool(row["parsed_ok"]) and all(
        float(row[field]) == 1.0
        for field in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
        )
    )


def _raw_pdms(row: dict[str, Any]) -> float:
    value = float(row["pdms"]) if bool(row["parsed_ok"]) else 0.0
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"raw PDMS is outside [0, 1]: {value}")
    return value


def build_fals_features(
    rollout_rows: list[dict[str, Any]], tokens: list[str]
) -> dict[str, dict[str, Any]]:
    groups = group_rows(rollout_rows, tokens, GROUP_SIZE)
    features = {}
    for token in tokens:
        rows = groups[token]
        rewards = [_raw_pdms(row) for row in rows]
        mean = statistics.fmean(rewards)
        best = max(rewards)
        headroom = best - mean
        difficulty = 1.0 - mean
        strict_clear_count = sum(_strict_clear(row) for row in rows)
        features[token] = {
            "mean_raw_pdms": mean,
            "best_raw_pdms": best,
            "difficulty": difficulty,
            "headroom": headroom,
            "fals": difficulty * headroom,
            "fals_positive": int(difficulty * headroom > 0.0),
            "strict_clear_count": strict_clear_count,
            "strict_clear_mixed": int(0 < strict_clear_count < GROUP_SIZE),
        }
    return features


def _stable_cost(namespace: str, token: str) -> float:
    return int(stable_key(SEED, namespace, token)[:13], 16) / float(16**13)


def _solve(
    rows: list[dict[str, Any]],
    objective: Callable[[dict[str, Any]], float],
    *,
    family_quotas: dict[str, int],
    intent_quotas: dict[str, int],
    log_cap: int,
    exact_features: list[tuple[Callable[[dict[str, Any]], bool], int]] | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["token"]))
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(indices: list[int], low: float, high: float) -> None:
        constraint_index = len(lower)
        for index in indices:
            matrix_rows.append(constraint_index)
            matrix_columns.append(index)
            matrix_values.append(1.0)
        lower.append(low)
        upper.append(high)

    add(list(range(len(ordered))), TOTAL_SCENES, TOTAL_SCENES)
    for family, quota in family_quotas.items():
        add(
            [index for index, row in enumerate(ordered) if row["exclusive_family"] == family],
            quota,
            quota,
        )
    for intent, quota in intent_quotas.items():
        add(
            [index for index, row in enumerate(ordered) if row["intent"] == intent],
            quota,
            quota,
        )
    by_log: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        by_log[str(row["log_name"])].append(index)
    for indices in by_log.values():
        add(indices, 0, log_cap)
    for predicate, value in exact_features or []:
        add([index for index, row in enumerate(ordered) if predicate(row)], value, value)

    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(lower), len(ordered)),
    ).tocsr()
    result = milp(
        c=np.asarray([objective(row) for row in ordered]),
        integrality=np.ones(len(ordered)),
        bounds=Bounds(np.zeros(len(ordered)), np.ones(len(ordered))),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 300},
    )
    if not result.success or result.x is None:
        raise ValueError(f"Exact 2K MILP is infeasible: {result.message}")
    selected = [row for row, value in zip(ordered, result.x) if value > 0.5]
    if len(selected) != TOTAL_SCENES:
        raise ValueError("MILP returned a non-exact selection")
    return selected


def select_risk50(
    rows: list[dict[str, Any]],
    *,
    family_quotas: dict[str, int] = FAMILY_QUOTAS,
    intent_quotas: dict[str, int] = INTENT_QUOTAS,
    log_cap: int = LOG_CAP,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["exclusive_family"] in family_quotas]
    return _solve(
        candidates,
        lambda row: _stable_cost("v5-risk50", str(row["token"])),
        family_quotas=family_quotas,
        intent_quotas=intent_quotas,
        log_cap=log_cap,
    )


def select_risk50_fals(
    rows: list[dict[str, Any]],
    *,
    family_quotas: dict[str, int] = FAMILY_QUOTAS,
    intent_quotas: dict[str, int] = INTENT_QUOTAS,
    log_cap: int = LOG_CAP,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates = [row for row in rows if row["exclusive_family"] in family_quotas]
    risk_fals = lambda row: row["exclusive_family"] == "risk" and bool(row["fals_positive"])
    any_fals = lambda row: bool(row["fals_positive"])
    mixed = lambda row: bool(row["strict_clear_mixed"])

    stage1 = _solve(
        candidates,
        lambda row: -float(risk_fals(row)),
        family_quotas=family_quotas,
        intent_quotas=intent_quotas,
        log_cap=log_cap,
    )
    max_risk_fals = sum(risk_fals(row) for row in stage1)
    stage2 = _solve(
        candidates,
        lambda row: -float(any_fals(row)),
        family_quotas=family_quotas,
        intent_quotas=intent_quotas,
        log_cap=log_cap,
        exact_features=[(risk_fals, max_risk_fals)],
    )
    max_total_fals = sum(any_fals(row) for row in stage2)
    stage3 = _solve(
        candidates,
        lambda row: -float(mixed(row)),
        family_quotas=family_quotas,
        intent_quotas=intent_quotas,
        log_cap=log_cap,
        exact_features=[(risk_fals, max_risk_fals), (any_fals, max_total_fals)],
    )
    max_mixed = sum(mixed(row) for row in stage3)
    selected = _solve(
        candidates,
        lambda row: -float(row["fals"])
        + 1e-9 * _stable_cost("v5-risk50-fals", str(row["token"])),
        family_quotas=family_quotas,
        intent_quotas=intent_quotas,
        log_cap=log_cap,
        exact_features=[
            (risk_fals, max_risk_fals),
            (any_fals, max_total_fals),
            (mixed, max_mixed),
        ],
    )
    return selected, {
        "max_risk_fals": max_risk_fals,
        "max_total_fals": max_total_fals,
        "max_mixed_after_fals": max_mixed,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    logs = Counter(str(row["log_name"]) for row in rows)
    return {
        "scenes": len(rows),
        "family_counts": dict(sorted(Counter(str(row["exclusive_family"]) for row in rows).items())),
        "intent_counts": dict(sorted(Counter(str(row["intent"]) for row in rows).items())),
        "unique_logs": len(logs),
        "max_per_log": max(logs.values()) if logs else 0,
        "fals_positive": sum(bool(row["fals_positive"]) for row in rows),
        "fals_positive_by_family": {
            family: sum(
                row["exclusive_family"] == family and bool(row["fals_positive"])
                for row in rows
            )
            for family in ("risk", "construction", "signal", "control")
        },
        "headroom_sensitivity": {
            str(threshold): sum(float(row["headroom"]) >= threshold for row in rows)
            for threshold in HEADROOM_THRESHOLDS
        },
        "strict_clear_mixed": sum(bool(row["strict_clear_mixed"]) for row in rows),
        "mean_raw_pdms": statistics.fmean(float(row["mean_raw_pdms"]) for row in rows),
        "mean_headroom": statistics.fmean(float(row["headroom"]) for row in rows),
        "mean_fals": statistics.fmean(float(row["fals"]) for row in rows),
        "risk_reasons": {
            key: sum(bool(row[key]) for row in rows)
            for key in (
                "primary_risk",
                "immediate_proximity",
                "projected_conflict",
                "dynamic_addition",
                "lateral_convergence",
                "immediate_vehicle",
                "immediate_vru",
                "projected_vehicle",
                "projected_vru",
            )
        },
    }


def _ordered_tokens(rows: list[dict[str, Any]], namespace: str) -> list[str]:
    return [
        str(row["token"])
        for row in sorted(rows, key=lambda row: _stable_cost(namespace, str(row["token"])))
    ]


def _materialize(
    output_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    screen_parquet: Path,
    data_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    tokens = _ordered_tokens(rows, f"v5-output-{name}")
    manifest = output_dir / f"{name}_2000.txt"
    parquet = output_dir / f"{name}_2000.parquet"
    manifest.write_text("".join(f"{token}\n" for token in tokens), encoding="utf-8")
    table, report = selected_table(screen_parquet, tokens, data_root)
    pq.write_table(table, parquet)
    written = pq.read_table(parquet)
    if written.num_rows != TOTAL_SCENES or written.schema != table.schema:
        raise ValueError(f"Written {name} parquet failed row or schema verification")
    return manifest, parquet, report


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite V5 dataset run: {args.output_dir}")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    screen_tokens = read_manifest(args.screen_manifest)
    monitor_tokens = read_manifest(args.monitor_manifest)
    if len(screen_tokens) != EXPECTED_SCREEN or len(screen_tokens) != len(set(screen_tokens)):
        raise ValueError("Frozen Screen must contain exactly 8,000 unique tokens")
    if set(screen_tokens) & set(monitor_tokens):
        raise ValueError("Frozen Screen overlaps the Train Monitor")

    master_rows = read_csv(args.master_index)
    screen_set = set(screen_tokens)
    master = {row["token"]: row for row in master_rows if row["token"] in screen_set}
    scene_labels = {row["token"]: row for row in read_csv(args.scene_labels)}
    if set(master) != screen_set or set(scene_labels) != screen_set:
        raise ValueError("Master index or scene labels do not exactly cover Screen")
    targets_by_log: dict[str, list[str]] = defaultdict(list)
    for token in screen_tokens:
        targets_by_log[master[token]["log_name"]].append(token)
    risk = extract_risk_labels(args.raw_logs, targets_by_log, args.workers)
    if set(risk) != screen_set:
        raise ValueError("Current-state risk extraction does not exactly cover Screen")
    fals = build_fals_features(read_jsonl(args.screen_enriched), screen_tokens)

    rows = []
    for token in screen_tokens:
        risk_row = risk[token]
        family = exclusive_family(scene_labels[token], bool(risk_row["primary_risk"]))
        rows.append(
            {
                "token": token,
                "log_name": master[token]["log_name"],
                "intent": master[token]["intent"],
                "exclusive_family": family,
                **risk_row,
                **fals[token],
            }
        )

    pool_summary = _summary(rows)
    expected_pool = {
        "risk": 1373,
        "immediate": 361,
        "projected": 1205,
        "dynamic_addition": 1012,
        "lateral": 118,
        "fals_positive": 5617,
        "risk_fals_positive": 953,
    }
    observed_pool = {
        "risk": pool_summary["risk_reasons"]["primary_risk"],
        "immediate": pool_summary["risk_reasons"]["immediate_proximity"],
        "projected": pool_summary["risk_reasons"]["projected_conflict"],
        "dynamic_addition": pool_summary["risk_reasons"]["dynamic_addition"],
        "lateral": pool_summary["risk_reasons"]["lateral_convergence"],
        "fals_positive": pool_summary["fals_positive"],
        "risk_fals_positive": pool_summary["fals_positive_by_family"]["risk"],
    }
    if observed_pool != expected_pool:
        raise ValueError(f"V5 preregistered pool distribution changed: {observed_pool}")

    risk50 = select_risk50(rows)
    risk50_fals, fals_optima = select_risk50_fals(rows)
    risk50_summary = _summary(risk50)
    fals_summary = _summary(risk50_fals)
    if fals_optima["max_risk_fals"] != 940 or fals_optima["max_total_fals"] != 1895:
        raise ValueError(f"V5 FALS capacity changed: {fals_optima}")
    expected_fals_by_family = {"risk": 940, "construction": 455, "signal": 500, "control": 0}
    if fals_summary["fals_positive_by_family"] != expected_fals_by_family:
        raise ValueError(
            "V5 FALS family distribution changed: "
            f"{fals_summary['fals_positive_by_family']}"
        )

    args.output_dir.mkdir(parents=True)
    risk50_manifest, risk50_parquet, risk50_materialized = _materialize(
        args.output_dir, "v5_risk50", risk50, args.screen_parquet, args.data_root
    )
    fals_manifest, fals_parquet, fals_materialized = _materialize(
        args.output_dir, "v5_risk50_fals", risk50_fals, args.screen_parquet, args.data_root
    )
    risk50_set = {str(row["token"]) for row in risk50}
    fals_set = {str(row["token"]) for row in risk50_fals}
    if (risk50_set | fals_set) & set(monitor_tokens):
        raise ValueError("A V5 optimizer dataset overlaps Train Monitor")
    membership = [
        {
            **row,
            "selected_risk50": int(str(row["token"]) in risk50_set),
            "selected_risk50_fals": int(str(row["token"]) in fals_set),
            "fals_role": (
                f"{row['exclusive_family']}_fals"
                if row["fals_positive"]
                else f"{row['exclusive_family']}_anchor"
            ),
        }
        for row in sorted(rows, key=lambda row: str(row["token"]))
    ]
    membership_path = args.output_dir / "v5_scene_fals_membership.csv"
    write_csv(membership_path, membership)

    overlap = len(risk50_set & fals_set)
    report = {
        "status": "V5_RISK_FALS_DATASETS_READY",
        "semantic_definition": {
            "primary_risk": "same-current-actor immediate proximity OR current-kinematics 4s projected conflict",
            "actor_identity": "position, velocity, type, and emitted reason use the same current-frame track_token",
            "immediate_radius_m": IMMEDIATE_RADIUS_M,
            "projected_conflict_radius_m": CONFLICT_RADIUS_M,
            "minimum_closing_speed_mps": MIN_CLOSING_SPEED_MPS,
            "fals": "(1 - mean(raw_pdms_G4)) * (max(raw_pdms_G4) - mean(raw_pdms_G4))",
        },
        "selection_protocol": {
            "family_quotas": FAMILY_QUOTAS,
            "intent_quotas": INTENT_QUOTAS,
            "max_per_log": LOG_CAP,
            "risk50": "deterministic stable-hash selection; FALS is not used",
            "risk50_fals": [
                "maximize FALS-positive risk count",
                "maximize total FALS-positive count",
                "maximize StrictClear-mixed count",
                "maximize FALS, then stable-hash tie-break",
            ],
        },
        "pool": pool_summary,
        "preregistered_pool_check": {
            "expected": expected_pool,
            "observed": observed_pool,
            "passed": True,
        },
        "datasets": {
            "risk50": {
                "summary": risk50_summary,
                "materialized": risk50_materialized,
            },
            "risk50_fals": {
                "summary": fals_summary,
                "lexicographic_optima": fals_optima,
                "anchors": TOTAL_SCENES - fals_summary["fals_positive"],
                "materialized": fals_materialized,
            },
            "overlap": overlap,
            "jaccard": overlap / (2 * TOTAL_SCENES - overlap),
        },
        "future_training_order": [
            {"cell": "V5-RISK50", "reward": "Raw-PDMS"},
            {"cell": "V5-RISK50-FALS", "reward": "Raw-PDMS"},
        ],
        "input_sha256": {
            "screen_manifest": sha256_file(args.screen_manifest),
            "monitor_manifest": sha256_file(args.monitor_manifest),
            "master_index": sha256_file(args.master_index),
            "scene_labels": sha256_file(args.scene_labels),
            "screen_enriched": sha256_file(args.screen_enriched),
            "screen_parquet": sha256_file(args.screen_parquet),
        },
        "output_sha256": {
            "risk50_manifest": sha256_file(risk50_manifest),
            "risk50_parquet": sha256_file(risk50_parquet),
            "risk50_fals_manifest": sha256_file(fals_manifest),
            "risk50_fals_parquet": sha256_file(fals_parquet),
            "membership": sha256_file(membership_path),
        },
        "dev_accessed": False,
        "final_accessed": False,
        "gpu_used": False,
        "training_launched": False,
    }
    report_path = args.output_dir / "v5_risk_fals_dataset_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-logs", type=Path, required=True)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--monitor-manifest", type=Path, required=True)
    parser.add_argument("--scene-labels", type=Path, required=True)
    parser.add_argument("--screen-enriched", type=Path, required=True)
    parser.add_argument("--screen-parquet", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
