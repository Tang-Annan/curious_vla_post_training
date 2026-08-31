from __future__ import annotations

import math
from typing import Any, Mapping


TIER_VALUE = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
AUDITED_EPS = 1e-6

# V4 safety-first continuous reward constants (frozen 2026-08-31).
SAFETY_HARD_WEIGHT = 0.55
SAFETY_TTC_WEIGHT = 0.30
SAFETY_DISTANCE_WEIGHT = 0.15
QUALITY_GATE_WEIGHT = 0.25
QUALITY_PROGRESS_WEIGHT = 0.7
QUALITY_COMFORT_WEIGHT = 0.3
TTC_SAFE_SECONDS = 4.0
DISTANCE_SAFE_METERS = 5.0


def _bounded(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} is outside [0,1]: {numeric}")
    return numeric


def _canonical(value: Any, allowed: tuple[float, ...], name: str) -> float:
    numeric = float(value)
    matches = [candidate for candidate in allowed if abs(numeric - candidate) <= AUDITED_EPS + 1e-12]
    if len(matches) != 1:
        raise ValueError(f"{name}={numeric} is outside the audited evaluator values")
    return matches[0]


def _nonnegative(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative: {numeric}")
    return numeric


def time_to_infraction(metrics: Mapping[str, Any]) -> float:
    """
    Returns the earliest hazard time in seconds: actual at-fault collision or
    TTC infraction, whichever comes first. Returns infinity if neither occurs.
    """
    earliest = math.inf
    for field in ("time_to_at_fault_collision", "time_to_ttc_infraction"):
        if metrics[field] is None:
            continue
        value = float(metrics[field])
        if math.isfinite(value):
            if value < 0.0:
                raise ValueError(f"{field} must be non-negative: {value}")
            earliest = min(earliest, value)
    return earliest


def safety_hard_gate(metrics: Mapping[str, Any]) -> float:
    """
    Hard safety switch H: 1 iff no at-fault collision and drivable-area
    compliance hold, otherwise 0.
    """
    collision = _canonical(metrics["no_at_fault_collisions"], (0.0, 0.5, 1.0), "collision")
    drivable = _canonical(metrics["drivable_area_compliance"], (0.0, 1.0), "drivable")
    return 1.0 if collision == 1.0 and drivable == 1.0 else 0.0


def classify_cdt_tier(parsed_ok: bool, metrics: Mapping[str, Any]) -> str | None:
    if not parsed_ok:
        return None
    collision = _canonical(metrics["no_at_fault_collisions"], (0.0, 0.5, 1.0), "collision")
    drivable = _canonical(metrics["drivable_area_compliance"], (0.0, 1.0), "drivable")
    ttc = _canonical(metrics["time_to_collision_within_bound"], (0.0, 1.0), "ttc")
    if collision == 0.0:
        return "L0"
    if collision < 1.0 or drivable < 1.0:
        return "L1"
    if ttc < 1.0:
        return "L2"
    return "L3"


def task_quality(metrics: Mapping[str, Any]) -> float:
    progress = _bounded(metrics["ego_progress"], "ego_progress")
    comfort = _bounded(metrics["history_comfort"], "history_comfort")
    return (5.0 * progress + 2.0 * comfort) / 7.0


def raw_pdms_reward(metrics: Mapping[str, Any]) -> float:
    return _bounded(metrics["pdms"], "pdms")


def safety_continuous_reward(metrics: Mapping[str, Any]) -> float:
    """
    V4 safety-first continuous reward.

    R = (S + 0.25 * H * Q) / 1.25
    S = 0.55 * H + 0.30 * R_TTC + 0.15 * R_distance
    Q = 0.7 * ego_progress + 0.3 * history_comfort

    H = 1 iff no at-fault collision and drivable-area compliance hold.
    R_TTC caps the earliest hazard time at 4.0s; no hazard maps to 1.0.
    R_distance caps min polygon clearance to dynamic agents at 5.0m.
    """
    distance = (
        math.inf
        if metrics["min_distance_to_actors"] is None
        else _nonnegative(metrics["min_distance_to_actors"], "min_distance_to_actors")
    )
    progress = _bounded(metrics["ego_progress"], "ego_progress")
    comfort = _bounded(metrics["history_comfort"], "history_comfort")

    hard_safe = safety_hard_gate(metrics)
    ttc_seconds = time_to_infraction(metrics)
    r_ttc = 1.0 if not math.isfinite(ttc_seconds) else min(ttc_seconds / TTC_SAFE_SECONDS, 1.0)
    r_distance = min(distance / DISTANCE_SAFE_METERS, 1.0)
    safety = (
        SAFETY_HARD_WEIGHT * hard_safe
        + SAFETY_TTC_WEIGHT * r_ttc
        + SAFETY_DISTANCE_WEIGHT * r_distance
    )
    quality = QUALITY_PROGRESS_WEIGHT * progress + QUALITY_COMFORT_WEIGHT * comfort
    raw = safety + QUALITY_GATE_WEIGHT * hard_safe * quality
    denominator = SAFETY_HARD_WEIGHT + SAFETY_TTC_WEIGHT + SAFETY_DISTANCE_WEIGHT + QUALITY_GATE_WEIGHT
    return min(max(raw / denominator, 0.0), 1.0)


def cdt_task_reward(parsed_ok: bool, metrics: Mapping[str, Any]) -> tuple[float, str | None]:
    tier = classify_cdt_tier(parsed_ok, metrics)
    if tier is None:
        return 0.0, None
    quality = task_quality(metrics)
    return (2.0 * TIER_VALUE[tier] + quality) / 7.0, tier
