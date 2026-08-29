from __future__ import annotations

import math
from typing import Any, Mapping


TIER_VALUE = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
AUDITED_EPS = 1e-6


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


def cdt_task_reward(parsed_ok: bool, metrics: Mapping[str, Any]) -> tuple[float, str | None]:
    tier = classify_cdt_tier(parsed_ok, metrics)
    if tier is None:
        return 0.0, None
    quality = task_quality(metrics)
    return (2.0 * TIER_VALUE[tier] + quality) / 7.0, tier
