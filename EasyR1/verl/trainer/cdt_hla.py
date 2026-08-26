"""Canonical CDT safety semantics shared by training and Dataset V2 replay."""

from __future__ import annotations

import math


CDT_EPS = 1e-6
TIER_VALUE = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def _canonicalize(value: float, allowed: tuple[float, ...], name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} is not finite: {numeric}")
    matches = [candidate for candidate in allowed if abs(numeric - candidate) <= CDT_EPS + 1e-12]
    if len(matches) != 1:
        raise ValueError(f"{name}={numeric} cannot be mapped to {allowed}")
    return matches[0]


def classify_cdt(
    parsed_ok: bool,
    no_at_fault_collisions: float,
    drivable_area_compliance: float,
    time_to_collision_within_bound: float,
) -> str | None:
    """Return L0-L3 for a parsed rollout; parse failures are not safety tiers."""
    if not bool(parsed_ok):
        return None
    collision = _canonicalize(no_at_fault_collisions, (0.0, 0.5, 1.0), "no_at_fault_collisions")
    drivable = _canonicalize(drivable_area_compliance, (0.0, 1.0), "drivable_area_compliance")
    ttc = _canonicalize(time_to_collision_within_bound, (0.0, 1.0), "time_to_collision_within_bound")
    if collision == 0.0:
        return "L0"
    if collision < 1.0 or drivable < 1.0:
        return "L1"
    if ttc < 1.0:
        return "L2"
    return "L3"


def is_strict_clear(
    parsed_ok: bool,
    no_at_fault_collisions: float,
    drivable_area_compliance: float,
    time_to_collision_within_bound: float,
) -> bool:
    return (
        classify_cdt(
            parsed_ok,
            no_at_fault_collisions,
            drivable_area_compliance,
            time_to_collision_within_bound,
        )
        == "L3"
    )


def compute_cdtr(tier: str, quality: float) -> float:
    if tier not in TIER_VALUE:
        raise ValueError(f"Unknown CDT tier: {tier}")
    within = float(quality) if tier in {"L2", "L3"} else 0.0
    return (2.0 * TIER_VALUE[tier] + within) / 7.0
