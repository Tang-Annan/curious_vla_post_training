"""Pure Safety-Lexicographic Dense Reward calculation."""

from typing import Any


REQUIRED_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
    "pdms",
    "pdms_scaled",
)


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_sldr(metrics: dict[str, Any]) -> float:
    """Return a reward in which every safe trajectory outranks every unsafe one."""
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise KeyError(f"Missing NAVSIM metrics: {', '.join(missing)}")

    no_collision = _clip01(metrics["no_at_fault_collisions"])
    drivable = _clip01(metrics["drivable_area_compliance"])
    progress = _clip01(metrics["ego_progress"])
    ttc = _clip01(metrics["time_to_collision_within_bound"])
    comfort = _clip01(metrics["history_comfort"])
    safe = float(no_collision > 0.0 and drivable > 0.0)

    progress_focal = 1.0 - (1.0 - progress) ** 0.6
    ttc_focal = 1.0 - (1.0 - ttc) ** 0.6
    quality = (5.0 * progress_focal + 5.0 * ttc_focal + 2.0 * comfort) / 12.0
    return 0.5 * safe + 0.5 * safe * quality + 0.1 * (1.0 - safe) * ttc
