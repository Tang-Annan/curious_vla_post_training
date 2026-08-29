import pytest

from projects.dataset_v3.r0_geometry import reward_value, task_quality


def metric_row(
    *, collision: float = 1.0, drivable: float = 1.0, ttc: float = 1.0, progress: float = 0.5, comfort: float = 1.0
) -> dict:
    return {
        "parsed_ok": True,
        "no_at_fault_collisions": collision,
        "drivable_area_compliance": drivable,
        "time_to_collision_within_bound": ttc,
        "ego_progress": progress,
        "history_comfort": comfort,
        "pdms": collision * drivable * (5 * progress + 5 * ttc + 2 * comfort) / 12,
    }


def test_task_quality_is_exact_non_safety_pdms_remainder() -> None:
    assert task_quality(metric_row(progress=0.5, comfort=1.0)) == pytest.approx(4.5 / 7)


def test_cdt_task_intervals_are_strictly_non_overlapping() -> None:
    lower_high, _, lower_tier = reward_value(
        metric_row(collision=0.0, progress=1.0, comfort=1.0), "cdt_task"
    )
    upper_low, _, upper_tier = reward_value(
        metric_row(collision=0.5, progress=0.0, comfort=0.0), "cdt_task"
    )
    assert lower_tier == "L0"
    assert upper_tier == "L1"
    assert upper_low - lower_high == pytest.approx(1 / 7)


def test_invalid_is_not_a_tier_and_receives_technical_zero() -> None:
    row = metric_row()
    row["parsed_ok"] = False
    reward, quality, tier = reward_value(row, "cdt_task")
    assert reward == 0.0
    assert quality == pytest.approx(4.5 / 7)
    assert tier is None


def test_raw_reward_is_unscaled_pdms() -> None:
    row = metric_row(progress=0.8)
    reward, _, _ = reward_value(row, "raw_pdms")
    assert reward == pytest.approx(row["pdms"])
