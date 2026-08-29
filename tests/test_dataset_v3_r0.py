import copy
import importlib.util
from pathlib import Path

import pytest

from projects.dataset_v3.r0_freeze import SEMANTIC_STATUS, freeze_protocol
from projects.dataset_v3.r0_geometry import reward_value, task_quality

_PRODUCTION_REWARD_PATH = (
    Path(__file__).parents[1] / "EasyR1/verl/utils/reward_score/navsim/cdt_scalar_reward.py"
)
_SPEC = importlib.util.spec_from_file_location("cdt_scalar_reward", _PRODUCTION_REWARD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PRODUCTION_REWARD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PRODUCTION_REWARD)
cdt_task_reward = _PRODUCTION_REWARD.cdt_task_reward
classify_cdt_tier = _PRODUCTION_REWARD.classify_cdt_tier


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


def test_production_cdt_matches_geometry_formula() -> None:
    for row in (
        metric_row(collision=0.0),
        metric_row(collision=0.5),
        metric_row(ttc=0.0),
        metric_row(),
    ):
        expected, _, tier = reward_value(row, "cdt_task")
        actual, production_tier = cdt_task_reward(True, row)
        assert production_tier == tier == classify_cdt_tier(True, row)
        assert actual == pytest.approx(expected)


def geometry_report() -> dict:
    cells = {}
    for selector in ("random", "tailmix"):
        cells[f"{selector}_raw_pdms"] = {
            "effective_group_rate": 0.7,
            "exact_zero_rate": 0.3,
            "low_nonzero_rate": 0.5,
            "cross_tier_inversions": 0,
            "cross_tier_ties": 1,
            "within_tier_quality_inversions_or_ties": 0,
        }
        cells[f"{selector}_cdt_task"] = {
            "effective_group_rate": 0.71,
            "exact_zero_rate": 0.29,
            "low_nonzero_rate": 0.6,
            "cross_tier_inversions": 0,
            "cross_tier_ties": 0,
            "within_tier_quality_inversions_or_ties": 0,
        }
    return {
        "status": "CANDIDATE_GEOMETRY_ONLY_NOT_REWARD_FREEZE",
        "task_quality_audit": {"semantic_status": SEMANTIC_STATUS},
        "cells": cells,
    }


def test_reward_freeze_accepts_only_pre_registered_task_gate() -> None:
    protocol = freeze_protocol(geometry_report())
    assert protocol["reward_id"] == "R_TASK_CDT_V3"
    assert protocol["production_function"] == "compute_score_cdt_task"


def test_reward_freeze_closes_on_ordering_failure() -> None:
    report = copy.deepcopy(geometry_report())
    report["cells"]["tailmix_cdt_task"]["cross_tier_ties"] = 1
    with pytest.raises(ValueError, match="cross-tier ordering"):
        freeze_protocol(report)
