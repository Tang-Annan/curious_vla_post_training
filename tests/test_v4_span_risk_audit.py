import math

import numpy as np

from projects.dataset_v3.span_risk_audit import add_train_role, build_report, label_window


def _frame(index: int, *, lateral: float = 0.0, names=(), boxes=(), traffic=False) -> dict:
    return {
        "token": f"token-{index}",
        "timestamp": index * 500_000,
        "ego2global_translation": np.array([index * 2.0, lateral, 0.0]),
        "ego2global_rotation": np.array([1.0, 0.0, 0.0, 0.0]),
        "anns": {
            "gt_names": np.array(names),
            "gt_boxes": np.array(boxes, dtype=float).reshape((-1, 7)),
        },
        "traffic_lights": [("lane", "red")] if traffic else [],
    }


def test_window_labels_future_interaction_and_expert_response() -> None:
    window = [_frame(index) for index in range(14)]
    window[6] = _frame(6, lateral=2.0, names=("vehicle", "traffic_cone"), boxes=((4, 0, 0, 4, 2, 2, 0), (8, 1, 0, 1, 1, 1, 0)))
    for index in range(7, 14):
        window[index] = _frame(index, lateral=2.0)

    row = label_window(window)

    assert row["current_interaction_flag"] == 0
    assert row["horizon_vehicle_interaction"] == 1
    assert row["construction_present"] == 1
    assert row["expert_lateral"] == 1
    assert row["event_risk_flag"] == 1
    assert row["learnable_risk_flag"] == 0
    assert "vehicle_interaction" in row["event_labels"]
    assert math.isfinite(row["center_speed_mps"])


def test_train_role_requires_event_gate_before_rollout_semantics() -> None:
    stability = {
        "raw_stability_flags": "stable_severe|stable_mixed_recoverable",
        "category": "stable_severe",
        "candidate": "1",
        "screen_mixed_recoverable": "1",
        "screen_valid_rollouts": "4",
        "screen_strict_clear_count": "2",
        "confirm_valid_rollouts": "4",
        "confirm_strict_clear_count": "2",
    }
    risk = {"event_risk_flag": 1, "learnable_risk_flag": 1}
    add_train_role(risk, stability)
    assert risk["stable_policy_negative"] == 1
    assert risk["confirmed_paired_recovery"] == 1
    assert risk["span_role"] == "paired_recovery"

    control = {"event_risk_flag": 1, "learnable_risk_flag": 0}
    add_train_role(control, stability)
    assert control["stable_policy_negative"] == 0
    assert control["confirmed_paired_recovery"] == 0
    assert control["span_role"] == "control"


def test_report_blocks_scaled_recipe_when_recovery_capacity_is_small() -> None:
    common = {
        "log_name": "log",
        "intent": "straight",
        "map_location": "map",
        "month": "2026.08",
        "current_interaction_flag": 1,
        "current_input_support": 1,
        "learnable_risk_flag": 1,
        "event_labels": "vehicle_interaction",
    }
    train = [
        {
            **common,
            "event_risk_flag": 1,
            "span_role": "paired_recovery",
            "stable_policy_negative": 1,
            "confirmed_paired_recovery": 1,
            "positive_supported": 0,
            "confirm_needed": 0,
        }
    ]
    dev = [{**common, "event_risk_flag": 1}]

    report = build_report(train, dev)

    assert report["recipe_status"] == "INSUFFICIENT_CONFIRMED_CAPACITY"
    assert report["recipe_capacity"]["recovery_available"] == 1
    assert report["final_accessed"] is False
