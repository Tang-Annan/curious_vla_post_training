from collections import Counter

from projects.dataset_v3.v4_risk_ratio_audit import (
    RATIO_FAMILY_QUOTAS,
    candidate_rows,
    composition,
    maximize_primary_risk,
)


def test_maximize_primary_risk_respects_intent_and_log_constraints() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "token": f"risk-{index}",
                "log_name": f"risk-log-{index // 2}",
                "intent": ("straight", "left", "right")[index % 3],
                "exclusive_family": "proximity",
            }
        )
    for index in range(12):
        rows.append(
            {
                "token": f"context-{index}",
                "log_name": f"context-log-{index // 2}",
                "intent": ("straight", "left", "right")[index % 3],
                "exclusive_family": "signal",
            }
        )

    selected = maximize_primary_risk(
        rows,
        total=9,
        intent_quotas={"straight": 3, "left": 3, "right": 3},
        log_cap=2,
        seed=1,
    )

    assert len(selected) == 9
    assert Counter(row["intent"] for row in selected) == {"straight": 3, "left": 3, "right": 3}
    assert max(Counter(row["log_name"] for row in selected).values()) <= 2
    assert sum(row["exclusive_family"] == "proximity" for row in selected) == 9


def test_composition_reports_mutually_exclusive_families() -> None:
    classified = {
        family: {
            "token": family,
            "log_name": f"log-{family}",
            "intent": "straight",
            "exclusive_family": family,
        }
        for family in ("proximity", "construction", "signal", "control")
    }

    result = composition(list(classified), classified, name="test")

    assert result["scenes"] == 4
    assert result["proximity_count"] == 1
    assert result["construction_count"] == 1
    assert result["signal_count"] == 1
    assert result["control_count"] == 1


def test_candidate_rows_excludes_control() -> None:
    scene_rows = [
        {"token": "risk", "log_name": "log-risk", "intent": "straight"},
        {"token": "control", "log_name": "log-control", "intent": "left"},
    ]
    tier_rows = [
        {
            "token": "risk",
            "train_tier1": "1",
            "visible_critical_proximity": "1",
            "front_construction_response": "0",
            "current_signal_hard_response": "0",
        },
        {
            "token": "control",
            "train_tier1": "0",
            "visible_critical_proximity": "0",
            "front_construction_response": "0",
            "current_signal_hard_response": "0",
        },
    ]

    candidates, classified = candidate_rows(scene_rows, tier_rows)

    assert [row["token"] for row in candidates] == ["risk"]
    assert classified["control"]["exclusive_family"] == "control"


def test_ratio_trials_only_change_primary_risk_share() -> None:
    assert RATIO_FAMILY_QUOTAS == {
        "40": {"proximity": 800, "construction": 600, "signal": 600},
        "50": {"proximity": 1000, "construction": 500, "signal": 500},
        "60": {"proximity": 1200, "construction": 400, "signal": 400},
    }
