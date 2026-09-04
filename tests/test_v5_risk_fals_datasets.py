from collections import Counter

import pytest

import projects.dataset_v3.v5_risk_fals_datasets as v5


def frame(actors: list[tuple[str, list[float], list[float], str]], ego_vx: float = 0.0) -> dict:
    return {
        "ego_dynamic_state": [ego_vx, 0.0, 0.0, 0.0],
        "anns": {
            "gt_names": [actor[0] for actor in actors],
            "gt_boxes": [actor[1] for actor in actors],
            "gt_velocity_3d": [actor[2] for actor in actors],
            "track_tokens": [actor[3] for actor in actors],
        },
    }


def row(
    token: str,
    family: str,
    intent: str,
    log_name: str,
    *,
    fals_positive: bool,
    mixed: bool = False,
    fals: float = 0.0,
) -> dict:
    return {
        "token": token,
        "exclusive_family": family,
        "intent": intent,
        "log_name": log_name,
        "fals_positive": int(fals_positive),
        "strict_clear_mixed": int(mixed),
        "fals": fals,
    }


def test_current_state_risk_requires_one_actor_to_satisfy_visibility_and_distance() -> None:
    mismatched = frame(
        [
            ("vehicle", [-1.0, 0.0], [0.0, 0.0], "near-behind"),
            ("vehicle", [10.0, 0.0], [0.0, 0.0], "far-front"),
        ]
    )

    result = v5.current_state_risk(mismatched)

    assert result["primary_risk"] == 0
    assert result["risk_actor_track_tokens"] == ""


def test_current_state_risk_emits_the_triggering_track_tokens() -> None:
    current = frame(
        [
            ("pedestrian", [8.0, 1.0], [0.0, 0.0], "close-vru"),
            ("vehicle", [20.0, 0.0], [0.0, 0.0], "closing-vehicle"),
        ],
        ego_vx=5.0,
    )

    result = v5.current_state_risk(current)

    assert result["primary_risk"] == 1
    assert result["immediate_actor_track_tokens"] == "close-vru"
    assert result["projected_actor_track_tokens"] == "close-vru|closing-vehicle"
    assert result["risk_actor_track_tokens"] == "close-vru|closing-vehicle"


def test_build_fals_features_uses_raw_pdms_group_mean_and_best() -> None:
    rows = [
        {
            "token": "a",
            "parsed_ok": True,
            "pdms": value,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "time_to_collision_within_bound": 1.0,
        }
        for value in (0.2, 0.4, 0.6, 0.8)
    ]

    feature = v5.build_fals_features(rows, ["a"])["a"]

    assert feature["mean_raw_pdms"] == pytest.approx(0.5)
    assert feature["headroom"] == pytest.approx(0.3)
    assert feature["fals"] == pytest.approx(0.15)
    assert feature["fals_positive"] == 1
    assert feature["strict_clear_count"] == 4


def test_dual_selectors_keep_exact_structure_and_fals_selector_minimizes_anchors(
    monkeypatch,
) -> None:
    monkeypatch.setattr(v5, "TOTAL_SCENES", 4)
    family_quotas = {"risk": 2, "construction": 1, "signal": 1}
    intent_quotas = {"straight": 2, "left": 1, "right": 1}
    rows = [
        row("r1", "risk", "straight", "a", fals_positive=True, mixed=True, fals=0.8),
        row("r2", "risk", "left", "b", fals_positive=True, fals=0.7),
        row("r3", "risk", "right", "c", fals_positive=False),
        row("r4", "risk", "straight", "d", fals_positive=False),
        row("c1", "construction", "right", "e", fals_positive=True, fals=0.6),
        row("c2", "construction", "straight", "f", fals_positive=False),
        row("s1", "signal", "straight", "g", fals_positive=True, fals=0.5),
        row("s2", "signal", "left", "h", fals_positive=True, fals=0.4),
    ]

    baseline = v5.select_risk50(
        rows, family_quotas=family_quotas, intent_quotas=intent_quotas, log_cap=2
    )
    selected, optima = v5.select_risk50_fals(
        rows, family_quotas=family_quotas, intent_quotas=intent_quotas, log_cap=2
    )

    for result in (baseline, selected):
        assert Counter(item["exclusive_family"] for item in result) == family_quotas
        assert Counter(item["intent"] for item in result) == intent_quotas
    assert optima == {
        "max_risk_fals": 2,
        "max_total_fals": 4,
        "max_mixed_after_fals": 1,
    }
    assert all(item["fals_positive"] for item in selected)
