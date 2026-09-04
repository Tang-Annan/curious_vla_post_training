from collections import Counter
from pathlib import Path

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


def test_current_state_risk_receding_actor_does_not_trigger_ttc() -> None:
    result = v5.current_state_risk(
        frame([("vehicle", [10.0, 0.0], [2.0, 0.0], "receding")])
    )

    assert result["projected_conflict"] == 0
    assert result["primary_risk"] == 0


def test_current_state_risk_conflict_after_horizon_does_not_trigger() -> None:
    result = v5.current_state_risk(
        frame([("vehicle", [30.0, 0.0], [0.0, 0.0], "late")], ego_vx=5.0)
    )

    assert result["projected_conflict"] == 0
    assert result["primary_risk"] == 0


def test_current_state_risk_detects_lateral_crossing_by_same_actor() -> None:
    result = v5.current_state_risk(
        frame([("vehicle", [8.0, 5.0], [0.0, -2.0], "crossing")], ego_vx=2.0)
    )

    assert result["projected_conflict"] == 1
    assert result["lateral_convergence"] == 1
    assert result["lateral_actor_track_tokens"] == "crossing"


def test_current_state_risk_rejects_misaligned_annotations() -> None:
    current = frame([("vehicle", [8.0, 0.0], [0.0, 0.0], "actor")])
    current["anns"]["gt_velocity_3d"] = []

    with pytest.raises(ValueError, match="not aligned"):
        v5.current_state_risk(current)


def test_current_state_risk_rejects_duplicate_track_tokens() -> None:
    current = frame(
        [
            ("vehicle", [8.0, 0.0], [0.0, 0.0], "same"),
            ("pedestrian", [6.0, 1.0], [0.0, 0.0], "same"),
        ]
    )

    with pytest.raises(ValueError, match="non-empty and unique"):
        v5.current_state_risk(current)


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
    assert optima["max_risk_fals"] == 2
    assert optima["max_total_fals"] == 4
    assert optima["max_mixed_after_fals"] == 1
    assert all(item["fals_positive"] for item in selected)


def test_risk_fals_priority_is_not_sacrificed_for_more_total_fals(monkeypatch) -> None:
    monkeypatch.setattr(v5, "TOTAL_SCENES", 3)
    rows = [
        row("risk-positive", "risk", "straight", "shared", fals_positive=True, fals=0.1),
        row("risk-anchor", "risk", "left", "risk", fals_positive=False),
        row("construction-positive", "construction", "straight", "construction", fals_positive=True, fals=0.9),
        row("construction-anchor", "construction", "left", "construction-anchor", fals_positive=False),
        row("signal-positive", "signal", "left", "shared", fals_positive=True, fals=0.8),
        row("signal-anchor", "signal", "left", "signal-anchor", fals_positive=False),
    ]

    selected, optima = v5.select_risk50_fals(
        rows,
        family_quotas={"risk": 1, "construction": 1, "signal": 1},
        intent_quotas={"straight": 1, "left": 2},
        log_cap=1,
    )

    assert optima["max_risk_fals"] == 1
    assert optima["max_total_fals"] == 1
    assert "risk-positive" in {item["token"] for item in selected}


def test_strict_clear_mixed_precedes_fals_sum(monkeypatch) -> None:
    monkeypatch.setattr(v5, "TOTAL_SCENES", 1)
    rows = [
        row("mixed", "risk", "straight", "a", fals_positive=True, mixed=True, fals=0.1),
        row("higher-fals", "risk", "straight", "b", fals_positive=True, fals=0.9),
    ]

    selected, optima = v5.select_risk50_fals(
        rows,
        family_quotas={"risk": 1},
        intent_quotas={"straight": 1},
        log_cap=1,
    )

    assert [item["token"] for item in selected] == ["mixed"]
    assert optima["max_mixed_after_fals"] == 1


def test_fals_sum_precedes_hash_even_below_old_combined_objective_scale(monkeypatch) -> None:
    monkeypatch.setattr(v5, "TOTAL_SCENES", 1)
    tokens = [f"candidate-{index}" for index in range(100)]
    low_hash = min(tokens, key=lambda token: v5._stable_cost("v5-risk50-fals", token))
    high_hash = max(tokens, key=lambda token: v5._stable_cost("v5-risk50-fals", token))
    rows = [
        row(low_hash, "risk", "straight", "low", fals_positive=True, fals=0.1),
        row(
            high_hash,
            "risk",
            "straight",
            "high",
            fals_positive=True,
            fals=0.100000000002,
        ),
    ]

    selected, optima = v5.select_risk50_fals(
        rows,
        family_quotas={"risk": 1},
        intent_quotas={"straight": 1},
        log_cap=1,
    )

    assert [item["token"] for item in selected] == [high_hash]
    assert optima["max_fals_quantized"] == v5._fals_quantized(rows[1])
    assert optima["selected_raw_fals_sum"] == pytest.approx(rows[1]["fals"])


def test_log_cap_is_a_binding_constraint(monkeypatch) -> None:
    monkeypatch.setattr(v5, "TOTAL_SCENES", 2)
    rows = [
        row("shared-1", "risk", "straight", "shared", fals_positive=True, fals=0.9),
        row("shared-2", "risk", "straight", "shared", fals_positive=True, fals=0.8),
        row("other", "risk", "straight", "other", fals_positive=False),
    ]

    selected, _ = v5.select_risk50_fals(
        rows,
        family_quotas={"risk": 2},
        intent_quotas={"straight": 2},
        log_cap=1,
    )

    assert Counter(item["log_name"] for item in selected)["shared"] == 1
    assert sum(item["fals_positive"] for item in selected) == 1


def test_unavoidable_anchor_is_reported_by_the_optimum(monkeypatch) -> None:
    monkeypatch.setattr(v5, "TOTAL_SCENES", 2)
    rows = [
        row("positive", "risk", "straight", "a", fals_positive=True, fals=0.5),
        row("anchor", "risk", "straight", "b", fals_positive=False),
    ]

    selected, optima = v5.select_risk50_fals(
        rows,
        family_quotas={"risk": 2},
        intent_quotas={"straight": 2},
        log_cap=1,
    )

    assert optima["max_risk_fals"] == 1
    assert optima["max_total_fals"] == 1
    assert sum(not item["fals_positive"] for item in selected) == 1


def test_exact_row_index_rejects_duplicate_master_and_scene_label_tokens(monkeypatch) -> None:
    monkeypatch.setattr(v5, "EXPECTED_SCREEN", 2)
    duplicate = [{"token": "a"}, {"token": "a"}]

    with pytest.raises(ValueError, match="Duplicate Screen master subset token"):
        v5._index_exact_rows(duplicate, {"a", "b"}, "Screen master subset")
    with pytest.raises(ValueError, match="Duplicate scene labels token"):
        v5._index_exact_rows(duplicate, {"a", "b"}, "scene labels")


def test_raw_log_index_rejects_duplicate_stems() -> None:
    paths = [Path("first/log.pkl"), Path("second/log.pkl")]

    with pytest.raises(ValueError, match="Duplicate raw log stem log"):
        v5._index_raw_logs(paths)
