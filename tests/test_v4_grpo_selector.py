from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from projects.dataset_v3.v4_grpo_selector import (
    HEADROOM_MIN,
    build_features,
    proportional_quotas,
    select_trial,
    selected_table,
    selector_explanation_table,
    sensitivity_summary,
)


def rollout(
    token: str,
    *,
    safe: bool,
    parsed_ok: bool = True,
    ttc_clear: bool = True,
    distance: float = 5.0,
) -> dict:
    return {
        "token": token,
        "parsed_ok": parsed_ok,
        "no_at_fault_collisions": 1.0 if safe else 0.0,
        "drivable_area_compliance": 1.0,
        "candidate_no_at_fault_collisions": 1.0 if safe else 0.0,
        "candidate_drivable_area_compliance": 1.0,
        "candidate_driving_direction_compliance": 1.0,
        "candidate_traffic_light_compliance": 1.0,
        "ego_progress": 1.0,
        "time_to_collision_within_bound": 1.0 if ttc_clear else 0.0,
        "history_comfort": 1.0,
        "time_to_at_fault_collision": None,
        "time_to_ttc_infraction": None if ttc_clear else 1.0,
        "min_distance_to_actors": distance,
        "pdms": 1.0 if safe else 0.0,
        "pdms_scaled": 1.0 if safe else 0.0,
    }


def test_current_reward_features_form_exclusive_buckets_and_recovery_proxy() -> None:
    labels = [
        {
            "token": token,
            "log_name": f"log-{token}",
            "intent": "straight",
            "exclusive_family": "proximity",
        }
        for token in "ABCDEFGH"
    ]
    screen = [
        *[rollout("A", safe=index >= 2) for index in range(4)],
        *[rollout("B", safe=True) for _ in range(4)],
        *[rollout("C", safe=False, distance=distance) for distance in (0.0, 0.2, 0.6, 1.0)],
        *[rollout("D", safe=False, distance=0.0) for _ in range(4)],
        *[rollout("E", safe=True, ttc_clear=index >= 2) for index in range(4)],
        *[rollout("F", safe=True, distance=distance) for distance in (0.0, 0.2, 0.6, 1.0)],
        *[rollout("G", safe=False, distance=distance) for distance in (0.0, 0.1, 0.2, 0.3)],
        *[rollout("H", safe=True, parsed_ok=index > 0) for index in range(4)],
    ]
    confirm = [rollout("D", safe=False, distance=0.0) for _ in range(4)]

    features = {row["token"]: row for row in build_features(labels, screen, confirm)}

    assert {token: row["bucket"] for token, row in features.items()} == {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "A",
        "F": "C",
        "G": "D",
        "H": "A",
    }
    assert features["A"]["screen_headroom"] > HEADROOM_MIN
    assert features["E"]["screen_strict_clear_count"] == 2
    assert features["E"]["screen_reward_hard_safe_count"] == 4
    assert features["E"]["screen_safety_label_disagreements"] == 2
    assert features["D"]["recovery_candidate"] == 1
    assert features["C"]["recovery_candidate"] == 0

    explanation = selector_explanation_table(
        list(features.values()),
        [features["A"], features["C"], features["E"], features["F"], features["H"]],
        [features["B"], features["D"], features["G"]],
        list(features.values()),
        {"A", "B", "C"},
        {"C", "D", "E"},
    )
    candidate = {row["metric"]: row["Candidate 4005"] for row in explanation}
    final = {row["metric"]: row["Final 2K"] for row in explanation}
    assert candidate["A parse-induced"] == 1
    assert candidate["C-safe"] == 1
    assert candidate["C-unsafe"] == 1
    assert final["Overlap with Risk50"] == 3
    assert final["Overlap with Random"] == 3
    failed_table = selector_explanation_table(
        list(features.values()), None, None, None, {"A"}, {"B"}
    )
    assert all(row["Learnable role"] is None for row in failed_table)

    sensitivity_low = sensitivity_summary(list(features.values()), 0.0025)
    sensitivity_frozen = sensitivity_summary(list(features.values()), HEADROOM_MIN)
    assert sensitivity_frozen["c_partition"] == {"C-safe": 1, "C-unsafe": 1}
    assert sensitivity_low["bucket_counts"]["C"] == sensitivity_frozen["bucket_counts"]["C"] + 1


def test_anchor_quotas_and_trial_keep_exact_role_margins(monkeypatch) -> None:
    assert proportional_quotas(600, {"straight": 1333, "left": 434, "right": 233}) == {
        "straight": 400,
        "left": 130,
        "right": 70,
    }
    features = []
    for family in ("proximity", "construction"):
        for intent in ("straight", "left"):
            for index in range(4):
                features.append(
                    {
                        "token": f"{family}-{intent}-{index}",
                        "log_name": f"log-{family}-{intent}-{index}",
                        "intent": intent,
                        "exclusive_family": family,
                        "bucket": "A" if index == 0 else "C",
                        "learnable_eligible": 1,
                        "recovery_candidate": 0,
                        "screen_headroom": 0.2 - index * 0.01,
                    }
                )
    import projects.dataset_v3.v4_grpo_selector as selector

    monkeypatch.setattr(selector, "TOTAL_SCENES", 8)
    monkeypatch.setattr(selector, "FAMILY_QUOTAS", {"proximity": 4, "construction": 4})
    monkeypatch.setattr(selector, "INTENT_QUOTAS", {"straight": 4, "left": 4})
    trial = select_trial(features, 25)

    assert len(trial["anchors"]) == 2
    assert len(trial["learnable"]) == 6
    assert Counter(row["exclusive_family"] for row in trial["selected"]) == {
        "proximity": 4,
        "construction": 4,
    }
    assert Counter(row["intent"] for row in trial["selected"]) == {
        "straight": 4,
        "left": 4,
    }
    assert Counter(row["exclusive_family"] for row in trial["anchors"]) == {
        "proximity": 1,
        "construction": 1,
    }
    assert {row["token"] for row in features if row["bucket"] == "A"} <= {
        row["token"] for row in trial["learnable"]
    }
    assert all(row["bucket"] != "A" for row in trial["anchors"])


def test_trial_fails_when_learnable_role_cannot_hold_every_a(monkeypatch) -> None:
    features = [
        {
            "token": f"token-{index}",
            "log_name": f"log-{index}",
            "intent": "straight",
            "exclusive_family": "proximity",
            "bucket": "A" if index < 3 else "C",
            "learnable_eligible": 1,
            "recovery_candidate": 0,
            "screen_headroom": 0.1,
        }
        for index in range(4)
    ]
    import projects.dataset_v3.v4_grpo_selector as selector

    monkeypatch.setattr(selector, "TOTAL_SCENES", 4)
    monkeypatch.setattr(selector, "FAMILY_QUOTAS", {"proximity": 4})
    monkeypatch.setattr(selector, "INTENT_QUOTAS", {"straight": 4})

    with pytest.raises(ValueError, match="Joint selector MILP"):
        select_trial(features, 50)


def test_selected_table_preserves_manifest_order_and_checks_images(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    tokens = ["a", "b", "c"]
    for token in tokens:
        (data_root / f"{token}.jpg").write_bytes(b"image")
    table = pa.table(
        {
            "problem": [f"p-{token}" for token in tokens],
            "answer": [{"token": token} for token in tokens],
            "images": [[f"{token}.jpg"] for token in tokens],
        }
    )
    parquet = tmp_path / "screen.parquet"
    pq.write_table(table, parquet)

    selected, report = selected_table(parquet, ["c", "a"], data_root)

    assert [answer["token"] for answer in selected.column("answer").to_pylist()] == ["c", "a"]
    assert report["rows"] == 2
    assert report["missing_images"] == 0
