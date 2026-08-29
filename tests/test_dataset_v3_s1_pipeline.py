import pytest

from projects.dataset_v3.s1_pipeline import (
    block_features,
    candidate_tier,
    classify_stability,
    group_rows,
    replay_group,
    select_candidate_tokens,
)


def rollout(token: str, parsed_ok: bool = True) -> dict:
    return {
        "token": token,
        "parsed_ok": parsed_ok,
        "poses": [[0.0, 0.0, 0.0]] * 8 if parsed_ok else [],
        "pdms": 0.5 if parsed_ok else 0.0,
        "pdms_scaled": 0.4 if parsed_ok else 0.0,
    }


def test_replay_group_preserves_invalid_and_adds_metrics() -> None:
    rows = [rollout("token"), rollout("token", False), rollout("token"), rollout("token")]

    def post_group(token: str, poses: list) -> list[dict]:
        assert token == "token"
        assert len(poses) == 3
        return [
            {
                "no_at_fault_collisions": 1.0,
                "drivable_area_compliance": 1.0,
                "ego_progress": 0.5,
                "time_to_collision_within_bound": 1.0,
                "history_comfort": 1.0,
                "pdms": 0.5,
                "pdms_scaled": 0.4,
            }
            for _ in poses
        ]

    enriched = replay_group("token", rows, post_group)

    assert [row["metric_replayed"] for row in enriched] == [True, False, True, True]
    assert enriched[0]["no_at_fault_collisions"] == 1.0
    assert enriched[1]["no_at_fault_collisions"] == 0.0


def test_replay_group_rejects_score_drift() -> None:
    rows = [rollout("token") for _ in range(4)]

    with pytest.raises(ValueError, match="PDMS replay mismatch"):
        replay_group(
            "token",
            rows,
            lambda _token, poses: [
                {
                    "no_at_fault_collisions": 1.0,
                    "drivable_area_compliance": 1.0,
                    "ego_progress": 1.0,
                    "time_to_collision_within_bound": 1.0,
                    "history_comfort": 1.0,
                    "pdms": 0.6,
                    "pdms_scaled": 0.4,
                }
                for _ in poses
            ],
        )


def test_candidate_tier_reaudits_discrete_evaluator_values() -> None:
    row = rollout("token")
    row.update(no_at_fault_collisions=1.0, drivable_area_compliance=1.0, time_to_collision_within_bound=1.0)
    assert candidate_tier(row) == "L3"
    row["time_to_collision_within_bound"] = 0.0
    assert candidate_tier(row) == "L2"
    row["no_at_fault_collisions"] = 0.5
    assert candidate_tier(row) == "L1"
    row["no_at_fault_collisions"] = 0.0
    assert candidate_tier(row) == "L0"
    row["parsed_ok"] = False
    assert candidate_tier(row) is None


def test_group_rows_requires_exact_manifest_coverage() -> None:
    rows = [rollout("a") for _ in range(4)]
    assert set(group_rows(rows, ["a"])) == {"a"}
    with pytest.raises(ValueError, match="manifest"):
        group_rows(rows, ["b"])


def test_candidate_selection_includes_risk_and_closes_batch() -> None:
    rows = [
        {
            "token": f"token-{index}",
            "severe_count": "1" if index == 11 else "0",
            "near_risk_count": "0",
            "headroom": str(index / 100),
        }
        for index in range(12)
    ]

    selected, report = select_candidate_tokens(rows, seed=7, high_headroom_fraction=0.10, batch_size=4)

    assert "token-11" in selected
    assert len(selected) == 4
    assert report["batch_closure_additions"] == 2


def test_stability_categories_are_exclusive_by_frozen_priority() -> None:
    severe_mixed = {
        "severe_count": 1,
        "near_risk_count": 0,
        "mixed_recoverable": 1,
    }
    category, raw_flags = classify_stability(severe_mixed, severe_mixed)
    assert category == "stable_severe"
    assert raw_flags == ("stable_severe", "stable_mixed_recoverable")

    near_mixed = {
        "severe_count": 0,
        "near_risk_count": 1,
        "mixed_recoverable": 1,
    }
    category, raw_flags = classify_stability(near_mixed, near_mixed)
    assert category == "stable_mixed_recoverable"
    assert raw_flags == ("stable_mixed_recoverable", "stable_near_risk")


def test_block_features_requires_risk_and_clear_for_recoverable_mix() -> None:
    rows = []
    for collision, ttc in ((1.0, 1.0), (1.0, 0.0), (1.0, 1.0), (1.0, 1.0)):
        row = rollout("token")
        row.update(
            no_at_fault_collisions=collision,
            drivable_area_compliance=1.0,
            time_to_collision_within_bound=ttc,
        )
        rows.append(row)
    features = block_features(rows)
    assert features["near_risk_count"] == 1
    assert features["strict_clear_count"] == 3
    assert features["mixed_recoverable"] == 1
