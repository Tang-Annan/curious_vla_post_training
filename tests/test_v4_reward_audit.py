import math

import pytest

from projects.dataset_v3.v4_reward_audit import (
    audit_report,
    candidate_rewards,
    group_rows,
    load_reward_module,
    replay_group,
    trainer_metrics,
)


REWARD = load_reward_module()


def metrics(
    *,
    collision: float = 1.0,
    drivable: float = 1.0,
    ttc_collision: float = math.inf,
    ttc_infraction: float = math.inf,
    distance: float = 8.0,
    progress: float = 0.5,
    comfort: float = 1.0,
    pdms: float = 0.7,
    pdms_scaled: float = 0.7,
) -> dict[str, float]:
    return {
        "no_at_fault_collisions": collision,
        "drivable_area_compliance": drivable,
        "ego_progress": progress,
        "time_to_collision_within_bound": 1.0 if ttc_infraction == math.inf else 0.0,
        "history_comfort": comfort,
        "time_to_at_fault_collision": ttc_collision,
        "time_to_ttc_infraction": ttc_infraction,
        "min_distance_to_actors": distance,
        "pdms": pdms,
        "pdms_scaled": pdms_scaled,
    }


def test_safety_continuous_reward_hard_gate_ordering() -> None:
    unsafe = metrics(
        collision=0.0,
        ttc_collision=math.inf,
        ttc_infraction=math.inf,
        distance=100.0,
        progress=1.0,
        pdms=0.0,
    )
    safe = metrics(
        ttc_infraction=0.0,
        distance=0.0,
        progress=0.0,
        comfort=0.0,
    )
    assert REWARD.safety_continuous_reward(safe) > REWARD.safety_continuous_reward(unsafe)


def test_safety_continuous_reward_ttc_continuity() -> None:
    r05 = REWARD.safety_continuous_reward(metrics(ttc_collision=0.5))
    r15 = REWARD.safety_continuous_reward(metrics(ttc_collision=1.5))
    r40 = REWARD.safety_continuous_reward(metrics(ttc_collision=4.0))
    rinf = REWARD.safety_continuous_reward(metrics())
    assert r05 < r15 < r40
    assert r40 == rinf


def test_safety_continuous_reward_distance_cap() -> None:
    r25 = REWARD.safety_continuous_reward(metrics(distance=2.5))
    r50 = REWARD.safety_continuous_reward(metrics(distance=5.0))
    r200 = REWARD.safety_continuous_reward(metrics(distance=20.0))
    assert r25 < r50
    assert r50 == r200


def test_safety_continuous_reward_bounds_and_invalid_inputs() -> None:
    worst = REWARD.safety_continuous_reward(
        metrics(
            collision=0.0,
            drivable=0.0,
            ttc_collision=0.0,
            ttc_infraction=0.0,
            distance=0.0,
            progress=0.0,
            comfort=0.0,
            pdms=0.0,
        )
    )
    best = REWARD.safety_continuous_reward(metrics(distance=20.0, progress=1.0))
    assert worst == 0.0
    assert best == 1.0
    with pytest.raises(ValueError):
        REWARD.safety_continuous_reward(metrics(distance=-1.0))
    with pytest.raises(ValueError):
        REWARD.safety_continuous_reward(metrics(collision=0.7))


def test_safety_continuous_reward_none_fields_are_safe() -> None:
    none_metrics = metrics(progress=1.0)
    none_metrics["time_to_at_fault_collision"] = None
    none_metrics["time_to_ttc_infraction"] = None
    none_metrics["min_distance_to_actors"] = None
    assert REWARD.safety_continuous_reward(none_metrics) == 1.0


_FIELD_MAP = {
    "collision": "no_at_fault_collisions",
    "drivable": "drivable_area_compliance",
    "ttc_collision": "time_to_at_fault_collision",
    "ttc_infraction": "time_to_ttc_infraction",
    "distance": "min_distance_to_actors",
    "progress": "ego_progress",
    "comfort": "history_comfort",
}


def _row(token: str, parsed_ok: bool = True, **overrides: float) -> dict[str, object]:
    row: dict[str, object] = {
        "token": token,
        "parsed_ok": parsed_ok,
        "poses": [],
        **metrics(),
    }
    for key, value in overrides.items():
        row[_FIELD_MAP.get(key, key)] = value
    return row


def test_candidate_rewards_zero_invalid_rows_like_trainer() -> None:
    row = _row("t0", parsed_ok=False, pdms=0.9, pdms_scaled=0.9)
    rewards = candidate_rewards(row, REWARD)
    assert rewards["raw_pdms"] == 0.0
    assert rewards["cdt_task"] == 0.0
    assert rewards["safety_continuous"] == 0.0
    assert rewards["hard_safe"] == 0.0


def test_trainer_metrics_treats_none_as_infinity() -> None:
    row = _row("t0")
    row["time_to_at_fault_collision"] = None
    row["time_to_ttc_infraction"] = None
    row["min_distance_to_actors"] = None
    metrics = trainer_metrics(row)
    assert metrics["time_to_at_fault_collision"] == math.inf
    assert metrics["time_to_ttc_infraction"] == math.inf
    assert metrics["min_distance_to_actors"] == math.inf


def test_replay_group_validates_new_fields() -> None:
    rows = [
        _row("t0", parsed_ok=True, pdms=0.7, pdms_scaled=0.7),
        _row("t0", parsed_ok=True, pdms=0.5, pdms_scaled=0.5),
        _row("t0", parsed_ok=False, pdms=0.3, pdms_scaled=0.3),
        _row("t0", parsed_ok=True, pdms=0.2, pdms_scaled=0.2),
    ]
    server_metrics = [
        metrics(pdms=0.7, pdms_scaled=0.7, ttc_collision=0.5, distance=2.0),
        metrics(pdms=0.5, pdms_scaled=0.5, ttc_collision=math.inf, distance=6.0),
        metrics(pdms=0.2, pdms_scaled=0.2, ttc_infraction=1.0, distance=3.0),
    ]

    def post_group(token: str, poses: list[list[list[float]]]) -> list[dict[str, float]]:
        assert token == "t0"
        assert len(poses) == 3
        return server_metrics

    enriched = replay_group("t0", rows, post_group)
    assert enriched[0]["metric_replayed"] is True
    assert enriched[0]["time_to_at_fault_collision"] == 0.5
    assert enriched[1]["time_to_ttc_infraction"] == math.inf
    assert enriched[2]["metric_replayed"] is False
    assert enriched[2]["min_distance_to_actors"] == 0.0
    assert enriched[3]["min_distance_to_actors"] == 3.0


def test_group_rows_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError):
        group_rows([_row("t0"), _row("t0"), _row("t0")], ["t0"])


def _audit_fixture() -> tuple[dict[str, list[dict[str, object]]], list[str], dict[str, dict[str, str]]]:
    tokens: list[str] = []
    labels: dict[str, dict[str, str]] = {}
    groups: dict[str, list[dict[str, object]]] = {}

    def add(token: str, family: str, rows: list[dict[str, object]]) -> None:
        tokens.append(token)
        labels[token] = {
            "token": token,
            "log_name": f"log-{token}",
            "intent": "straight",
            "exclusive_family": family,
        }
        groups[token] = rows

    for index in range(4):
        token = f"p{index}"
        if index == 0:
            rows = [
                _row(token, collision=0.0, ttc_collision=0.5, distance=0.0, pdms=0.0, pdms_scaled=0.0),
                _row(token, collision=0.0, ttc_collision=1.0, distance=0.5, pdms=0.0, pdms_scaled=0.0),
                _row(token, collision=0.0, ttc_collision=1.5, distance=0.8, pdms=0.0, pdms_scaled=0.0),
                _row(token, collision=0.0, ttc_collision=2.0, distance=1.0, pdms=0.0, pdms_scaled=0.0),
            ]
        elif index == 1:
            rows = [
                _row(token, collision=0.0, ttc_collision=0.5, distance=0.0, pdms=0.0, pdms_scaled=0.0),
                _row(token, collision=0.0, ttc_collision=1.0, distance=0.5, pdms=0.0, pdms_scaled=0.0),
                _row(token, distance=6.0, progress=0.2),
                _row(token, distance=12.0, progress=0.9),
            ]
        elif index == 2:
            rows = [_row(token, distance=6.0, progress=0.2), _row(token, distance=12.0, progress=0.9)] + [
                _row(token, distance=6.0, progress=0.5),
                _row(token, distance=12.0, progress=0.5),
            ]
        else:
            rows = [_row(token) for _ in range(4)]
        add(token, "proximity", rows)

    for family in ("construction", "signal"):
        for index in range(2):
            token = f"{family[0]}{index}"
            add(token, family, [_row(token) for _ in range(4)])
    return groups, tokens, labels


def test_audit_report_detects_effective_gain_and_no_inversion() -> None:
    groups, tokens, labels = _audit_fixture()
    table, report = audit_report(groups, tokens, labels, REWARD)
    assert len(table) == len(tokens) * 4
    assert report["hard_safety_inversion"]["checked_groups"] >= 1
    assert report["hard_safety_inversion"]["violations"] == 0
    assert report["effective_gain"]["raw_pdms_equal_groups"] >= 1
    assert report["effective_gain"]["raw_pdms_equal_safety_spread_groups"] >= 1
    assert report["gates"]["differentiation"]["effective_gain_groups_gt_zero"] is True
    assert report["gates"]["inversion"]["violations_zero"] is True
    assert report["gates"]["family"]["proximity_ge_construction"] is True
    assert report["gates"]["family"]["proximity_ge_signal"] is True
    assert report["gates"]["not_gt_imitation"]["by_construction"] is True
    assert report["dev_accessed"] is False
    assert report["final_accessed"] is False
