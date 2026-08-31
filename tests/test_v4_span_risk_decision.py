from projects.dataset_v3.span_risk_decision import build_report, derive_tiers


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "token": "token",
        "split": "grpo_screen",
        "horizon_vehicle_distance_m": "10",
        "horizon_vru_distance_m": "10",
        "current_vehicle_front_context": "0",
        "current_vru_front_context": "0",
        "construction_present": "0",
        "current_construction_front_context": "0",
        "current_traffic_control": "0",
        "expert_turn": "0",
        "expert_lateral": "0",
        "expert_braking": "0",
        "expert_stop_to_go": "0",
        "positive_supported": "0",
        "stable_policy_negative": "0",
        "confirmed_paired_recovery": "0",
    }
    row.update(overrides)
    return row


def test_critical_proximity_requires_current_matching_context_for_training() -> None:
    hidden = derive_tiers(_row(horizon_vehicle_distance_m="2.5"))
    visible = derive_tiers(
        _row(horizon_vehicle_distance_m="2.5", current_vehicle_front_context="1")
    )

    assert hidden["eval_tier1"] == 1
    assert hidden["train_tier1"] == 0
    assert visible["train_tier1"] == 1


def test_response_complexity_and_small_capacity_are_reported() -> None:
    train = [
        _row(
            current_traffic_control="1",
            expert_braking="1",
            positive_supported="1",
        )
    ]
    dev = [_row(token="dev", split="dev_tail", horizon_vru_distance_m="4")]

    report = build_report(train, dev)

    assert report["train_tier_counts"]["response_complexity"] == 1
    assert report["dev_tier_counts"]["critical_proximity"] == 1
    assert report["recipe_status"] == "INSUFFICIENT_CONFIRMED_CAPACITY"
    assert report["final_accessed"] is False
