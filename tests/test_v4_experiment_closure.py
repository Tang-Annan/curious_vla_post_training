from projects.dataset_v3.v4_experiment_closure import (
    build_dev_report,
    build_train_selection,
    current_visible_interaction,
    exclusive_family,
    revised_train_tiers,
)


def test_exclusive_family_uses_frozen_priority() -> None:
    row = {
        "visible_critical_proximity": "1",
        "front_construction_response": "1",
        "current_signal_hard_response": "1",
    }
    assert exclusive_family(row) == "proximity"
    row["visible_critical_proximity"] = "0"
    assert exclusive_family(row) == "construction"
    row["front_construction_response"] = "0"
    assert exclusive_family(row) == "signal"


def test_current_visible_revision_requires_matching_front_actor() -> None:
    row = {
        "token": "token",
        "current_vehicle_distance_m": "4",
        "current_vru_distance_m": "",
        "current_vehicle_front_context": "0",
        "current_vru_front_context": "0",
        "construction_present": "0",
        "current_construction_front_context": "0",
        "current_traffic_control": "0",
        "expert_turn": "0",
        "expert_lateral": "0",
        "expert_braking": "0",
        "expert_stop_to_go": "0",
    }
    assert current_visible_interaction(row) is False
    row["current_vehicle_front_context"] = "1"
    assert current_visible_interaction(row) is True
    assert revised_train_tiers([row])[0]["visible_critical_proximity"] == "1"


def test_selection_satisfies_family_intent_and_log_constraints() -> None:
    families = ("proximity", "construction", "signal")
    intents = ("straight", "left", "right")
    scene_rows = []
    tier_rows = []
    for index, (family, intent) in enumerate(zip(families * 2, intents * 2)):
        token = f"token-{index}"
        scene_rows.append({"token": token, "log_name": f"log-{index}", "intent": intent})
        tier_rows.append(
            {
                "token": token,
                "train_tier1": "1",
                "visible_critical_proximity": str(int(family == "proximity")),
                "front_construction_response": str(int(family == "construction")),
                "current_signal_hard_response": str(int(family == "signal")),
            }
        )

    labels, selected, report = build_train_selection(
        scene_rows,
        tier_rows,
        family_quotas={family: 2 for family in families},
        intent_quotas={intent: 2 for intent in intents},
        cap_candidates=(1,),
        seed=7,
    )

    assert len(labels) == 6
    assert len(selected) == 6
    assert report["selected_max_per_log"] == 1
    assert report["selected_family_counts"] == {family: 2 for family in families}
    assert report["selected_intent_counts"] == {intent: 2 for intent in intents}


def test_dev_gate_passes_only_when_tier1_and_critical_are_harder() -> None:
    tier_rows = [
        {"token": "critical", "eval_tier1": "1", "critical_proximity": "1", "response_complexity": "0"},
        {"token": "response", "eval_tier1": "1", "critical_proximity": "0", "response_complexity": "1"},
        {"token": "control", "eval_tier1": "0", "critical_proximity": "0", "response_complexity": "0"},
    ]
    model_rows = [
        {
            "token": token,
            "model": "model",
            "log_name": f"log-{token}",
            "strict_clear": strict,
            "pdms_scaled": score,
            "no_at_fault_collisions": score,
            "time_to_collision_within_bound": score,
            "interaction_tail_flag": interaction,
        }
        for token, strict, score, interaction in (
            ("critical", "False", "0.2", "1"),
            ("response", "False", "0.4", "0"),
            ("control", "True", "0.9", "0"),
        )
    ]

    _, report = build_dev_report(model_rows, tier_rows, resamples=100, seed=3)

    assert report["status"] == "LABEL_DIFFICULTY_GATE_PASS"
    assert report["training_ready"] is True
