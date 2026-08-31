from projects.dataset_v3.tail_semantic_audit import build_report, feature_row, proximity_bin


def _feature(token: str, log_name: str, interaction: bool, category: str) -> dict:
    row = feature_row(
        token=token,
        log_name=log_name,
        intent="straight",
        map_location="sg-one-north",
        vehicle=3.0 if interaction else 30.0,
        vru=None,
    )
    row["category"] = category
    return row


def _outcome(row: dict, *, clear: bool) -> dict:
    return {
        **row,
        "strict_clear": clear,
        "no_at_fault_collisions": 1.0 if clear else 0.5,
        "drivable_area_compliance": 1.0,
        "time_to_collision_within_bound": 1.0,
        "ego_progress": 1.0,
        "history_comfort": 1.0,
        "pdms": 1.0 if clear else 0.5,
        "pdms_scaled": 1.0 if clear else 0.5,
    }


def test_proximity_bin_uses_frozen_actor_specific_thresholds() -> None:
    assert proximity_bin(5.0, None) == "vehicle"
    assert proximity_bin(None, 10.0) == "vru"
    assert proximity_bin(4.0, 9.0) == "vehicle_and_vru"
    assert proximity_bin(6.0, 11.0) == "near_noninteraction"
    assert proximity_bin(30.0, None) == "far"
    assert proximity_bin(None, None) == "no_actor"


def test_report_detects_directional_tail_alignment() -> None:
    train_rows = []
    random_tokens = set()
    tailmix_tokens = set()
    for index in range(4):
        log_name = f"2021.08.0{index + 1}.log"
        random_token = f"r{index}"
        tailmix_token = f"t{index}"
        random_tokens.add(random_token)
        tailmix_tokens.add(tailmix_token)
        train_rows.append(_feature(random_token, log_name, False, "random_anchor"))
        train_rows.append(
            _feature(
                tailmix_token,
                log_name,
                index < 3,
                "stable_severe" if index < 3 else "random_anchor",
            )
        )

    natural = {
        **_feature("n0", "2021.09.01.natural", False, "random_anchor"),
        "split": "dev_natural",
    }
    tail_interaction = {
        **_feature("d0", "2021.09.02.tail", True, "random_anchor"),
        "split": "dev_tail",
    }
    tail_noninteraction = {
        **_feature("d1", "2021.09.03.tail", False, "random_anchor"),
        "split": "dev_tail",
    }
    dev_rows = [natural, tail_interaction, tail_noninteraction]
    model_rows = {
        "SFT": [
            _outcome(natural, clear=True),
            _outcome(tail_interaction, clear=False),
            _outcome(tail_noninteraction, clear=True),
        ]
    }

    report = build_report(
        train_rows,
        dev_rows,
        model_rows,
        random_tokens,
        tailmix_tokens,
        resamples=200,
        seed=7,
    )

    checks = report["train_to_dev_support"]["directional_checks"]
    assert report["semantic_definitions"]["definition_identity"] is False
    assert report["train_selector_comparison"]["tailmix_minus_random_interaction_rate"]["point_delta"] == 0.75
    assert checks["tailmix_gt_interaction_point_enrichment"] is True
    assert checks["stable_policy_risk_gt_interaction_point_enrichment"] is True
    assert checks["tailmix_proximity_distribution_closer_to_dev_tail_than_random"] is True
    assert report["dev_outcome_alignment"]["SFT"]["tail_by_scene_interaction"]["strict_clear_delta"] == -1.0
    assert report["final_accessed"] is False
