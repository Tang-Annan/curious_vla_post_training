from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from projects.dataset_v3.s1_pipeline import sha256_file


DISCOVERY_SEED = 20260827
CONFIRMATION_SEEDS = [20260827, 20260828, 20260829]
MATRIX_PRIORITY = ["V3-RR", "V3-TC", "V3-TR", "V3-RC"]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_m0(args: argparse.Namespace) -> None:
    dataset = _read(args.dataset_card)
    d0r2 = _read(args.d0r2_decision)
    selector = _read(args.selector_report)
    reward = _read(args.reward_protocol)
    h0 = _read(args.h0_freeze)

    checks = {
        "dataset_route": dataset["route"] == "REUSE_SFT_CONTROLLED_GRPO_OVERLAP",
        "final_locked": dataset["final_access"] == "LOCKED",
        "strict_unseen": dataset["counts"]["strict_unseen_logs"] == 118
        and dataset["counts"]["strict_unseen_tokens"] == 835,
        "d0r2_frozen": d0r2["status"] == "FROZEN",
        "matrix_priority": d0r2["matrix_priority"] == MATRIX_PRIORITY,
        "discovery_seed": d0r2["seed"] == DISCOVERY_SEED,
        "selector_counts": selector["tokens_per_selector"] == 2000
        and selector["intent_quota"] == {"straight": 1333, "left": 434, "right": 233},
        "selector_per_log_cap": selector["random_per_log"]["max"] <= 8
        and selector["tailmix_per_log"]["max"] <= 8,
        "selector_eval_overlap": selector["dev_overlap"] == 0 and selector["final_overlap"] == 0,
        "selector_monitor_overlap": selector["train_monitor_overlap"] == 0,
        "tailmix_capacity": selector["tailmix_category_counts"]
        == {
            "stable_severe": 578,
            "stable_mixed_recoverable": 68,
            "stable_near_risk": 7,
            "random_anchor": 1347,
        },
        "reward_frozen": reward["status"] == "FROZEN"
        and reward["reward_id"] == "R_TASK_CDT_V3"
        and reward["production_function"] == "compute_score_cdt_task",
        "h0_frozen": h0["status"] == "H0_FROZEN",
        "h0_no_batch8": h0["batch8_trigger"]["required"] is False,
        "formal_budget": h0["group_size"] == 4
        and h0["groups_per_update"] == 4
        and h0["formal_training_groups"] == 2000
        and h0["formal_rollout_queries"] == 8000
        and h0["formal_max_steps"] == 500,
    }
    if not all(checks.values()):
        raise ValueError(f"M0 input freeze mismatch: {checks}")

    protocol = {
        "status": "M0_FROZEN",
        "dataset_route": dataset["route"],
        "tail_evaluation": {
            **d0r2["evaluation"],
            "dev_counts": {"natural": 210, "tail": 206},
            "final_counts": {"natural": 214, "tail": 205},
        },
        "selectors": {
            "tokens_per_cell": selector["tokens_per_selector"],
            "intent_quota": selector["intent_quota"],
            "per_log_cap": 8,
            "tailmix_category_counts": selector["tailmix_category_counts"],
            "tailmix_class_intent_quota": selector["tailmix_class_intent_quota"],
            "distribution_reporting": {
                "intent_js": selector["distribution_js_divergence"]["intent"],
                "month_js": selector["distribution_js_divergence"]["month"],
                "log_name_js": selector["distribution_js_divergence"]["log_name"],
                "gate": "intent quota must match exactly and month JS must be <=0.01; log-name JS is reported but not gated because semantic selection changes log membership; unavailable region/route_type are not imputed",
            },
        },
        "rewards": {
            "raw": reward["raw_control_function"],
            "cdt": reward["production_function"],
            "cdt_id": reward["reward_id"],
            "cdt_formula": reward["formula"],
            "task_quality": reward["task_quality"],
            "invalid_policy": reward["invalid_policy"],
        },
        "optimization": {
            "learning_rate": h0["selected_lr"],
            "advantage_estimator": h0["selected_estimator"],
            "group_size": h0["group_size"],
            "groups_per_update": h0["groups_per_update"],
            "training_groups": h0["formal_training_groups"],
            "rollout_queries": h0["formal_rollout_queries"],
            "max_steps": h0["formal_max_steps"],
            "train_monitor_steps": h0["formal_train_monitor_steps"],
            **h0["fixed_optimization"],
        },
        "resolved_config_gate": {
            "source_clean_and_exact": True,
            "stage2_model_hash_exact": True,
            "selector_manifest_and_parquet_hash_exact": True,
            "reward_entrypoint_exact": True,
            "optimization_fields_exact": True,
            "rollout_n": 4,
            "monitor_n": 1,
            "response_length": 512,
            "parse_rate_min": 0.99,
            "clip_rate_max": 0.01,
            "nonfinite_allowed": 0,
        },
        "evaluation": {
            "natural_primary": "pdms_scaled",
            "tail_primary": "strict_clear_rate",
            "tail_primary_definition": "parsed_ok and canonical CDT tier L3, equivalently Collision=1 and DAC=1 and TTC=1",
            "natural_noninferiority_margin": -0.01,
            "secondary_natural": [
                "pdms",
                "ego_progress",
                "history_comfort",
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "time_to_collision_within_bound",
            ],
            "secondary_tail": [
                "tier_L0_rate",
                "tier_L1_rate",
                "tier_L2_rate",
                "tier_L3_rate",
                "no_at_fault_collisions",
                "time_to_collision_within_bound",
                "pdms_cvar20",
                "paired_tier_transition",
            ],
            "bootstrap": {"resamples": 20000, "cluster": "log_name", "paired": True},
        },
        "matrix": {
            "priority": MATRIX_PRIORITY,
            "discovery_seed": DISCOVERY_SEED,
            "cells": {
                "V3-RR": {"selector": "Random", "reward": "Raw-PDMS"},
                "V3-TR": {"selector": "TailMix", "reward": "Raw-PDMS"},
                "V3-RC": {"selector": "Random", "reward": "R_TASK_CDT_V3"},
                "V3-TC": {"selector": "TailMix", "reward": "R_TASK_CDT_V3"},
            },
            "contrasts": {
                "grpo_baseline": "V3-RR - V3-E0-SFT",
                "complete_endpoint": "V3-TC - V3-RR",
                "selector": "V3-TR - V3-RR",
                "reward": "V3-RC - V3-RR",
                "interaction": "(V3-TC - V3-TR) - (V3-RC - V3-RR)",
            },
        },
        "promotion": {
            "discovery": {
                "tail_primary_delta_min": 0.01,
                "tail_primary_ci_upper_gt": 0.0,
                "natural_primary_point_delta_min": -0.01,
                "natural_primary_ci_lower_severe_harm_floor": -0.03,
                "safety_component_point_drop_max": 0.005,
                "interpretation": "cost-control gate only; single-seed discovery is not a stability claim",
            },
            "confirmation": {
                "seeds": CONFIRMATION_SEEDS,
                "per_seed_tail_delta_positive": "3/3",
                "mean_tail_primary_delta_min": 0.01,
                "two_level_tail_ci_lower_gt": 0.0,
                "two_level_natural_ci_lower_gt": -0.01,
                "safety_component_mean_drop_max": 0.005,
                "simple_contrast": "run only the matched cell pair for the two added seeds",
                "interaction": "run all four cells for all three matched seeds; otherwise no formal interaction claim",
            },
            "no_discovery_pass": "do not add training seeds and do not access Final for an ineligible post-training model",
        },
        "final_access": {
            "locked": True,
            "rule": "one access after Dev-based method/seed freeze; evaluate SFT-E0 and every confirmed eligible candidate in the same access",
        },
        "input_checks": checks,
        "input_sha256": {
            "dataset_card": sha256_file(args.dataset_card),
            "d0r2_decision": sha256_file(args.d0r2_decision),
            "selector_report": sha256_file(args.selector_report),
            "reward_protocol": sha256_file(args.reward_protocol),
            "h0_freeze": sha256_file(args.h0_freeze),
        },
        "dev_accessed": False,
        "final_accessed": False,
    }
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--d0r2-decision", type=Path, required=True)
    parser.add_argument("--selector-report", type=Path, required=True)
    parser.add_argument("--reward-protocol", type=Path, required=True)
    parser.add_argument("--h0-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    freeze_m0(parser.parse_args())


if __name__ == "__main__":
    main()
