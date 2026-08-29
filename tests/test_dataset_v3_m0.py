import json
from pathlib import Path
from types import SimpleNamespace

from projects.dataset_v3.m0_pipeline import freeze_m0


def _write(tmp_path: Path, name: str, value: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_m0_freezes_complete_protocol_without_eval_access(tmp_path: Path) -> None:
    dataset = _write(
        tmp_path,
        "dataset.json",
        {
            "route": "REUSE_SFT_CONTROLLED_GRPO_OVERLAP",
            "final_access": "LOCKED",
            "counts": {"strict_unseen_logs": 118, "strict_unseen_tokens": 835},
        },
    )
    d0r2 = _write(
        tmp_path,
        "d0r2.json",
        {
            "status": "FROZEN",
            "seed": 20260827,
            "matrix_priority": ["V3-RR", "V3-TC", "V3-TR", "V3-RC"],
            "evaluation": {"tail_route": "POLICY_INDEPENDENT_GT_ACTOR_PROXIMITY"},
        },
    )
    selector = _write(
        tmp_path,
        "selector.json",
        {
            "tokens_per_selector": 2000,
            "intent_quota": {"straight": 1333, "left": 434, "right": 233},
            "random_per_log": {"max": 6},
            "tailmix_per_log": {"max": 7},
            "dev_overlap": 0,
            "final_overlap": 0,
            "train_monitor_overlap": 0,
            "tailmix_category_counts": {
                "stable_severe": 578,
                "stable_mixed_recoverable": 68,
                "stable_near_risk": 7,
                "random_anchor": 1347,
            },
            "tailmix_class_intent_quota": {},
            "distribution_js_divergence": {"intent": 0.0, "month": 0.003, "log_name": 0.2},
        },
    )
    reward = _write(
        tmp_path,
        "reward.json",
        {
            "status": "FROZEN",
            "reward_id": "R_TASK_CDT_V3",
            "production_function": "compute_score_cdt_task",
            "raw_control_function": "compute_score_raw_pdms",
            "formula": "(2*tier_value + Q_task)/7",
            "task_quality": "Q_task=(5*ego_progress+2*history_comfort)/7",
            "invalid_policy": "technical zero",
        },
    )
    h0 = _write(
        tmp_path,
        "h0.json",
        {
            "status": "H0_FROZEN",
            "batch8_trigger": {"required": False},
            "selected_lr": 1e-6,
            "selected_estimator": "grpo",
            "group_size": 4,
            "groups_per_update": 4,
            "formal_training_groups": 2000,
            "formal_rollout_queries": 8000,
            "formal_max_steps": 500,
            "formal_train_monitor_steps": [0, 100, 200, 300, 400, 500],
            "fixed_optimization": {"ppo_epochs": 1, "lora_rank": 8, "kl_coefficient": 0.01},
        },
    )
    output = tmp_path / "m0.json"
    freeze_m0(
        SimpleNamespace(
            dataset_card=dataset,
            d0r2_decision=d0r2,
            selector_report=selector,
            reward_protocol=reward,
            h0_freeze=h0,
            output=output,
        )
    )
    protocol = json.loads(output.read_text())
    assert protocol["status"] == "M0_FROZEN"
    assert protocol["evaluation"]["natural_primary"] == "pdms_scaled"
    assert protocol["evaluation"]["tail_primary"] == "strict_clear_rate"
    assert protocol["promotion"]["confirmation"]["seeds"] == [20260827, 20260828, 20260829]
    assert protocol["dev_accessed"] is False
    assert protocol["final_accessed"] is False
