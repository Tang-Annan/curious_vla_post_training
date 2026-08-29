import json
from pathlib import Path
from types import SimpleNamespace

from projects.dataset_v3.h0_pipeline import (
    H0_INTENT_QUOTA,
    choose_estimator,
    choose_lr,
    freeze_h0,
    select_hparam_tokens,
)


def test_h0_subset_has_frozen_intent_quota() -> None:
    tokens = []
    master = {}
    for intent, count in {"straight": 1333, "left": 434, "right": 233}.items():
        for index in range(count):
            token = f"{intent}-{index:04d}"
            tokens.append(token)
            master[token] = {"intent": intent}
    selected = select_hparam_tokens(tokens, master)
    assert len(selected) == 512
    assert {
        intent: sum(master[token]["intent"] == intent for token in selected)
        for intent in H0_INTENT_QUOTA
    } == H0_INTENT_QUOTA


def pilot_report(lr: float, estimator: str, final_gain: float, mean_gain: float) -> dict:
    checkpoints = {}
    gains = [0.0, mean_gain, mean_gain, mean_gain, mean_gain, final_gain]
    for step, gain in zip((0, 26, 51, 77, 102, 128), gains):
        checkpoints[str(step)] = {
            "pdms": 0.5 + gain,
            "parse_rate": 1.0,
            "clip_rate": 0.0,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "time_to_collision_within_bound": 1.0,
        }
    metric = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "cv_abs": 0.0}
    return {
        "lr": lr,
        "estimator": estimator,
        "monitor_checkpoints": checkpoints,
        "training_metrics": {
            "actor/pg_clipfrac_higher": metric,
            "actor/pg_clipfrac_lower": metric,
            "actor/ppo_kl": metric,
            "actor/grad_norm": {**metric, "mean": 0.5, "max": 1.0},
        },
    }


def write_reports(tmp_path: Path, reports: list[dict]) -> list[Path]:
    paths = []
    for index, report in enumerate(reports):
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    return paths


def test_lr_gate_keeps_conservative_lr_below_margin(tmp_path: Path) -> None:
    paths = write_reports(
        tmp_path,
        [pilot_report(1e-6, "grpo", 0.01, 0.005), pilot_report(3e-6, "grpo", 0.014, 0.006)],
    )
    output = tmp_path / "decision.json"
    choose_lr(SimpleNamespace(reports=paths, output=output))
    assert json.loads(output.read_text())["selected_lr"] == 1e-6


def test_lr_gate_selects_higher_lr_only_above_both_margins(tmp_path: Path) -> None:
    paths = write_reports(
        tmp_path,
        [pilot_report(1e-6, "grpo", 0.01, 0.005), pilot_report(3e-6, "grpo", 0.016, 0.008)],
    )
    output = tmp_path / "decision.json"
    choose_lr(SimpleNamespace(reports=paths, output=output))
    assert json.loads(output.read_text())["selected_lr"] == 3e-6


def test_estimator_gate_requires_monitor_and_safety_margins(tmp_path: Path) -> None:
    paths = write_reports(
        tmp_path,
        [pilot_report(1e-6, "grpo", 0.01, 0.005), pilot_report(1e-6, "std_floor_grpo", 0.014, 0.007)],
    )
    output = tmp_path / "decision.json"
    choose_estimator(SimpleNamespace(reports=paths, output=output))
    assert json.loads(output.read_text())["selected_estimator"] == "std_floor_grpo"


def test_h0_freeze_keeps_batch4_when_trigger_is_clear(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"fixed_optimization": {"ppo_epochs": 1}}), encoding="utf-8")
    lr_decision = tmp_path / "lr.json"
    lr_decision.write_text(json.dumps({"status": "LR_FROZEN", "selected_lr": 1e-6}), encoding="utf-8")
    estimator_decision = tmp_path / "estimator.json"
    estimator_decision.write_text(
        json.dumps({"status": "ESTIMATOR_FROZEN", "selected_estimator": "grpo"}), encoding="utf-8"
    )
    report = pilot_report(1e-6, "grpo", 0.01, 0.005)
    report["groups_per_update"] = 4
    report["training_metrics"]["actor/grad_norm"]["cv_abs"] = 0.25
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    experiment_log = tmp_path / "experiment_log.jsonl"
    experiment_log.write_text(
        "".join(json.dumps({"step": step, "actor": {"grad_norm": 0.1}}) + "\n" for step in range(129)),
        encoding="utf-8",
    )
    output = tmp_path / "freeze.json"
    freeze_h0(
        SimpleNamespace(
            protocol=protocol,
            lr_decision=lr_decision,
            estimator_decision=estimator_decision,
            selected_report=report_path,
            selected_experiment_log=experiment_log,
            output=output,
        )
    )
    frozen = json.loads(output.read_text())
    assert frozen["status"] == "H0_FROZEN"
    assert frozen["groups_per_update"] == 4
    assert frozen["formal_max_steps"] == 500
    assert frozen["batch8_trigger"]["required"] is False


def test_h0_freeze_requires_batch8_pilot_when_high_grad_rate_triggers(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"fixed_optimization": {}}), encoding="utf-8")
    lr_decision = tmp_path / "lr.json"
    lr_decision.write_text(json.dumps({"status": "LR_FROZEN", "selected_lr": 1e-6}), encoding="utf-8")
    estimator_decision = tmp_path / "estimator.json"
    estimator_decision.write_text(
        json.dumps({"status": "ESTIMATOR_FROZEN", "selected_estimator": "grpo"}), encoding="utf-8"
    )
    report = pilot_report(1e-6, "grpo", 0.01, 0.005)
    report["groups_per_update"] = 4
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    experiment_log = tmp_path / "experiment_log.jsonl"
    experiment_log.write_text(
        "".join(
            json.dumps({"step": step, "actor": {"grad_norm": 1.0 if 1 <= step <= 7 else 0.1}}) + "\n"
            for step in range(129)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "freeze.json"
    freeze_h0(
        SimpleNamespace(
            protocol=protocol,
            lr_decision=lr_decision,
            estimator_decision=estimator_decision,
            selected_report=report_path,
            selected_experiment_log=experiment_log,
            output=output,
        )
    )
    frozen = json.loads(output.read_text())
    assert frozen["status"] == "BATCH8_PILOT_REQUIRED"
    assert frozen["groups_per_update"] is None


def test_h0_infrastructure_records_step_aware_monitor_evidence() -> None:
    root = Path(__file__).parents[1]
    trainer = (root / "EasyR1/verl/trainer/ray_trainer.py").read_text(encoding="utf-8")
    reward = (root / "EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py").read_text(encoding="utf-8")
    core = (root / "EasyR1/verl/trainer/core_algos.py").read_text(encoding="utf-8")
    assert 'test_batch.non_tensor_batch["evidence_step"]' in trainer
    assert 'save_dict["evidence_step"]' in reward
    assert "STD_FLOOR_GRPO" in core
    assert "torch.clamp(id2std[index[i]], min=std_floor)" in core
