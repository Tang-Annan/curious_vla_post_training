import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "EasyR1"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metrics(**overrides):
    values = {
        "no_at_fault_collisions": 1.0,
        "drivable_area_compliance": 1.0,
        "ego_progress": 0.5,
        "time_to_collision_within_bound": 0.5,
        "history_comfort": 0.5,
        "pdms": 0.5,
        "pdms_scaled": 0.5,
    }
    values.update(overrides)
    return values


def test_sldr_preserves_safety_ordering():
    reward = load_module(
        ROOT / "EasyR1/verl/utils/reward_score/navsim/safety_dense_reward.py", "safety_dense_reward"
    )
    unsafe_best = reward.compute_sldr(
        metrics(no_at_fault_collisions=0.0, ego_progress=1.0, time_to_collision_within_bound=1.0)
    )
    safe_worst = reward.compute_sldr(metrics(ego_progress=0.0, time_to_collision_within_bound=0.0, history_comfort=0.0))
    assert 0.0 <= unsafe_best <= 0.1
    assert 0.5 <= safe_worst <= 1.0
    assert safe_worst > unsafe_best


def test_sldr_rejects_missing_metrics():
    reward = load_module(
        ROOT / "EasyR1/verl/utils/reward_score/navsim/safety_dense_reward.py", "safety_dense_reward_missing"
    )
    with pytest.raises(KeyError):
        reward.compute_sldr({})


def test_fals_ranking_prefers_difficult_scene_with_headroom(tmp_path):
    fals = load_module(ROOT / "EasyR1/scripts/adas/build_fals_filter.py", "build_fals_filter")
    rows = []
    for token, rewards in {
        "impossible": [0.1, 0.1, 0.1, 0.1],
        "easy": [0.9, 0.9, 0.95, 0.95],
        "learnable": [0.3, 0.35, 0.8, 0.85],
    }.items():
        rows.extend({"token": token, "pdms_scaled": value} for value in rewards)
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ranking = fals.build_ranking(rollout_path, ["impossible", "easy", "learnable"], 4)
    assert ranking[0]["token"] == "learnable"
    assert ranking[-1]["token"] == "impossible"


def test_fals_requires_complete_rollout_coverage(tmp_path):
    fals = load_module(ROOT / "EasyR1/scripts/adas/build_fals_filter.py", "build_fals_filter_coverage")
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_text(json.dumps({"token": "a", "pdms_scaled": 0.5}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        fals.build_ranking(rollout_path, ["a"], 4)


def test_std_floor_grpo_matches_grpo_above_floor_and_damps_small_std():
    torch = pytest.importorskip("torch")
    core = importlib.import_module("verl.trainer.core_algos")
    mask = torch.ones((4, 1))
    index = ["a", "a", "b", "b"]
    rewards = torch.tensor([[0.0], [1.0], [0.900], [0.901]])
    vanilla, _ = core.compute_grpo_outcome_advantage(rewards, mask, index)
    floored, _ = core.compute_std_floor_grpo_outcome_advantage(rewards, mask, index, std_floor=0.05)
    assert torch.allclose(vanilla[:2], floored[:2])
    assert torch.all(torch.abs(floored[2:]) < torch.abs(vanilla[2:]))


def test_std_floor_zero_variance_is_zero():
    torch = pytest.importorskip("torch")
    core = importlib.import_module("verl.trainer.core_algos")
    advantages, _ = core.compute_std_floor_grpo_outcome_advantage(
        torch.tensor([[0.5], [0.5]]), torch.ones((2, 1)), ["a", "a"], std_floor=0.05
    )
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_rollout_pairwise_distances():
    pytest.importorskip("numpy")
    analysis = load_module(ROOT / "projects/safe_grpo/analyze_rollouts.py", "analyze_rollouts")
    rows = [
        {"parsed_ok": True, "poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
        {"parsed_ok": True, "poses": [[0.0, 1.0, 0.0], [1.0, 2.0, 0.0]]},
    ]
    assert analysis.pairwise_distance(rows) == pytest.approx(1.5)
    assert analysis.pairwise_distance(rows, -1) == pytest.approx(2.0)


def test_data_config_resolves_validation_manifest(tmp_path):
    pytest.importorskip("torch")
    from verl.trainer.config import DataConfig

    manifest = tmp_path / "dev_tokens.txt"
    manifest.write_text("token\n", encoding="utf-8")
    config = DataConfig(val_token_filter_file=str(manifest))
    config.post_init()
    assert config.val_token_filter_file == str(manifest.resolve())
