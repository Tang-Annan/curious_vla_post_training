import importlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "EasyR1"))
sys.path.insert(0, str(ROOT / "projects/safe_grpo"))


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


def test_fals_rejects_rollouts_outside_train_manifest(tmp_path):
    fals = load_module(ROOT / "EasyR1/scripts/adas/build_fals_filter.py", "build_fals_filter_leakage")
    rollout_path = tmp_path / "rollouts.jsonl"
    rows = [
        {"token": token, "pdms_scaled": 0.5}
        for token in ("train", "train", "train", "train", "held_out")
    ]
    rollout_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the training manifest"):
        fals.build_ranking(rollout_path, ["train"], 4)


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


def test_dr_grpo_centers_without_std_normalization():
    torch = pytest.importorskip("torch")
    core = importlib.import_module("verl.trainer.core_algos")
    rewards = torch.tensor([[0.0], [1.0], [0.90], [0.91], [0.5], [0.5]])
    mask = torch.ones_like(rewards)
    index = ["large", "large", "small", "small", "zero", "zero"]
    advantages, returns = core.compute_dr_grpo_outcome_advantage(rewards, mask, index)

    expected = torch.tensor([[-0.5], [0.5], [-0.005], [0.005], [0.0], [0.0]])
    assert torch.allclose(advantages, expected, atol=1e-7)
    assert torch.equal(returns, advantages)


def test_zero_variance_group_filter_keeps_only_informative_groups():
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    numpy = pytest.importorskip("numpy")
    trainer = importlib.import_module("verl.trainer.ray_trainer")
    uids = numpy.array(["zero", "zero", "signal", "signal", "other", "other"], dtype=object)
    scores = [0.5, 0.5, 0.2, 0.8, 0.0, 0.0]

    assert trainer.select_group_filter_indices(uids, scores, "zero_variance", 0.01, 0.99) == [2, 3]


def test_rollout_pairwise_distances():
    pytest.importorskip("numpy")
    analysis = load_module(ROOT / "projects/safe_grpo/analyze_rollouts.py", "analyze_rollouts")
    rows = [
        {"parsed_ok": True, "poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
        {"parsed_ok": True, "poses": [[0.0, 1.0, 0.0], [1.0, 2.0, 0.0]]},
    ]
    assert analysis.pairwise_distance(rows) == pytest.approx(1.5)
    assert analysis.pairwise_distance(rows, -1) == pytest.approx(2.0)


def test_rollout_analysis_enforces_manifest_coverage(tmp_path):
    pytest.importorskip("numpy")
    analysis = load_module(ROOT / "projects/safe_grpo/analyze_rollouts.py", "analyze_rollouts_manifest")
    manifest = tmp_path / "train.txt"
    manifest.write_text("a\nb\n", encoding="utf-8")
    rollouts = tmp_path / "rollouts.jsonl"
    rows = [
        {
            "token": token,
            "pdms_scaled": reward,
            "poses": [[0.0, 0.0, 0.0]],
            "parsed_ok": True,
            "response_length": 512 if token == "a" and reward == 0.4 else 100,
            "pdms": reward + 0.1,
            "safe": float(token == "a"),
            "no_at_fault_collisions": 1.0,
        }
        for token in ("a", "b")
        for reward in (0.4, 0.4, 0.41, 0.42)
    ]
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = analysis.analyze(rollouts, 0.05, manifest, expected_rollouts=4)
    assert report["groups"] == 2
    assert report["rollouts"] == 8
    assert report["exact_zero_std_ratio"] == pytest.approx(0.0)
    assert report["low_nonzero_std_ratio"] == pytest.approx(1.0)
    assert report["reward_mean"] == pytest.approx(0.4075)
    assert report["reward_std"] == pytest.approx(0.0082915619759)
    assert report["headroom_mean"] == pytest.approx(0.0125)
    assert report["pairwise_ade_mean"] == pytest.approx(0.0)
    assert report["pairwise_fde_mean"] == pytest.approx(0.0)
    assert report["safe_rate"] == pytest.approx(0.5)
    assert report["response_length_mean"] == pytest.approx(203.0)
    assert report["clipped_responses"] == 2
    assert report["metric_means"]["pdms_scaled"] == pytest.approx(0.4075)
    assert report["metric_means"]["pdms"] == pytest.approx(0.5075)
    assert report["metric_means"]["no_at_fault_collisions"] == pytest.approx(1.0)


def test_rollout_analysis_rejects_rows_outside_manifest(tmp_path):
    pytest.importorskip("numpy")
    analysis = load_module(ROOT / "projects/safe_grpo/analyze_rollouts.py", "analyze_rollouts_leakage")
    manifest = tmp_path / "train.txt"
    manifest.write_text("a\n", encoding="utf-8")
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        "".join(json.dumps({"token": token, "pdms_scaled": 0.5}) + "\n" for token in ("a", "heldout")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the manifest"):
        analysis.analyze(rollouts, 0.05, manifest)


def test_rollout_analysis_uses_training_reward_and_sample_group_std(tmp_path):
    pytest.importorskip("numpy")
    analysis = load_module(ROOT / "projects/safe_grpo/analyze_rollouts.py", "analyze_training_reward")
    manifest = tmp_path / "train.txt"
    rollouts = tmp_path / "rollouts.jsonl"
    manifest.write_text("a\n", encoding="utf-8")
    rows = [
        {"token": "a", "training_reward": reward, "pdms_scaled": 0.5, "poses": []}
        for reward in (0.0, 0.1)
    ]
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = analysis.analyze(rollouts, 0.08, manifest, expected_rollouts=2)

    assert report["reward_mean"] == pytest.approx(0.05)
    assert report["group_metrics"][0]["reward_std"] == pytest.approx(0.070710678)
    assert report["low_nonzero_std_ratio"] == pytest.approx(1.0)


def test_r0_difficulty_bias_analysis_and_gates(tmp_path):
    analysis = load_module(
        ROOT / "projects/safe_grpo/analyze_difficulty_bias.py", "analyze_difficulty_bias"
    )
    tokens = [f"t{index}" for index in range(10)]
    fals = tokens[:4]
    random = tokens[4:8]
    train_manifest = tmp_path / "train.txt"
    fals_manifest = tmp_path / "fals.txt"
    random_manifest = tmp_path / "random.txt"
    train_manifest.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    fals_manifest.write_text("\n".join(fals) + "\n", encoding="utf-8")
    random_manifest.write_text("\n".join(random) + "\n", encoding="utf-8")

    d0_rows = []
    for index, token in enumerate(tokens):
        rewards = [0.1 * index, 0.1 * index + 0.1, 0.1 * index + 0.2, 0.1 * index + 0.3]
        d0_rows.extend(
            {"token": token, "pdms_scaled": reward, "safe": float(reward > 0.5)} for reward in rewards
        )
    e2_rewards = {
        "t0": [0.0, 1.0],
        "t1": [0.1, 0.9],
        "t2": [0.4, 0.6],
        "t3": [0.5, 0.5],
    }
    e2_rows = [
        {"token": token, "pdms_scaled": reward, "safe": float(reward > 0.5)}
        for token, rewards in e2_rewards.items()
        for reward in rewards
    ]
    d0_rollouts = tmp_path / "d0.jsonl"
    e2_rollouts = tmp_path / "e2.jsonl"
    d0_rollouts.write_text("".join(json.dumps(row) + "\n" for row in d0_rows), encoding="utf-8")
    e2_rollouts.write_text("".join(json.dumps(row) + "\n" for row in e2_rows), encoding="utf-8")
    output_dir = tmp_path / "r0"

    report = analysis.analyze(
        Namespace(
            d0_rollouts=d0_rollouts,
            e2_rollouts=e2_rollouts,
            train_manifest=train_manifest,
            fals_manifest=fals_manifest,
            random_manifest=random_manifest,
            output_dir=output_dir,
            expected_selection_size=4,
            bootstrap_samples=100,
            monte_carlo_trials=10000,
            target_groups=4,
            steps=250,
            max_generation_batches=8,
            seed=20260814,
        )
    )

    assert report["e2"]["informative_group_ratio"] == pytest.approx(0.75)
    assert report["gates"]["r1"]["passed"] is True
    assert report["gates"]["r2"]["passed"] is True
    assert report["next_stage"] == "r1"
    for filename in ("group_metrics.csv", "advantage_scale.csv", "r0_report.json", "difficulty_bias.svg"):
        assert (output_dir / filename).stat().st_size > 0


def test_split_rollouts_enforces_train_and_final_dev_coverage(tmp_path):
    splitter = load_module(ROOT / "projects/safe_grpo/split_rollouts.py", "split_rollouts")
    train_manifest = tmp_path / "train.txt"
    dev_manifest = tmp_path / "dev.txt"
    source = tmp_path / "raw.jsonl"
    train_manifest.write_text("train_a\ntrain_b\n", encoding="utf-8")
    dev_manifest.write_text("dev_a\n", encoding="utf-8")
    rows = [
        {"token": token, "pdms_scaled": 0.5}
        for token in ("train_a", "train_b")
        for _ in range(2)
    ] + [{"token": "dev_a", "pdms_scaled": 0.6}]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    train_rows, dev_rows = splitter.split_rollouts(source, train_manifest, dev_manifest, 2, 1)

    assert len(train_rows) == 4
    assert len(dev_rows) == 1


def test_split_rollouts_rejects_unknown_token(tmp_path):
    splitter = load_module(ROOT / "projects/safe_grpo/split_rollouts.py", "split_rollouts_unknown")
    train_manifest = tmp_path / "train.txt"
    dev_manifest = tmp_path / "dev.txt"
    source = tmp_path / "raw.jsonl"
    train_manifest.write_text("train\n", encoding="utf-8")
    dev_manifest.write_text("dev\n", encoding="utf-8")
    source.write_text(json.dumps({"token": "held_out"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside both manifests"):
        splitter.split_rollouts(source, train_manifest, dev_manifest, 1, 1)


def test_adas_preserves_manifest_and_final_partial_batch():
    source = (ROOT / "EasyR1/verl/trainer/main_adas.py").read_text(encoding="utf-8")
    assert "ppo_config.data.token_filter_file = None" not in source
    assert "train_drop_last=False" in source


def test_e0_and_d0_keep_zero_effect_lora_wrapper():
    source = (ROOT / "scripts/run_safe_grpo_experiment.sh").read_text(encoding="utf-8")
    assert "worker.actor.model.lora.rank=0" not in source


def test_formal_launcher_keeps_e2_e3_e4_as_single_factor_ablations():
    source = (ROOT / "scripts/run_safe_grpo_experiment.sh").read_text(encoding="utf-8")
    assert 'e2)\n        EXP_NAME=e2_fals_lora_1k_seed20260812' in source
    assert 'ITERATION_MANIFEST="$FALS_MANIFEST"' in source
    assert 'e3)\n        EXP_NAME=e3_sldr_lora_1k_seed20260812' in source
    assert "compute_score_sldr" in source
    assert 'e4)\n        EXP_NAME=e4_std_floor_lora_1k_seed20260812' in source
    assert "ratio < 0.10" in source
    assert "ADV_ESTIMATOR=std_floor_grpo" in source
    assert 'algorithm.adv_estimator="$ADV_ESTIMATOR"' in source
    assert 'comm -12 <(sort "$ITERATION_MANIFEST") <(sort "$HELDOUT_MANIFEST")' in source
    assert 'ACTIVE_MANIFEST="$TRAIN_MANIFEST"' in source
    assert "MAX_STEPS=250" in source
    assert 'tracker.get("last_global_step") != expected_step' in source
    assert 'r1)\n        EXP_NAME=r1_fals_dr_grpo_lora_1k_seed20260812' in source
    assert "ADV_ESTIMATOR=dr_grpo" in source
    assert 'json.load(handle)["gates"]["r1"]["passed"]' in source
    assert 'R1_SMOKE_STEPS="${R1_SMOKE_STEPS:-}"' in source
    assert 'trainer.skip_final_validation="$SKIP_FINAL_VALIDATION"' in source
    assert 'tracker_path.parent / f"global_step_{expected_step}" / "actor"' in source
    assert 'r2p)' in source
    assert 'R2_PARENT must be e2 or r1' in source
    assert 'r2g)' in source
    assert 'R2-G requires a passed 20-step R2-P report.' in source
    assert 'EXP_NAME=r2g_e2_dynamic_lora_1k_seed20260812' in source
    assert "FILTER_MODE=zero_variance" in source
    assert "MAX_TRY_MAKE_BATCH=5" in source
    assert '--pilot-log "$RUN_DIR/checkpoints/experiment_log.jsonl"' in source
    assert "--expected-steps 250" in source
    assert "--max-mean-raw-overhead 2.15" in source
    assert "raw_train_query_rollouts.jsonl" in source


def test_split_rollouts_allows_variable_train_queries_but_keeps_dev_exact(tmp_path):
    splitter = load_module(ROOT / "projects/safe_grpo/split_rollouts.py", "split_rollouts_variable")
    train_manifest = tmp_path / "train.txt"
    dev_manifest = tmp_path / "dev.txt"
    source = tmp_path / "raw.jsonl"
    train_manifest.write_text("train_a\ntrain_b\n", encoding="utf-8")
    dev_manifest.write_text("dev\n", encoding="utf-8")
    rows = [
        {"token": "train_a"},
        {"token": "train_a"},
        {"token": "train_b"},
        {"token": "dev"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    train_rows, dev_rows = splitter.split_rollouts(source, train_manifest, dev_manifest, None, 1)

    assert len(train_rows) == 3
    assert len(dev_rows) == 1


def test_split_rollouts_variable_train_queries_still_require_manifest_coverage(tmp_path):
    splitter = load_module(ROOT / "projects/safe_grpo/split_rollouts.py", "split_rollouts_variable_missing")
    train_manifest = tmp_path / "train.txt"
    dev_manifest = tmp_path / "dev.txt"
    source = tmp_path / "raw.jsonl"
    train_manifest.write_text("train_a\ntrain_b\n", encoding="utf-8")
    dev_manifest.write_text("dev\n", encoding="utf-8")
    source.write_text(json.dumps({"token": "train_a"}) + "\n" + json.dumps({"token": "dev"}) + "\n")

    with pytest.raises(ValueError, match="Expected at least one train rollout"):
        splitter.split_rollouts(source, train_manifest, dev_manifest, None, 1)


def test_dynamic_sampling_pilot_analysis_uses_preregistered_cost_gates(tmp_path):
    analysis = load_module(
        ROOT / "projects/safe_grpo/analyze_dynamic_sampling_pilot.py", "analyze_dynamic_sampling_pilot"
    )
    pilot_log = tmp_path / "pilot.jsonl"
    parent_log = tmp_path / "parent.jsonl"
    pilot_rows = []
    parent_rows = []
    for step in range(1, 21):
        pilot_rows.append(
            {
                "step": step,
                "sampling": {
                    "generated_groups": 8,
                    "kept_groups": 5,
                    "used_groups": 4,
                    "dropped_groups": 3,
                    "unused_kept_groups": 1,
                    "generation_batches": 2,
                    "raw_rollout_overhead": 2.0,
                },
                "timing_s": {"step": 60.0},
            }
        )
        parent_rows.append({"step": step, "timing_s": {"step": 40.0}})
    pilot_log.write_text("".join(json.dumps(row) + "\n" for row in pilot_rows), encoding="utf-8")
    parent_log.write_text("".join(json.dumps(row) + "\n" for row in parent_rows), encoding="utf-8")
    with pilot_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 20, "validation": {"reward": 1.0}}) + "\n")
    with parent_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 20, "validation": {"reward": 1.0}}) + "\n")

    report = analysis.analyze(pilot_log, parent_log)

    assert report["sampling"]["mean_raw_rollout_overhead"] == pytest.approx(2.0)
    assert report["timing"]["wall_time_ratio"] == pytest.approx(1.5)
    assert report["gates"]["passed"] is True


def test_recovery_proxy_candidates_preserve_frozen_manifest_order(tmp_path):
    preparation = load_module(
        ROOT / "projects/safe_grpo/prepare_recovery_candidates.py", "prepare_recovery_candidates"
    )
    manifest = tmp_path / "manifest.txt"
    rollouts = tmp_path / "rollouts.jsonl"
    manifest.write_text("a\nb\nc\n", encoding="utf-8")
    rows = []
    for token, safe, pdms in (("a", 0.0, 0.0), ("b", 1.0, 0.5), ("c", 0.0, 0.0)):
        rows.extend({"token": token, "safe": safe, "pdms_scaled": pdms} for _ in range(2))
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    candidates, report = preparation.prepare(rollouts, manifest)

    assert candidates == ["a", "c"]
    assert report["proxy_candidates"] == 2


def test_persistent_failure_gate_uses_four_rollouts_and_full_manifest_lower_bound(tmp_path):
    analysis = load_module(
        ROOT / "projects/safe_grpo/analyze_persistent_failures.py", "analyze_persistent_failures"
    )
    proxy = tmp_path / "proxy.txt"
    full = tmp_path / "full.txt"
    rollouts = tmp_path / "baseline.jsonl"
    proxy.write_text("a\nc\n", encoding="utf-8")
    full.write_text("a\nb\nc\nd\n", encoding="utf-8")
    rows = []
    rows.extend({"token": "a", "safe": 0.0, "pdms_scaled": 0.1} for _ in range(4))
    rows.extend({"token": "c", "safe": 1.0, "pdms_scaled": 0.0} for _ in range(4))
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    persistent, selected, report = analysis.analyze(
        rollouts, proxy, full, minimum_persistent=2, selection_limit=1
    )

    assert persistent == ["a", "c"]
    assert selected == ["a"]
    assert report["persistent_failure_lower_bound_full_manifest"] == pytest.approx(0.5)
    assert report["gate_passed"] is True


def test_r3_baseline_launcher_uses_frozen_e2_checkpoint_and_four_rollouts():
    source = (ROOT / "scripts/run_r3_recovery_baseline.sh").read_text(encoding="utf-8")
    assert "E2_CHECKPOINT=\"$E2_RUN/checkpoints/global_step_250\"" in source
    assert 'trainer.load_checkpoint_path="$E2_CHECKPOINT"' in source
    assert "worker.rollout.n=4" in source
    assert "--expected-candidates 345" in source
    assert "--minimum-persistent 100" in source
    assert "--selection-limit 200" in source
    assert 'comm -12 <(sort "$RUN_DIR/proxy_candidates.txt") <(sort "$HELDOUT_MANIFEST")' in source


def test_paired_rollout_comparison_preserves_token_pairing(tmp_path):
    comparison = load_module(
        ROOT / "projects/safe_grpo/compare_paired_rollouts.py", "compare_paired_rollouts"
    )
    manifest = tmp_path / "dev.txt"
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    manifest.write_text("a\nb\nc\n", encoding="utf-8")
    baseline_rows = []
    candidate_rows = []
    for index, token in enumerate(("a", "b", "c")):
        baseline_row = {"token": token}
        candidate_row = {"token": token}
        for metric in comparison.METRICS:
            baseline_row[metric] = float(index)
            candidate_row[metric] = float(index + 1)
        baseline_rows.append(baseline_row)
        candidate_rows.append(candidate_row)
    baseline.write_text("".join(json.dumps(row) + "\n" for row in baseline_rows), encoding="utf-8")
    candidate.write_text("".join(json.dumps(row) + "\n" for row in reversed(candidate_rows)), encoding="utf-8")

    report = comparison.compare(baseline, candidate, manifest, bootstrap_samples=100, seed=7)

    assert report["tokens"] == 3
    for values in report["metrics"].values():
        assert values["mean_difference"] == pytest.approx(1.0)
        assert values["paired_bootstrap_95_ci"] == pytest.approx([1.0, 1.0])


def test_inference_loader_keeps_final_partial_batch(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    data_loader = importlib.import_module("verl.trainer.data_loader")
    dataset = list(range(5))

    class FakeDataset:
        def __init__(self, **kwargs):
            self.rows = dataset

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return {"value": self.rows[index]}

    monkeypatch.setattr(data_loader, "RLHFDataset", FakeDataset)
    monkeypatch.setattr(data_loader, "collate_fn", lambda rows: rows)
    config = data_loader.DataConfig(
        train_files="train",
        val_files="val",
        rollout_batch_size=4,
        mini_rollout_batch_size=4,
        val_batch_size=4,
        filter_overlong_prompts=False,
    )
    train, _ = data_loader.create_dataloader(config, tokenizer=None, processor=None, train_drop_last=False)
    assert [len(batch) for batch in train] == [4, 1]


def test_data_config_resolves_validation_manifest(tmp_path):
    pytest.importorskip("torch")
    from verl.trainer.config import DataConfig

    manifest = tmp_path / "dev_tokens.txt"
    manifest.write_text("token\n", encoding="utf-8")
    config = DataConfig(val_token_filter_file=str(manifest))
    config.post_init()
    assert config.val_token_filter_file == str(manifest.resolve())
