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


def test_g4_smoke_verifier_enforces_group_and_resolved_config(tmp_path):
    verifier = load_module(ROOT / "projects/safe_grpo/verify_g4_smoke.py", "verify_g4_smoke")
    manifest = tmp_path / "train.txt"
    manifest.write_text("a\nb\nc\nd\n", encoding="utf-8")
    rollouts = tmp_path / "rollouts.jsonl"
    rows = []
    for token in ("a", "b", "c", "d"):
        rows.extend(
            {
                "token": token,
                "parsed_ok": True,
                "training_reward": 0.5,
                "pdms": 0.5,
                "pdms_scaled": 0.5,
                "safe": 1.0,
                "no_at_fault_collisions": 1.0,
                "drivable_area_compliance": 1.0,
                "ego_progress": 0.5,
                "time_to_collision_within_bound": 0.5,
                "history_comfort": 0.5,
            }
            for _ in range(4)
        )
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    training_log = tmp_path / "experiment_log.jsonl"
    training_log.write_text('{"step": 1, "actor": {"loss": 0.1}}\n{"step": 2, "actor": {"loss": 0.2}}\n')
    config = tmp_path / "experiment_config.json"
    config.write_text(
        json.dumps(
            {
                "data": {"rollout_batch_size": 2},
                "worker": {
                    "rollout": {"n": 4},
                    "actor": {
                        "global_batch_size": 2,
                        "micro_batch_size_per_device_for_update": 1,
                        "micro_batch_size_per_device_for_experience": 1,
                        "model": {"enable_gradient_checkpointing": True},
                    },
                },
                "trainer": {"max_steps": 2, "skip_final_validation": True},
            }
        ),
        encoding="utf-8",
    )
    gpu_memory = tmp_path / "gpu_memory.csv"
    gpu_memory.write_text(
        "timestamp,memory_used_mib,memory_free_mib,utilization_percent\n1,100,24400,50\n",
        encoding="utf-8",
    )

    report = verifier.verify(rollouts, training_log, config, gpu_memory, manifest, 2, 2, 4)

    assert report["passed"] is True
    assert report["groups"] == 4
    assert report["rollouts"] == 16
    assert report["peak_memory_used_mib"] == 100


def test_t0_launcher_freezes_full_g4_smoke_protocol():
    source = (ROOT / "scripts/run_safe_grpo_experiment.sh").read_text(encoding="utf-8")
    assert "t0_g4_sdr_smoke10_seed20260812" in source
    assert "ROLLOUT_N=4" in source
    assert 'MAX_STEPS=10' in source
    assert 'SKIP_FINAL_VALIDATION=true' in source
    assert 'SAVE_MODEL_ONLY=true' in source
    assert 'worker.rollout.n="$ROLLOUT_N"' in source
    assert '--expected-steps 10' in source
    assert '--groups-per-step 4' in source
    assert '--group-size 4' in source
    assert 'gpu_memory.csv' in source


def test_r4_and_f4_launchers_freeze_formal_g4_protocol_and_checkpoint_evaluations():
    source = (ROOT / "scripts/run_safe_grpo_experiment.sh").read_text(encoding="utf-8")
    assert 'r4)\n        EXP_NAME=r4_sdr_random_lora_1k_g4_seed20260812' in source
    assert 'f4)\n        EXP_NAME=f4_sdr_fals_lora_1k_g4_seed20260812' in source
    assert 'FALS_MANIFEST is required for F4' in source
    assert 'ROLLOUT_N=4' in source
    assert 'SAVE_FREQ=125' in source
    assert 'SAVE_LIMIT=2' in source
    assert 'TRAIN_COVERAGE_ARGS=(--expected-train-rollouts 4)' in source
    assert 'TRAIN_DIAGNOSIS_ARGS=(--expected-rollouts 4)' in source
    assert 'STEP125_CHECKPOINT="$RUN_DIR/checkpoints/global_step_125"' in source
    assert 'STEP250_CHECKPOINT="$RUN_DIR/checkpoints/global_step_250"' in source
    assert 'trainer.load_checkpoint_path="$STEP125_CHECKPOINT"' in source
    assert 'REFERENCE_STEP125_ROLLOUTS="$R4_RUN/step125_dev_rollouts.jsonl"' in source
    assert 'REFERENCE_STEP250_ROLLOUTS="$R4_RUN/dev_rollouts.jsonl"' in source
    assert 'step125_vs_${REFERENCE_LABEL}_paired.json' in source
    assert 'step250_vs_${REFERENCE_LABEL}_paired.json' in source
    assert 'step250_vs_step125_paired.json' in source
    assert 'step250_vs_e2_paired.json' in source


def test_r4_raw_launcher_freezes_sdr_ablation_and_training_evidence():
    source = (ROOT / "scripts/run_safe_grpo_experiment.sh").read_text(encoding="utf-8")
    assert 'r4raw)\n        EXP_NAME=r4_raw_pdms_random_lora_1k_g4_seed20260812' in source
    assert "compute_score_group_raw_pdms" in source
    assert "Missing required R4-SDR reference" in source
    assert '3ae99bb940fad6fab3b488bc4ea7d01e8755a3677161f0c29dffb5e476721fa8  $ITERATION_MANIFEST' in source
    assert 'sha256sum -c "$R4_RUN/model_sha256.txt"' in source
    assert "REFERENCE_LABEL=r4_sdr" in source
    assert 'REFERENCE_STEP125_ROLLOUTS="$R4_RUN/step125_dev_rollouts.jsonl"' in source
    assert 'REFERENCE_STEP250_ROLLOUTS="$R4_RUN/dev_rollouts.jsonl"' in source
    assert 'step125_vs_${REFERENCE_LABEL}_paired.json' in source
    assert 'step250_vs_${REFERENCE_LABEL}_paired.json' in source
    assert "export_training_evidence.py" in source


def test_raw_pdms_reward_is_the_training_scalar_and_raw_response_is_logged(tmp_path, monkeypatch):
    pytest.importorskip("codetiming")
    monkeypatch.setenv("NAVSIM_STAT_PATH", str(ROOT / "stats/trajectory_stats_train.json"))
    monkeypatch.chdir(tmp_path)
    reward = load_module(
        ROOT / "EasyR1/verl/utils/reward_score/navsim/navsim_reward_grouped.py", "raw_pdms_reward"
    )
    monkeypatch.setattr(reward, "get_trajectory_parser", lambda: lambda response: [[0.0, 0.0, 0.0]] * 8)
    monkeypatch.setattr(reward, "denormalize", lambda poses: poses)
    reward._log_path = str(tmp_path / "rollouts.jsonl")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{**metrics(pdms=0.25, pdms_scaled=0.75)}]

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(reward.httpx, "Client", Client)
    result = reward.compute_score_group_raw_pdms(
        [{"ground_truth": {"token": "scene"}, "response": "raw trajectory text", "response_length": 3}]
    )

    assert result[0]["overall"] == pytest.approx(0.25)
    assert result[0]["accuracy"] == pytest.approx(0.75)
    logged = json.loads((tmp_path / "rollouts.jsonl").read_text(encoding="utf-8"))
    assert logged["training_reward"] == pytest.approx(0.25)
    assert logged["response"] == "raw trajectory text"

    monkeypatch.setattr(reward, "get_trajectory_parser", lambda: lambda response: [])
    zero_result = reward.compute_score_group_raw_pdms(
        [{"ground_truth": {"token": "bad"}, "response": "unparsed", "response_length": 1}]
    )
    assert zero_result[0]["overall"] == 0.0
    zero_logged = json.loads((tmp_path / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert zero_logged["no_at_fault_collisions"] == 0.0
    assert zero_logged["drivable_area_compliance"] == 0.0


def test_training_evidence_export_keeps_curves_resources_and_samples(tmp_path):
    exporter = load_module(ROOT / "projects/safe_grpo/export_training_evidence.py", "training_evidence")
    experiment_log = tmp_path / "experiment_log.jsonl"
    rows = []
    for step in (1, 2):
        rows.append(
            {
                "step": step,
                "reward": {"pdms_scaled": 0.4 + step / 10, "safe": 0.5, "parsed_ok": 1.0},
                "actor": {
                    "pg_loss": -0.01 * step,
                    "entropy_loss": 0.2 - step / 100,
                    "kl_loss": 0.001 * step,
                    "ppo_kl": 0.0,
                    "pg_clipfrac_higher": 0.0,
                    "pg_clipfrac_lower": 0.0,
                    "grad_norm": 0.02,
                    "lr": 1e-6,
                },
                "critic": {"advantages": {"mean": 0.0, "min": -1.0, "max": 1.0}},
                "response_length": {"mean": 360.0, "max": 380.0, "clip_ratio": 0.0},
                "timing_s": {"step": 40.0, "gen": 24.0, "reward": 0.01, "ref": 3.0, "update_actor": 9.0},
                "perf": {"throughput": 300.0},
            }
        )
    rows.append({"step": 2, "validation": {"pdms_scaled": 0.7}})
    experiment_log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    gpu = tmp_path / "gpu.csv"
    gpu.write_text(
        "timestamp,memory_used_mib,memory_free_mib,utilization_percent\n1,100,200,50\n2,150,150,90\n",
        encoding="utf-8",
    )
    rollouts = tmp_path / "rollouts.jsonl"
    rollout_rows = [
        {"token": token, "training_reward": reward, "parsed_ok": True, "response_length": 10, "response": token}
        for token, reward in (("a", 0.1), ("a", 0.9), ("b", 0.5), ("b", 0.5))
    ]
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rollout_rows), encoding="utf-8")

    report = exporter.export(experiment_log, gpu, rollouts, tmp_path / "evidence")

    assert report["steps"] == 2
    assert report["gpu"]["peak_memory_used_mib"] == pytest.approx(150)
    assert report["representative_samples"]["raw_response_available"] is True
    for name in (
        "training_history.csv",
        "training_curves.svg",
        "training_curve_summary.json",
        "representative_train_samples.jsonl",
        "training_evidence_manifest.json",
    ):
        assert (tmp_path / "evidence" / name).stat().st_size > 0


def test_s0_geometry_recovers_only_parse_failures_and_detects_partial_collision(tmp_path):
    pytest.importorskip("torch")
    analysis = load_module(
        ROOT / "projects/safe_grpo/analyze_sldr_geometry.py", "analyze_sldr_geometry"
    )
    manifest = tmp_path / "train.txt"
    rollouts = tmp_path / "d0.jsonl"
    output = tmp_path / "s0"
    tokens = ("safe", "unsafe", "partial")
    manifest.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    rows = []
    rows.extend(
        {
            "token": "safe",
            "parsed_ok": True,
            "safe": 1.0,
            "training_reward": value,
            **metrics(pdms_scaled=value, ego_progress=value),
        }
        for value in (0.4, 0.5, 0.6, 0.7)
    )
    rows.extend(
        {
            "token": "unsafe",
            "parsed_ok": True,
            "safe": 0.0,
            "training_reward": 0.0,
            **metrics(
                drivable_area_compliance=0.0,
                pdms=0.0,
                pdms_scaled=0.0,
                time_to_collision_within_bound=value,
            ),
        }
        for value in (0.0, 0.25, 0.5, 1.0)
    )
    rows.extend(
        {
            "token": "partial",
            "parsed_ok": True,
            "safe": 1.0,
            "training_reward": 0.5,
            **metrics(no_at_fault_collisions=0.5, pdms_scaled=0.5),
        }
        for _ in range(3)
    )
    rows.append(
        {
            "token": "partial",
            "parsed_ok": False,
            "safe": 0.0,
            "training_reward": 0.0,
            "pdms": 0.0,
            "pdms_scaled": 0.0,
        }
    )
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = analysis.analyze(
        Namespace(
            d0_rollouts=rollouts,
            train_manifest=manifest,
            output_dir=output,
            bootstrap_samples=100,
            seed=7,
        )
    )

    assert report["field_coverage_and_recovery"]["recovered_parse_failure_rows"] == 1
    assert report["field_coverage_and_recovery"]["recovered_zero_fields"]["ego_progress"] == 1
    assert report["safe_semantics"]["partial_collision_rows"] == 3
    assert report["safe_semantics"]["systematic_mislabel"] is True
    assert report["g2_pair_audit"]["total"] == 18
    assert report["g2_pair_audit"]["pairs_per_g4_group"] == 6
    assert report["g2_pair_audit"]["modes"]["sdr"]["tie_pairs"] > 0
    assert report["g2_pair_audit"]["modes"]["sdr"]["max_error_vs_production_formula"] < 1e-4
    assert report["unsafe_new_preference_bootstrap"]["bootstrap_samples"] == 100
    assert report["gates"]["safe_semantics_valid"] is False
    assert report["decision"] == "close_all_sldr_formal_training"
    assert (output / "s0_report.json").stat().st_size > 0
    assert (output / "group_geometry.csv").stat().st_size > 0


def test_s0_geometry_rejects_missing_metric_on_parsed_rollout(tmp_path):
    pytest.importorskip("torch")
    analysis = load_module(
        ROOT / "projects/safe_grpo/analyze_sldr_geometry.py", "analyze_sldr_geometry_missing"
    )
    manifest = tmp_path / "train.txt"
    rollouts = tmp_path / "d0.jsonl"
    manifest.write_text("a\n", encoding="utf-8")
    row = {"token": "a", "parsed_ok": True, **metrics()}
    del row["ego_progress"]
    rollouts.write_text("".join(json.dumps(row) + "\n" for _ in range(4)), encoding="utf-8")

    with pytest.raises(ValueError, match="Parsed rollout.*missing"):
        analysis.analyze(
            Namespace(
                d0_rollouts=rollouts,
                train_manifest=manifest,
                output_dir=tmp_path / "out",
                bootstrap_samples=10,
                seed=1,
            )
        )


def test_s0_launcher_is_train_only_and_hash_frozen():
    source = (ROOT / "scripts/run_s0_sldr_geometry.sh").read_text(encoding="utf-8")
    assert "d0_stage2_train_n4_seed20260812" in source
    assert "2ededee1d08d754c251a1f1777d2df4e44e52f4a859e884afeed95521e6ef9d6" in source
    assert "4a19947abd86d4265e055a6408fc8a6d579fcc083cb5bc4c207159d5c60d8168" in source
    assert "--bootstrap-samples 20000" in source
    assert "CUDA_VISIBLE_DEVICES=''" in source
    assert "S0_RUN_NAME" in source
    assert "dev_tokens" not in source
    assert "heldout" not in source


def test_a0_audit_blocks_released_filter_that_admits_all_train_tokens(tmp_path):
    audit = load_module(ROOT / "projects/safe_grpo/audit_adas_pool.py", "audit_adas_pool")
    released = tmp_path / "released.txt"
    train = tmp_path / "train.txt"
    dev = tmp_path / "dev.txt"
    heldout = tmp_path / "heldout.txt"
    random = tmp_path / "random.txt"
    output = tmp_path / "report.json"
    train_tokens = [f"t{index}" for index in range(1000)]
    train.write_text("\n".join(train_tokens) + "\n", encoding="utf-8")
    dev.write_text("dev\n", encoding="utf-8")
    heldout.write_text("heldout\n", encoding="utf-8")
    random.write_text("\n".join(train_tokens) + "\n", encoding="utf-8")
    released.write_text("\n".join(train_tokens + ["dev", "heldout"]) + "\n", encoding="utf-8")

    report = audit.audit(
        Namespace(
            released_filter=released,
            train_manifest=train,
            dev_manifest=dev,
            heldout_manifest=heldout,
            random_manifest=random,
            output=output,
        )
    )

    assert report["coverage"]["eligible_train_ratio"] == pytest.approx(1.0)
    assert report["coverage"]["train_tokens_excluded_by_released_gate"] == 0
    assert report["gates"]["eligible_pool_at_least_1000"] is True
    assert report["gates"]["released_gate_is_selective_within_train"] is False
    assert report["g4_bernoulli_boundary"]["minimum_p_to_n_plus_one_minus_p_to_n"] == pytest.approx(0.125)
    assert report["decision"] == "freeze_adas_and_hybrid_routes_as_undefined"
    assert report["manifest_written"] is False
    assert output.stat().st_size > 0


def test_a0_launcher_is_cpu_only_and_does_not_write_a_manifest():
    source = (ROOT / "scripts/run_a0_adas_pool_audit.sh").read_text(encoding="utf-8")
    assert "curious_vla_qwen2_5_vl_3b_sft_stage2_adas1x_6k.txt" in source
    assert "CUDA_VISIBLE_DEVICES=''" in source
    assert "manifest_write=false" in source
    assert "--train-manifest" in source
    assert "--dev-manifest" in source
    assert "--heldout-manifest" in source


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


def test_f0_checkpoint_audit_is_validation_only_and_preregistered():
    source = (ROOT / "scripts/run_f0_checkpoint_audit.sh").read_text(encoding="utf-8")
    assert 'STEP50_CHECKPOINT="$E2_RUN/checkpoints/global_step_50"' in source
    assert 'STEP250_CHECKPOINT="$E2_RUN/checkpoints/global_step_250"' in source
    assert 'data.token_filter_file="$DEV_MANIFEST"' in source
    assert 'data.val_token_filter_file="$DEV_MANIFEST"' in source
    assert "worker.rollout.n=1" in source
    assert "worker.rollout.temperature=0.6" in source
    assert "worker.rollout.top_p=0.95" in source
    assert 'trainer.load_checkpoint_path="$STEP50_CHECKPOINT"' in source
    assert "trainer.max_steps=" not in source
    assert '"pdms_scaled_higher"' in source
    assert '"safe_not_lower"' in source
    assert '"collision_not_lower"' in source
    assert '"ttc_not_lower"' in source
    assert '"heldout_used": False' in source


def test_f1_launcher_is_one_time_and_uses_only_the_frozen_checkpoint():
    source = (ROOT / "scripts/run_f1_heldout_once.sh").read_text(encoding="utf-8")
    assert "EXPECTED_HELDOUT_SHA256=6972791333181f03143f636ab565771c970c01a54b5920df3c8c5645dc2085ef" in source
    assert 'F1_LOCK="$EXPERIMENT_ROOT/F1_HELDOUT_ACCESSED"' in source
    assert '[[ ! -e "$F1_LOCK" ]]' in source
    assert "set -o noclobber" in source
    assert 'FROZEN_CHECKPOINT="$E2_RUN/checkpoints/global_step_250"' in source
    assert 'selection.get("selected_step") != 250' in source
    assert 'data.token_filter_file="$HELDOUT_MANIFEST"' in source
    assert 'data.val_token_filter_file="$HELDOUT_MANIFEST"' in source
    assert "worker.rollout.n=1" in source
    assert "worker.rollout.temperature=0.6" in source
    assert "worker.rollout.top_p=0.95" in source
    assert 'trainer.load_checkpoint_path="$FROZEN_CHECKPOINT"' in source
    assert "trainer.max_steps=" not in source
    assert "--expected-rollouts 1" in source


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
    assert 'data.val_token_filter_file="$RUN_DIR/proxy_candidates.txt"' in source
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
