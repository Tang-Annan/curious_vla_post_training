import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "projects/safe_preference/build_preference_dataset.py"
    spec = importlib.util.spec_from_file_location("build_preference_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_analyzer():
    path = ROOT / "projects/safe_preference/analyze_preference_dataset.py"
    spec = importlib.util.spec_from_file_location("analyze_preference_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_matched_builder():
    path = ROOT / "projects/safe_preference/build_matched_preference_dataset.py"
    spec = importlib.util.spec_from_file_location("build_matched_preference_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_response(value):
    response = json.loads(value)
    points = response["future_trajectory"].split("[PT, ", 1)[1].split("]", 1)[0]
    poses = []
    for point in points.split("), "):
        poses.append([float(item) for item in point.strip("() ").split(", ")])
    return {"poses": poses}


def assistant_response():
    trajectory = ", ".join("(0.0, 0.0, 0.0)" for _ in range(8))
    return json.dumps(
        {
            "critical_objects": {
                "nearby_vehicle": "no",
                "conflicting_pedestrian": "no",
                "cyclist": "no",
                "construction": "no",
                "traffic_element": "no",
                "weather_condition": "no",
                "road_hazard": "no",
                "emergency_vehicle": "no",
                "animal": "no",
                "special_vehicle": "no",
                "conflicting_vehicle": "no",
                "door_opening_vehicle": "no",
            },
            "meta_behaviour": {"speed": "keep", "command": "straight"},
            "explanation": "<thinking>Proceed straight.</thinking>",
            "future_trajectory": f"<answer>[PT, {trajectory}]</answer>",
        }
    )


def records(tmp_path):
    image_root = tmp_path / "datasets"
    image = image_root / "navsim/sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    rl_prompt = "Describe the optimal future 5-second trajectory."
    sft_prompt = "Describe the optimal future 4-second trajectory."
    rollouts = []
    for index in range(4):
        rollouts.append(
            {
                "token": "train",
                "parsed_ok": index != 3,
                "poses": [[float(index), 0.0, 0.0] for _ in range(8)] if index != 3 else [],
            }
        )
    rl_rows = [
        {
            "images": ["navsim/sample.jpg"],
            "problem": rl_prompt,
            "answer": {"token": "train", "gt": []},
        }
    ]
    sft_rows = [
        {
            "id": "train",
            "image": ["navsim/sample.jpg"],
            "system": "You are an expert driver.",
            "conversations": [
                {"from": "human", "value": sft_prompt},
                {"from": "gpt", "value": assistant_response()},
            ],
        }
    ]
    return image_root, rollouts, rl_rows, sft_rows


def run_audit(tmp_path, *, update_sft=None):
    builder = load_builder()
    image_root, rollouts, rl_rows, sft_rows = records(tmp_path)
    if update_sft is not None:
        update_sft(sft_rows)
    report, join_failures, roundtrip_failures = builder.audit_records(
        rollouts=rollouts,
        rl_rows=rl_rows,
        sft_rows=sft_rows,
        means=np.zeros((8, 3)),
        stds=np.ones((8, 3)),
        train_tokens={"train"},
        dev_tokens={"dev"},
        heldout_tokens={"heldout"},
        image_root=image_root,
        parse_response=parse_response,
        expected_rows=4,
        expected_scenes=1,
        expected_rollouts=4,
    )
    return report, join_failures, roundtrip_failures


def test_core_audit_accepts_known_prompt_delta_and_excludes_parse_failures(tmp_path):
    report, join_failures, roundtrip_failures = run_audit(tmp_path)

    assert report["all_core_gates_passed"]
    assert report["d0"]["candidate_rollouts"] == 3
    assert report["d0"]["excluded_parse_or_shape"] == 1
    assert report["sft"]["prompt_exact_matches"] == 0
    assert report["sft"]["prompt_5s_to_4s_only_differences"] == 1
    assert report["roundtrip"]["passed"] == 3
    assert report["roundtrip"]["max_abs_error"] == 0.0
    assert join_failures == []
    assert roundtrip_failures == []


def test_core_audit_blocks_missing_sft_without_placeholder(tmp_path):
    report, join_failures, roundtrip_failures = run_audit(tmp_path, update_sft=lambda rows: rows.clear())

    assert not report["all_core_gates_passed"]
    assert join_failures == [{"token": "train", "reasons": ["missing_sft_row"]}]
    assert len(roundtrip_failures) == 3
    assert {failure["reason"] for failure in roundtrip_failures} == {"missing_template"}


def test_core_audit_blocks_unregistered_prompt_difference(tmp_path):
    def change_prompt(rows):
        rows[0]["conversations"][0]["value"] = "A different prompt."

    report, join_failures, _ = run_audit(tmp_path, update_sft=change_prompt)

    assert not report["gates"]["prompt_relation_known"]
    assert join_failures[0]["reasons"] == ["unexpected_sft_rl_prompt_difference"]


def test_processor_examples_use_rl_prompt_and_replace_only_trajectory(tmp_path):
    builder = load_builder()
    _, rollouts, rl_rows, sft_rows = records(tmp_path)

    examples = builder.build_processor_examples(
        rollouts=rollouts,
        rl_rows=rl_rows,
        sft_rows=sft_rows,
        means=np.zeros((8, 3)),
        stds=np.ones((8, 3)),
        parse_response=parse_response,
        sample_size=1,
        seed=20260812,
    )

    assert examples[0]["conversations"][0]["value"] == rl_rows[0]["problem"]
    chosen = json.loads(examples[0]["chosen"]["value"])
    rejected = json.loads(examples[0]["rejected"]["value"])
    original = json.loads(sft_rows[0]["conversations"][1]["value"])
    for key in ("critical_objects", "meta_behaviour", "explanation"):
        assert chosen[key] == original[key]
        assert rejected[key] == original[key]
    assert chosen["future_trajectory"] != rejected["future_trajectory"]


def test_template_rejects_duplicate_json_keys():
    builder = load_builder()
    duplicate = '{"future_trajectory":"a","future_trajectory":"b"}'

    try:
        builder.parse_unique_json(duplicate)
    except ValueError as exc:
        assert "Duplicate JSON key" in str(exc)
    else:
        raise AssertionError("duplicate JSON keys were accepted")


def metric_row(token, pdms, *, safe=True, valid=True, ttc=None):
    return {
        "token": token,
        "parsed_ok": valid,
        "poses": [[0.0, 0.0, 0.0] for _ in range(8)] if valid else [],
        "pdms_scaled": pdms,
        "no_at_fault_collisions": 1 if safe else 0,
        "drivable_area_compliance": 1,
        "time_to_collision_within_bound": int(safe) if ttc is None else ttc,
    }


def test_pair_capacity_builds_exact_60_40_budget():
    analyzer = load_analyzer()
    rows = []
    for index in range(3):
        rows.extend(
            [metric_row(f"a{index}", 1.0, safe=True), metric_row(f"a{index}", 0.0, safe=False)]
            + [metric_row(f"a{index}", 0.5, safe=True) for _ in range(2)]
        )
    for index in range(2):
        rows.extend(metric_row(f"b{index}", value, safe=True) for value in (0.0, 0.2, 0.8, 1.0))

    report = analyzer.analyze_pair_capacity(
        rows, gap_quantile=0.0, max_pairs=5, min_pairs=5, expected_rollouts=4
    )

    assert report["gate_passed"]
    assert report["N_A"] == 3
    assert report["N_B"] == 2
    assert report["B"] == 5
    assert report["tier_a_required"] == 3
    assert report["tier_b_required"] == 2


def test_pair_capacity_requires_ttc_for_safe_tier():
    analyzer = load_analyzer()
    rows = [metric_row("scene", value, safe=True) for value in (0.0, 0.2, 0.8, 1.0)]
    rows[0]["time_to_collision_within_bound"] = 0

    report = analyzer.analyze_pair_capacity(
        rows, gap_quantile=0.0, max_pairs=5, min_pairs=5, expected_rollouts=4
    )

    assert report["N_A"] == 1
    assert report["N_B"] == 0
    assert report["B"] == 0
    assert report["decision"] == "close_offline_preference_route"


def matched_metric_row(token, pdms, progress, *, safe, pose):
    row = metric_row(token, pdms, safe=safe)
    row.update(
        {
            "poses": [[pose, 0.0, 0.0] for _ in range(8)],
            "ego_progress": progress,
            "history_comfort": 1.0,
            "pdms": pdms,
        }
    )
    return row


def test_matched_selection_freezes_common_chosen_and_distinct_unsafe_rejections():
    builder = load_matched_builder()
    rows = [
        matched_metric_row("scene", 1.0, 1.0, safe=True, pose=0.0),
        matched_metric_row("scene", 0.8, 0.8, safe=True, pose=1.0),
        matched_metric_row("scene", 0.7, 0.9, safe=False, pose=2.0),
        matched_metric_row("scene", 0.1, 0.1, safe=False, pose=3.0),
    ]

    selected, report = builder.select_matched_scenes(rows, expected_rollouts=4, pair_count=1)

    assert report["strict_eligible"] == 1
    assert selected[0]["chosen"]["rollout_index"] == 0
    assert selected[0]["easy_rejected"]["rollout_index"] == 3
    assert selected[0]["hard_rejected"]["rollout_index"] == 2


def test_matched_quality_tie_break_prefers_lower_rollout_index():
    builder = load_matched_builder()
    row = matched_metric_row("scene", 0.5, 0.5, safe=False, pose=0.0)

    assert builder.quality_tuple(row, 0) > builder.quality_tuple(row, 1)


def test_matched_selection_excludes_rejections_that_differ_only_by_index_tie_break():
    builder = load_matched_builder()
    rows = [
        matched_metric_row("eligible", 1.0, 1.0, safe=True, pose=0.0),
        matched_metric_row("eligible", 0.8, 0.8, safe=True, pose=1.0),
        matched_metric_row("eligible", 0.7, 0.9, safe=False, pose=2.0),
        matched_metric_row("eligible", 0.1, 0.1, safe=False, pose=3.0),
        matched_metric_row("tie", 1.0, 1.0, safe=True, pose=4.0),
        matched_metric_row("tie", 0.8, 0.8, safe=True, pose=5.0),
        matched_metric_row("tie", 0.0, 0.0, safe=False, pose=6.0),
        matched_metric_row("tie", 0.0, 0.0, safe=False, pose=7.0),
    ]

    _, report = builder.select_matched_scenes(rows, expected_rollouts=4, pair_count=1)

    assert report["strict_eligible"] == 1
    assert report["excluded"]["rejected_differs_only_by_index_tie_break"] == 1


def test_matched_datasets_are_deterministic_with_same_chosen_and_different_rejected(tmp_path):
    builder = load_matched_builder()
    image_root, _, rl_rows, sft_rows = records(tmp_path)
    rows = [
        matched_metric_row("train", 1.0, 1.0, safe=True, pose=0.0),
        matched_metric_row("train", 0.8, 0.8, safe=True, pose=1.0),
        matched_metric_row("train", 0.7, 0.9, safe=False, pose=2.0),
        matched_metric_row("train", 0.1, 0.1, safe=False, pose=3.0),
    ]
    selected, _ = builder.select_matched_scenes(rows, expected_rollouts=4, pair_count=1)

    first, _ = builder.build_matched_datasets(
        selected=selected,
        rl_rows=rl_rows,
        sft_rows=sft_rows,
        means=np.zeros((8, 3)),
        stds=np.ones((8, 3)),
        parse_response=parse_response,
    )
    second, _ = builder.build_matched_datasets(
        selected=selected,
        rl_rows=rl_rows,
        sft_rows=sft_rows,
        means=np.zeros((8, 3)),
        stds=np.ones((8, 3)),
        parse_response=parse_response,
    )
    audit = builder.audit_datasets(
        datasets=first,
        selected=selected,
        train_tokens={"train"},
        dev_tokens={"dev"},
        heldout_tokens={"heldout"},
        means=np.zeros((8, 3)),
        stds=np.ones((8, 3)),
        parse_response=parse_response,
        pair_count=1,
    )

    assert builder.serialize_json(first) == builder.serialize_json(second)
    assert first["m2"][0]["conversations"][1]["value"] == first["m3"][0]["chosen"]["value"]
    assert first["m3"][0]["chosen"]["value"] == first["m4"][0]["chosen"]["value"]
    assert first["m3"][0]["rejected"]["value"] != first["m4"][0]["rejected"]["value"]
    assert audit["all_core_gates_passed"]
    assert audit["roundtrip_passed"] == 5


def test_matched_audit_blocks_dev_leakage(tmp_path):
    builder = load_matched_builder()
    _, _, rl_rows, sft_rows = records(tmp_path)
    rows = [
        matched_metric_row("train", 1.0, 1.0, safe=True, pose=0.0),
        matched_metric_row("train", 0.8, 0.8, safe=True, pose=1.0),
        matched_metric_row("train", 0.7, 0.9, safe=False, pose=2.0),
        matched_metric_row("train", 0.1, 0.1, safe=False, pose=3.0),
    ]
    selected, _ = builder.select_matched_scenes(rows, expected_rollouts=4, pair_count=1)
    datasets, _ = builder.build_matched_datasets(
        selected=selected,
        rl_rows=rl_rows,
        sft_rows=sft_rows,
        means=np.zeros((8, 3)),
        stds=np.ones((8, 3)),
        parse_response=parse_response,
    )

    audit = builder.audit_datasets(
        datasets=datasets,
        selected=selected,
        train_tokens={"train"},
        dev_tokens={"train"},
        heldout_tokens=set(),
        means=np.zeros((8, 3)),
        stds=np.ones((8, 3)),
        parse_response=parse_response,
        pair_count=1,
    )

    assert not audit["gates"]["zero_dev_overlap"]
    assert not audit["all_core_gates_passed"]


def load_preference_config(name):
    return yaml.safe_load((ROOT / "sft/preference" / name).read_text(encoding="utf-8"))


def test_preference_training_configs_freeze_budget_and_method_variables():
    m2 = load_preference_config("m2_rsft.yaml")
    m3 = load_preference_config("m3_easyneg_dpo.yaml")
    m4 = load_preference_config("m4_hardneg_dpo.yaml")

    for config in (m2, m3, m4):
        assert config["model_name_or_path"].endswith("/models/sft_stage2")
        assert config["image_max_pixels"] == 262144
        assert config["finetuning_type"] == "lora"
        assert config["lora_rank"] == 8
        assert config["lora_alpha"] == 16
        assert config["lora_target"] == "q_proj,k_proj,v_proj,o_proj"
        assert config["freeze_vision_tower"]
        assert config["freeze_multi_modal_projector"]
        assert not config["disable_gradient_checkpointing"]
        assert config["cutoff_len"] == 4096
        assert config["media_dir"].endswith("/curious-vla-workspace/data")
        assert config["per_device_train_batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 16
        assert config["num_train_epochs"] == 3.0
        assert config["seed"] == config["data_seed"] == 20260812
        assert not config["skip_memory_metrics"]
        assert 960 * config["num_train_epochs"] / config["gradient_accumulation_steps"] == 180

    assert m2["stage"] == "sft" and m2["learning_rate"] == 1.0e-5
    for config in (m3, m4):
        assert config["stage"] == "dpo"
        assert config["learning_rate"] == 5.0e-6
        assert config["pref_loss"] == "sigmoid"
        assert config["pref_beta"] == 0.1

    allowed_differences = {"dataset", "output_dir"}
    assert {key for key in m3 if m3[key] != m4[key]} == allowed_differences


def test_preference_smoke_configs_are_exact_20_step_derivatives():
    pairs = (
        ("m2_rsft.yaml", "m2_rsft_smoke.yaml"),
        ("m3_easyneg_dpo.yaml", "m3_easyneg_dpo_smoke.yaml"),
        ("m4_hardneg_dpo.yaml", "m4_hardneg_dpo_smoke.yaml"),
    )
    for formal_name, smoke_name in pairs:
        formal = load_preference_config(formal_name)
        smoke = load_preference_config(smoke_name)
        assert smoke["max_steps"] == 20
        assert smoke["save_steps"] == 20
        assert "max_steps" not in formal
        assert formal["save_steps"] == 180
        differing = {key for key in set(formal) | set(smoke) if formal.get(key) != smoke.get(key)}
        assert differing == {"max_steps", "output_dir", "save_steps"}


def test_preference_resume_check_runs_one_step_without_new_checkpoint():
    source = (ROOT / "scripts/run_safe_preference_experiment.sh").read_text(encoding="utf-8")

    assert "TRAIN_ARGS+=(max_steps=21 'save_strategy=\"no\"')" in source
    assert 'llamafactory-cli" train "${TRAIN_ARGS[@]}"' in source


def test_preference_export_configs_only_change_adapter_and_output():
    names = ("m2_rsft_export.yaml", "m3_easyneg_dpo_export.yaml", "m4_hardneg_dpo_export.yaml")
    configs = [load_preference_config(name) for name in names]
    for config in configs:
        assert config["model_name_or_path"].endswith("/models/sft_stage2")
        assert config["template"] == "qwen2_vl"
        assert config["export_device"] == "cpu"
        assert config["export_size"] == 5
        assert not config["export_legacy_format"]
    allowed = {"adapter_name_or_path", "export_dir"}
    assert {key for key in configs[0] if configs[0][key] != configs[1][key]} == allowed
    assert {key for key in configs[1] if configs[1][key] != configs[2][key]} == allowed


def test_preference_eval_launcher_freezes_one_time_dev_protocol():
    source = (ROOT / "scripts/run_safe_preference_eval.sh").read_text(encoding="utf-8")

    assert "EXPECTED_DEV_SHA256=49dd1fae7f8e77589a27af832835bce8f705c0c5b9062145e180890bf3934cfd" in source
    assert 'worker.rollout.n=1' in source
    assert 'worker.rollout.temperature=0.6' in source
    assert 'worker.rollout.top_p=0.95' in source
    assert 'data.max_response_length=512' in source
    assert 'worker.actor.model.lora.rank=0' in source
    assert '"processor_config.json"' in source
    assert 'prepare_attempt0_exit_code' in source
    assert 'set -o noclobber' in source
    assert 'DEV_LOCK="$EXPERIMENT_ROOT/${METHOD^^}_DEV_ACCESSED"' in source
    assert 'tokens != [str(row["token"]) for row in baseline]' in source


def test_preference_eval_replay_is_explicit_and_keeps_formal_evidence_immutable():
    source = (ROOT / "scripts/run_safe_preference_eval.sh").read_text(encoding="utf-8")

    assert '"$MODE" == prepare-replay || "$MODE" == replay' in source
    assert 'cp -al "$FORMAL_MERGED_DIR" "$MERGED_DIR"' in source
    assert 'cp "$WORKSPACE_ROOT/models/sft_stage2/config.json" "$MERGED_DIR/config.json"' in source
    assert '"input_output_embeddings_tied": tied' in source
    assert 'exploratory_replay' in source
    assert 'M3 replay requires completed M2 replay.' in source
    assert 'if [[ "$REPLAY" == false ]]; then' in source


def test_preference_dataset_registration_matches_training_schema():
    info = json.loads((ROOT / "sft/preference/dataset_info.json").read_text(encoding="utf-8"))

    assert not info["m2_matched_rsft"].get("ranking", False)
    assert info["m3_matched_easyneg_dpo"]["ranking"]
    assert info["m4_matched_hardneg_dpo"]["ranking"]
    assert info["m2_matched_rsft"]["file_name"].endswith("/m2_rsft.json")
    assert info["m3_matched_easyneg_dpo"]["file_name"].endswith("/m3_pdms_easyneg_dpo.json")
    assert info["m4_matched_hardneg_dpo"]["file_name"].endswith("/m4_safety_hardneg_dpo.json")
    for dataset in info.values():
        assert dataset["columns"]["messages"] == "conversations"
        assert dataset["columns"]["images"] == "images"
        assert dataset["columns"]["system"] == "system"
