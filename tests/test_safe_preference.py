import importlib.util
import json
from pathlib import Path

import numpy as np


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
