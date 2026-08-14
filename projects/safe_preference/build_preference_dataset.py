"""Audit D0 before constructing offline trajectory-preference pairs."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq


CRITICAL_OBJECTS = {
    "nearby_vehicle",
    "conflicting_pedestrian",
    "cyclist",
    "construction",
    "traffic_element",
    "weather_condition",
    "road_hazard",
    "emergency_vehicle",
    "animal",
    "special_vehicle",
    "conflicting_vehicle",
    "door_opening_vehicle",
}
SPEED_LABELS = {"keep", "accelerate", "decelerate", "other"}
COMMAND_LABELS = {
    "straight",
    "yield",
    "left_turn",
    "right_turn",
    "lane_follow",
    "lane_change_left",
    "lane_change_right",
    "reverse",
    "other",
}


def read_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Manifest contains duplicate tokens: {path}")
    return tokens


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rl_rows(path: Path) -> list[dict]:
    return pq.read_table(path, columns=["images", "problem", "answer"]).to_pylist()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_unique_json(value: str) -> dict:
    parsed = json.loads(value, object_pairs_hook=_unique_object)
    if not isinstance(parsed, dict):
        raise ValueError("Assistant response is not a JSON object")
    return parsed


def _poses_from_parser_output(output: object) -> np.ndarray:
    poses = output.get("poses", []) if isinstance(output, dict) else output
    return np.asarray(poses, dtype=float)


def validate_template(response: object, parse_response: Callable[[str], object]) -> dict:
    if not isinstance(response, str):
        raise ValueError("Assistant response is not a string")
    template = parse_unique_json(response)
    if set(template) != {"critical_objects", "meta_behaviour", "explanation", "future_trajectory"}:
        raise ValueError("Assistant response has an unexpected top-level schema")

    critical = template["critical_objects"]
    if not isinstance(critical, dict) or set(critical) != CRITICAL_OBJECTS:
        raise ValueError("critical_objects has an unexpected schema")
    if any(value not in {"yes", "no"} for value in critical.values()):
        raise ValueError("critical_objects contains an invalid label")

    meta = template["meta_behaviour"]
    if not isinstance(meta, dict) or set(meta) != {"speed", "command"}:
        raise ValueError("meta_behaviour has an unexpected schema")
    if meta["speed"] not in SPEED_LABELS or meta["command"] not in COMMAND_LABELS:
        raise ValueError("meta_behaviour contains an invalid label")
    if not isinstance(template["explanation"], str) or not template["explanation"].strip():
        raise ValueError("explanation is empty or not a string")
    if not isinstance(template["future_trajectory"], str):
        raise ValueError("future_trajectory is not a string")
    if _poses_from_parser_output(parse_response(response)).shape != (8, 3):
        raise ValueError("Official future_trajectory does not parse to 8x3")
    return template


def format_response(template: dict, normalized_poses: np.ndarray) -> str:
    response = dict(template)
    points = ", ".join(
        "(" + ", ".join(f"{value:.6f}" for value in point) + ")"
        for point in normalized_poses
    )
    response["future_trajectory"] = f"<answer>[PT, {points}]</answer>"
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def audit_records(
    *,
    rollouts: list[dict],
    rl_rows: list[dict],
    sft_rows: list[dict],
    means: np.ndarray,
    stds: np.ndarray,
    train_tokens: set[str],
    dev_tokens: set[str],
    heldout_tokens: set[str],
    image_root: Path,
    parse_response: Callable[[str], object],
    expected_rows: int,
    expected_scenes: int,
    expected_rollouts: int,
) -> tuple[dict, list[dict], list[dict]]:
    d0_tokens = [row.get("token") for row in rollouts]
    d0_counts = collections.Counter(d0_tokens)
    d0_unique = set(d0_tokens)

    rl_counts = collections.Counter(row.get("answer", {}).get("token") for row in rl_rows)
    rl_by_token = {
        row["answer"]["token"]: row
        for row in rl_rows
        if row.get("answer", {}).get("token") in d0_unique
    }
    sft_counts = collections.Counter(row.get("id") for row in sft_rows)
    sft_by_token = {row["id"]: row for row in sft_rows if row.get("id") in d0_unique}

    join_failures = []
    templates = {}
    image_matches = 0
    existing_images = 0
    prompt_exact = 0
    prompt_horizon_delta = 0
    valid_templates = 0

    for token in sorted(d0_unique):
        reasons = []
        rl_row = rl_by_token.get(token)
        sft_row = sft_by_token.get(token)
        if rl_row is None:
            reasons.append("missing_rl_row")
        if sft_row is None:
            reasons.append("missing_sft_row")
        if reasons:
            join_failures.append({"token": token, "reasons": reasons})
            continue

        conversations = sft_row.get("conversations")
        if (
            not isinstance(conversations, list)
            or len(conversations) != 2
            or conversations[0].get("from") != "human"
            or conversations[1].get("from") != "gpt"
        ):
            reasons.append("invalid_sft_conversation")
        else:
            sft_prompt = conversations[0].get("value")
            rl_prompt = rl_row.get("problem")
            if sft_prompt == rl_prompt:
                prompt_exact += 1
            elif (
                isinstance(rl_prompt, str)
                and rl_prompt.replace(
                    "optimal future 5-second trajectory", "optimal future 4-second trajectory"
                )
                == sft_prompt
            ):
                prompt_horizon_delta += 1
            else:
                reasons.append("unexpected_sft_rl_prompt_difference")
            try:
                templates[token] = validate_template(conversations[1].get("value"), parse_response)
                valid_templates += 1
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                reasons.append(f"invalid_assistant_template:{exc}")

        images = sft_row.get("image")
        if images == rl_row.get("images"):
            image_matches += 1
        else:
            reasons.append("sft_rl_image_mismatch")
        if isinstance(images, list) and images and all((image_root / path).is_file() for path in images):
            existing_images += 1
        else:
            reasons.append("missing_image")
        if sft_row.get("system") != "You are an expert driver.":
            reasons.append("unexpected_system_prompt")
        if reasons:
            join_failures.append({"token": token, "reasons": reasons})

    candidate_count = 0
    roundtrip_ok = 0
    max_abs_error = 0.0
    roundtrip_failures = []
    rollout_indices = collections.Counter()
    for line_number, row in enumerate(rollouts, start=1):
        token = row.get("token")
        rollout_index = rollout_indices[token]
        rollout_indices[token] += 1
        poses = np.asarray(row.get("poses", []), dtype=float)
        if not row.get("parsed_ok") or poses.shape != (8, 3):
            continue
        candidate_count += 1
        template = templates.get(token)
        if template is None:
            roundtrip_failures.append(
                {"line": line_number, "token": token, "rollout_index": rollout_index, "reason": "missing_template"}
            )
            continue
        normalized = (poses - means) / stds
        response = format_response(template, normalized)
        try:
            parse_unique_json(response)
            parsed = _poses_from_parser_output(parse_response(response))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            roundtrip_failures.append(
                {
                    "line": line_number,
                    "token": token,
                    "rollout_index": rollout_index,
                    "reason": f"parse_failure:{exc}",
                }
            )
            continue
        if parsed.shape != (8, 3):
            roundtrip_failures.append(
                {"line": line_number, "token": token, "rollout_index": rollout_index, "reason": "parsed_shape_not_8x3"}
            )
            continue
        error = float(np.max(np.abs(parsed * stds + means - poses)))
        max_abs_error = max(max_abs_error, error)
        if error > 1e-4:
            roundtrip_failures.append(
                {
                    "line": line_number,
                    "token": token,
                    "rollout_index": rollout_index,
                    "reason": "roundtrip_error",
                    "max_abs_error": error,
                }
            )
            continue
        roundtrip_ok += 1

    d0_shape_ok = (
        len(rollouts) == expected_rows
        and len(d0_unique) == expected_scenes
        and set(d0_counts.values()) == {expected_rollouts}
    )
    gates = {
        "d0_shape": d0_shape_ok,
        "train_manifest_exact_match": d0_unique == train_tokens,
        "zero_dev_overlap": not (d0_unique & dev_tokens),
        "zero_heldout_overlap": not (d0_unique & heldout_tokens),
        "rl_unique_and_complete": len(rl_by_token) == expected_scenes
        and not any(count != 1 for count in rl_counts.values()),
        "sft_unique_and_complete": len(sft_by_token) == expected_scenes
        and not any(count != 1 for count in sft_counts.values()),
        "prompt_relation_known": prompt_exact + prompt_horizon_delta == expected_scenes,
        "images_match_and_exist": image_matches == expected_scenes and existing_images == expected_scenes,
        "assistant_templates_valid": valid_templates == expected_scenes,
        "candidate_roundtrip": roundtrip_ok == candidate_count and not roundtrip_failures,
        "zero_join_failures": not join_failures,
    }
    report = {
        "d0": {
            "rows": len(rollouts),
            "unique_tokens": len(d0_unique),
            "rollouts_per_token_histogram": dict(collections.Counter(d0_counts.values())),
            "parsed_ok": sum(bool(row.get("parsed_ok")) for row in rollouts),
            "pose_shape_8x3": sum(np.asarray(row.get("poses", [])).shape == (8, 3) for row in rollouts),
            "candidate_rollouts": candidate_count,
            "excluded_parse_or_shape": len(rollouts) - candidate_count,
            "outside_train": len(d0_unique - train_tokens),
            "dev_overlap": len(d0_unique & dev_tokens),
            "heldout_overlap": len(d0_unique & heldout_tokens),
        },
        "rl": {
            "rows": len(rl_rows),
            "unique_tokens": len(rl_counts),
            "duplicate_tokens": sum(count != 1 for count in rl_counts.values()),
            "d0_joined_tokens": len(rl_by_token),
        },
        "sft": {
            "rows": len(sft_rows),
            "unique_tokens": len(sft_counts),
            "duplicate_tokens": sum(count != 1 for count in sft_counts.values()),
            "d0_joined_tokens": len(sft_by_token),
            "valid_assistant_templates": valid_templates,
            "prompt_exact_matches": prompt_exact,
            "prompt_5s_to_4s_only_differences": prompt_horizon_delta,
            "image_matches": image_matches,
            "existing_images": existing_images,
        },
        "roundtrip": {
            "passed": roundtrip_ok,
            "failed": len(roundtrip_failures),
            "max_abs_error": max_abs_error,
            "threshold": 1e-4,
            "serialized_decimals": 6,
        },
        "gates": gates,
        "all_core_gates_passed": all(gates.values()),
    }
    return report, join_failures, roundtrip_failures


def build_processor_examples(
    *,
    rollouts: list[dict],
    rl_rows: list[dict],
    sft_rows: list[dict],
    means: np.ndarray,
    stds: np.ndarray,
    parse_response: Callable[[str], object],
    sample_size: int,
    seed: int,
) -> list[dict]:
    rl_by_token = {row["answer"]["token"]: row for row in rl_rows}
    sft_by_token = {row["id"]: row for row in sft_rows}
    rollouts_by_token = collections.defaultdict(list)
    for row in rollouts:
        poses = np.asarray(row.get("poses", []), dtype=float)
        if row.get("parsed_ok") and poses.shape == (8, 3):
            rollouts_by_token[row["token"]].append(poses)

    eligible = [token for token, rows in rollouts_by_token.items() if len(rows) >= 2]
    eligible.sort(key=lambda token: hashlib.sha256(f"{seed}:{token}".encode()).hexdigest())
    if len(eligible) < sample_size:
        raise ValueError(f"Only {len(eligible)} tokens have two processor-audit responses")

    examples = []
    for token in eligible[:sample_size]:
        rl_row = rl_by_token[token]
        sft_row = sft_by_token[token]
        template = validate_template(sft_row["conversations"][1]["value"], parse_response)
        chosen = format_response(template, (rollouts_by_token[token][0] - means) / stds)
        rejected = format_response(template, (rollouts_by_token[token][1] - means) / stds)
        examples.append(
            {
                "token": token,
                "conversations": [{"from": "human", "value": rl_row["problem"]}],
                "chosen": {"from": "gpt", "value": chosen},
                "rejected": {"from": "gpt", "value": rejected},
                "images": rl_row["images"],
                "system": sft_row["system"],
            }
        )
    return examples


def run_llamafactory_processor_audit(
    *,
    examples: list[dict],
    llamafactory_root: Path,
    model_path: Path,
    image_root: Path,
    response_limit: int,
) -> dict:
    sys.path.insert(0, str(llamafactory_root / "src"))
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor.pairwise import PairwiseDatasetProcessor
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    source_commit = subprocess.run(
        ["git", "-C", str(llamafactory_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_status = subprocess.run(
        ["git", "-C", str(llamafactory_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_status:
        raise ValueError("LLaMA-Factory source checkout is not clean")

    model_args = ModelArguments(model_name_or_path=str(model_path))
    tokenizer_module = load_tokenizer(model_args)
    data_args = DataArguments(template="qwen2_vl", cutoff_len=65536, media_dir=str(image_root))
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    dataset_attr = DatasetAttr("file", "p1_processor_audit")
    dataset_attr.join(
        {
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "images": "images",
                "system": "system",
            },
        }
    )
    converter = SharegptDatasetConverter(dataset_attr, data_args)
    aligned = [converter(example) for example in examples]
    batch = {key: [example[key] for example in aligned] for key in aligned[0]}
    dataset_processor = PairwiseDatasetProcessor(
        template=template,
        tokenizer=tokenizer_module["tokenizer"],
        processor=tokenizer_module["processor"],
        data_args=data_args,
    )
    processed = dataset_processor.preprocess_dataset(batch)
    chosen_response_lengths = [
        sum(token != IGNORE_INDEX for token in labels) for labels in processed["chosen_labels"]
    ]
    rejected_response_lengths = [
        sum(token != IGNORE_INDEX for token in labels) for labels in processed["rejected_labels"]
    ]
    chosen_total_lengths = [len(tokens) for tokens in processed["chosen_input_ids"]]
    rejected_total_lengths = [len(tokens) for tokens in processed["rejected_input_ids"]]
    processor_files = [
        model_path / name
        for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json")
    ]
    passed = (
        tokenizer_module["processor"] is not None
        and len(chosen_response_lengths) == len(examples)
        and max(chosen_response_lengths + rejected_response_lengths) <= response_limit
        and len(processed["images"]) == len(examples)
    )
    return {
        "passed": passed,
        "sample_size": len(examples),
        "sample_tokens": [example["token"] for example in examples],
        "template": "qwen2_vl",
        "processor": type(tokenizer_module["processor"]).__name__,
        "response_token_limit": response_limit,
        "chosen_response_tokens": {
            "min": min(chosen_response_lengths),
            "max": max(chosen_response_lengths),
        },
        "rejected_response_tokens": {
            "min": min(rejected_response_lengths),
            "max": max(rejected_response_lengths),
        },
        "chosen_total_tokens": {"min": min(chosen_total_lengths), "max": max(chosen_total_lengths)},
        "rejected_total_tokens": {"min": min(rejected_total_lengths), "max": max(rejected_total_lengths)},
        "llamafactory_commit": source_commit,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("llamafactory", "torch", "transformers", "datasets", "accelerate", "peft", "trl")
        },
        "model_path": str(model_path.resolve()),
        "processor_files_sha256": {path.name: sha256(path) for path in processor_files},
    }


def load_current_parser(parser_file: Path, stats_path: Path) -> Callable[[str], object]:
    os.environ["NAVSIM_STAT_PATH"] = str(stats_path)
    spec = importlib.util.spec_from_file_location("safe_preference_navsim_helper", parser_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load parser module: {parser_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_text_waypoint_dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--rl-data", type=Path, required=True)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--trajectory-stats", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--llamafactory-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=18100)
    parser.add_argument("--expected-scenes", type=int, default=4525)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--processor-sample-size", type=int, default=30)
    parser.add_argument("--response-token-limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    source_status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_status:
        raise SystemExit("Source checkout is not clean")
    source_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    input_paths = [
        args.rollouts,
        args.rl_data,
        args.sft_data,
        args.trajectory_stats,
        args.train_manifest,
        args.dev_manifest,
        args.heldout_manifest,
    ]
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")

    stats = json.loads(args.trajectory_stats.read_text(encoding="utf-8"))
    means = np.asarray(stats["mean"], dtype=float)
    stds = np.asarray(stats["std"], dtype=float)
    if means.shape != (8, 3) or stds.shape != (8, 3) or np.any(stds <= 0):
        raise SystemExit("Trajectory stats must contain positive 8x3 mean/std arrays")
    parse_response = load_current_parser(
        repo_root / "EasyR1/verl/utils/reward_score/navsim/helper.py", args.trajectory_stats
    )
    rollouts = read_jsonl(args.rollouts)
    rl_rows = load_rl_rows(args.rl_data)
    sft_rows = json.loads(args.sft_data.read_text(encoding="utf-8"))
    report, join_failures, roundtrip_failures = audit_records(
        rollouts=rollouts,
        rl_rows=rl_rows,
        sft_rows=sft_rows,
        means=means,
        stds=stds,
        train_tokens=set(read_manifest(args.train_manifest)),
        dev_tokens=set(read_manifest(args.dev_manifest)),
        heldout_tokens=set(read_manifest(args.heldout_manifest)),
        image_root=args.image_root,
        parse_response=parse_response,
        expected_rows=args.expected_rows,
        expected_scenes=args.expected_scenes,
        expected_rollouts=args.expected_rollouts,
    )
    if report["all_core_gates_passed"]:
        processor_examples = build_processor_examples(
            rollouts=rollouts,
            rl_rows=rl_rows,
            sft_rows=sft_rows,
            means=means,
            stds=stds,
            parse_response=parse_response,
            sample_size=args.processor_sample_size,
            seed=args.seed,
        )
        report["processor"] = run_llamafactory_processor_audit(
            examples=processor_examples,
            llamafactory_root=args.llamafactory_root,
            model_path=args.model_path,
            image_root=args.image_root,
            response_limit=args.response_token_limit,
        )
    else:
        report["processor"] = {"passed": False, "reason": "core_gate_failed"}
    report["all_gates_passed"] = report["all_core_gates_passed"] and report["processor"]["passed"]

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "schema_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(args.output_dir / "join_failures.jsonl", join_failures)
    write_jsonl(args.output_dir / "roundtrip_failures.jsonl", roundtrip_failures)
    (args.output_dir / "input_sha256.txt").write_text(
        "".join(f"{sha256(path)}  {path.resolve()}\n" for path in input_paths), encoding="utf-8"
    )
    (args.output_dir / "source_commit.txt").write_text(source_commit + "\n", encoding="utf-8")
    if not report["all_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
