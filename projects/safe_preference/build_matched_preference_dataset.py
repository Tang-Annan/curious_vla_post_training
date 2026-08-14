"""Build the preregistered matched Tier-A RSFT/DPO datasets."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from projects.safe_preference.analyze_preference_dataset import is_safe, is_valid
from projects.safe_preference.build_preference_dataset import (
    _poses_from_parser_output,
    format_response,
    load_current_parser,
    load_rl_rows,
    parse_unique_json,
    read_jsonl,
    read_manifest,
    sha256,
    validate_template,
)


QUALITY_FIELDS = ("pdms_scaled", "ego_progress", "history_comfort", "pdms")


def quality_tuple(row: dict, rollout_index: int) -> tuple[float, ...]:
    values = tuple(float(row[field]) for field in QUALITY_FIELDS)
    safety_sum = sum(
        float(row[field])
        for field in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
        )
    )
    quality = values + (safety_sum, -rollout_index)
    if not all(np.isfinite(value) for value in quality):
        raise ValueError("Quality tuple contains a non-finite value")
    return quality


def select_matched_scenes(
    rows: list[dict], *, expected_rollouts: int, pair_count: int
) -> tuple[list[dict], dict]:
    groups = collections.defaultdict(list)
    for row in rows:
        token = row.get("token")
        groups[token].append(row)

    filters = collections.Counter()
    eligible = []
    for token, group in groups.items():
        if len(group) != expected_rollouts:
            filters["rollout_count"] += 1
            continue
        if not all(is_valid(row) for row in group):
            filters["parse_or_shape"] += 1
            continue
        safe_flags = [is_safe(row) for row in group]
        if all(safe_flags) or not any(safe_flags):
            filters["not_mixed_safety"] += 1
            continue
        if max(float(row["pdms_scaled"]) for row in group) <= min(
            float(row["pdms_scaled"]) for row in group
        ):
            filters["no_positive_pdms_gap"] += 1
            continue

        indexed = [
            {"rollout_index": index, "row": row, "quality": quality_tuple(row, index)}
            for index, row in enumerate(group)
        ]
        safe_rows = [item for item in indexed if is_safe(item["row"])]
        unsafe_rows = [item for item in indexed if not is_safe(item["row"])]
        chosen = max(indexed, key=lambda item: item["quality"])
        if chosen["rollout_index"] != max(safe_rows, key=lambda item: item["quality"])["rollout_index"]:
            filters["common_chosen_not_safe_best"] += 1
            continue
        easy = min(indexed, key=lambda item: item["quality"])
        if is_safe(easy["row"]):
            filters["easy_rejected_safe"] += 1
            continue
        hard = max(unsafe_rows, key=lambda item: item["quality"])
        if easy["rollout_index"] == hard["rollout_index"]:
            filters["same_rejected"] += 1
            continue
        if easy["quality"][:-1] == hard["quality"][:-1]:
            filters["rejected_differs_only_by_index_tie_break"] += 1
            continue
        eligible.append(
            {
                "token": token,
                "chosen": chosen,
                "easy_rejected": easy,
                "hard_rejected": hard,
            }
        )

    eligible.sort(key=lambda item: str(item["token"]))
    eligible.sort(key=lambda item: item["hard_rejected"]["quality"], reverse=True)
    if len(eligible) < pair_count:
        raise ValueError(f"Only {len(eligible)} matched scenes are eligible; {pair_count} required")
    selected = eligible[:pair_count]
    hard_minus_easy = {
        field: [
            float(item["hard_rejected"]["row"][field])
            - float(item["easy_rejected"]["row"][field])
            for item in selected
        ]
        for field in QUALITY_FIELDS
    }
    return selected, {
        "scenes": len(groups),
        "strict_eligible": len(eligible),
        "selected": len(selected),
        "excluded": dict(sorted(filters.items())),
        "selection_order": "hard_rejected_quality_desc_then_token_asc",
        "selected_safety": {
            "chosen_safe": sum(is_safe(item["chosen"]["row"]) for item in selected),
            "easy_rejected_unsafe": sum(
                not is_safe(item["easy_rejected"]["row"]) for item in selected
            ),
            "hard_rejected_unsafe": sum(
                not is_safe(item["hard_rejected"]["row"]) for item in selected
            ),
            "different_rejected_index": sum(
                item["easy_rejected"]["rollout_index"]
                != item["hard_rejected"]["rollout_index"]
                for item in selected
            ),
        },
        "hard_minus_easy": {
            field: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for field, values in hard_minus_easy.items()
        },
    }


def _response(template: dict, row: dict, means: np.ndarray, stds: np.ndarray) -> str:
    poses = np.asarray(row["poses"], dtype=float)
    return format_response(template, (poses - means) / stds)


def build_matched_datasets(
    *,
    selected: list[dict],
    rl_rows: list[dict],
    sft_rows: list[dict],
    means: np.ndarray,
    stds: np.ndarray,
    parse_response,
) -> tuple[dict[str, list[dict]], list[dict]]:
    rl_by_token = {row["answer"]["token"]: row for row in rl_rows}
    sft_by_token = {row["id"]: row for row in sft_rows}
    datasets = {"m2": [], "m3": [], "m4": []}
    manifest = []

    for item in selected:
        token = item["token"]
        if token not in rl_by_token or token not in sft_by_token:
            raise ValueError(f"Missing RL/SFT join for selected token: {token}")
        rl_row = rl_by_token[token]
        sft_row = sft_by_token[token]
        conversations = sft_row.get("conversations", [])
        if len(conversations) != 2 or conversations[0].get("from") != "human":
            raise ValueError(f"Invalid SFT conversations for selected token: {token}")
        template = validate_template(conversations[1].get("value"), parse_response)
        if rl_row.get("images") != sft_row.get("image"):
            raise ValueError(f"RL/SFT image mismatch for selected token: {token}")

        chosen = _response(template, item["chosen"]["row"], means, stds)
        easy_rejected = _response(template, item["easy_rejected"]["row"], means, stds)
        hard_rejected = _response(template, item["hard_rejected"]["row"], means, stds)
        prompt = {"from": "human", "value": rl_row["problem"]}
        common = {
            "token": token,
            "images": rl_row["images"],
            "system": sft_row["system"],
        }
        datasets["m2"].append(
            {
                **common,
                "conversations": [prompt, {"from": "gpt", "value": chosen}],
            }
        )
        for name, rejected in (("m3", easy_rejected), ("m4", hard_rejected)):
            datasets[name].append(
                {
                    **common,
                    "conversations": [prompt],
                    "chosen": {"from": "gpt", "value": chosen},
                    "rejected": {"from": "gpt", "value": rejected},
                }
            )
        manifest.append(
            {
                "token": token,
                "chosen_index": item["chosen"]["rollout_index"],
                "easy_rejected_index": item["easy_rejected"]["rollout_index"],
                "hard_rejected_index": item["hard_rejected"]["rollout_index"],
                "chosen_quality": list(item["chosen"]["quality"]),
                "easy_rejected_quality": list(item["easy_rejected"]["quality"]),
                "hard_rejected_quality": list(item["hard_rejected"]["quality"]),
                "easy_rejected_safe": is_safe(item["easy_rejected"]["row"]),
                "hard_rejected_safe": is_safe(item["hard_rejected"]["row"]),
            }
        )
    return datasets, manifest


def serialize_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def audit_datasets(
    *,
    datasets: dict[str, list[dict]],
    selected: list[dict],
    train_tokens: set[str],
    dev_tokens: set[str],
    heldout_tokens: set[str],
    means: np.ndarray,
    stds: np.ndarray,
    parse_response,
    pair_count: int,
) -> dict:
    selected_by_token = {item["token"]: item for item in selected}
    token_lists = {name: [row["token"] for row in rows] for name, rows in datasets.items()}
    common_tokens = set(token_lists["m2"])
    byte_identical_chosen = 0
    different_unsafe_rejected = 0
    parsed = 0
    roundtrip_passed = 0
    max_abs_error = 0.0

    for m2, m3, m4 in zip(datasets["m2"], datasets["m3"], datasets["m4"], strict=True):
        token = m2["token"]
        item = selected_by_token[token]
        chosen_values = (
            m2["conversations"][1]["value"],
            m3["chosen"]["value"],
            m4["chosen"]["value"],
        )
        if chosen_values[0] == chosen_values[1] == chosen_values[2]:
            byte_identical_chosen += 1
        if (
            m3["rejected"]["value"] != m4["rejected"]["value"]
            and not is_safe(item["easy_rejected"]["row"])
            and not is_safe(item["hard_rejected"]["row"])
        ):
            different_unsafe_rejected += 1

        response_rows = (
            (chosen_values[0], item["chosen"]["row"]),
            (chosen_values[1], item["chosen"]["row"]),
            (chosen_values[2], item["chosen"]["row"]),
            (m3["rejected"]["value"], item["easy_rejected"]["row"]),
            (m4["rejected"]["value"], item["hard_rejected"]["row"]),
        )
        for response, source_row in response_rows:
            parsed_json = parse_unique_json(response)
            if list(parsed_json).count("future_trajectory") != 1:
                raise ValueError("Serialized response does not have exactly one future_trajectory")
            parsed += 1
            poses = _poses_from_parser_output(parse_response(response))
            error = float(np.max(np.abs(poses * stds + means - np.asarray(source_row["poses"]))))
            max_abs_error = max(max_abs_error, error)
            if poses.shape == (8, 3) and error <= 1e-4:
                roundtrip_passed += 1

    gates = {
        "dataset_sizes": all(len(rows) == pair_count for rows in datasets.values()),
        "unique_tokens": all(len(tokens) == len(set(tokens)) == pair_count for tokens in token_lists.values()),
        "matched_token_order": token_lists["m2"] == token_lists["m3"] == token_lists["m4"],
        "train_only": common_tokens <= train_tokens,
        "zero_dev_overlap": not (common_tokens & dev_tokens),
        "zero_heldout_overlap": not (common_tokens & heldout_tokens),
        "chosen_byte_identical": byte_identical_chosen == pair_count,
        "rejected_different_and_unsafe": different_unsafe_rejected == pair_count,
        "json_valid": parsed == pair_count * 5,
        "roundtrip": roundtrip_passed == parsed and max_abs_error <= 1e-4,
    }
    return {
        "gates": gates,
        "all_core_gates_passed": all(gates.values()),
        "dataset_rows": {name: len(rows) for name, rows in datasets.items()},
        "common_unique_tokens": len(common_tokens),
        "chosen_byte_identical": byte_identical_chosen,
        "rejected_different_and_unsafe": different_unsafe_rejected,
        "json_responses_validated": parsed,
        "roundtrip_passed": roundtrip_passed,
        "roundtrip_max_abs_error": max_abs_error,
    }


def run_llamafactory_audit(
    *,
    datasets: dict[str, list[dict]],
    llamafactory_root: Path,
    model_path: Path,
    image_root: Path,
    response_limit: int,
    total_limit: int,
    image_max_pixels: int,
    batch_size: int,
) -> dict:
    sys.path.insert(0, str(llamafactory_root / "src"))
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor.pairwise import PairwiseDatasetProcessor
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    commit = subprocess.run(
        ["git", "-C", str(llamafactory_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(llamafactory_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("LLaMA-Factory source checkout is not clean")

    model_args = ModelArguments(
        model_name_or_path=str(model_path), image_max_pixels=image_max_pixels
    )
    tokenizer_module = load_tokenizer(model_args)
    data_args = DataArguments(template="qwen2_vl", cutoff_len=65536, media_dir=str(image_root))
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    results = {}

    for name, examples in datasets.items():
        ranking = name != "m2"
        dataset_attr = DatasetAttr("file", f"{name}_processor_audit")
        columns = {
            "messages": "conversations",
            "images": "images",
            "system": "system",
        }
        if ranking:
            columns.update({"chosen": "chosen", "rejected": "rejected"})
        dataset_attr.join(
            {"formatting": "sharegpt", "ranking": ranking, "columns": columns}
        )
        converter = SharegptDatasetConverter(dataset_attr, data_args)
        processor_class = PairwiseDatasetProcessor if ranking else SupervisedDatasetProcessor
        processor = processor_class(
            template=template,
            tokenizer=tokenizer_module["tokenizer"],
            processor=tokenizer_module["processor"],
            data_args=data_args,
        )
        response_lengths = []
        total_lengths = []
        rejected_response_lengths = []
        rejected_total_lengths = []
        image_count = 0
        for start in range(0, len(examples), batch_size):
            aligned = [converter(example) for example in examples[start : start + batch_size]]
            batch = {key: [example[key] for example in aligned] for key in aligned[0]}
            processed = processor.preprocess_dataset(batch)
            if ranking:
                response_lengths.extend(
                    sum(token != IGNORE_INDEX for token in labels)
                    for labels in processed["chosen_labels"]
                )
                rejected_response_lengths.extend(
                    sum(token != IGNORE_INDEX for token in labels)
                    for labels in processed["rejected_labels"]
                )
                total_lengths.extend(len(tokens) for tokens in processed["chosen_input_ids"])
                rejected_total_lengths.extend(
                    len(tokens) for tokens in processed["rejected_input_ids"]
                )
            else:
                response_lengths.extend(
                    sum(token != IGNORE_INDEX for token in labels) for labels in processed["labels"]
                )
                total_lengths.extend(len(tokens) for tokens in processed["input_ids"])
            image_count += len(processed["images"])
            print(f"processor {name}: {min(start + batch_size, len(examples))}/{len(examples)}", flush=True)

        all_response_lengths = response_lengths + rejected_response_lengths
        all_total_lengths = total_lengths + rejected_total_lengths
        passed = (
            tokenizer_module["processor"] is not None
            and len(response_lengths) == len(examples)
            and (not ranking or len(rejected_response_lengths) == len(examples))
            and image_count == len(examples)
            and max(all_response_lengths) <= response_limit
            and max(all_total_lengths) <= total_limit
        )
        results[name] = {
            "passed": passed,
            "examples": len(response_lengths),
            "images": image_count,
            "response_tokens": {"min": min(all_response_lengths), "max": max(all_response_lengths)},
            "total_tokens": {"min": min(all_total_lengths), "max": max(all_total_lengths)},
        }

    processor_files = [
        model_path / name
        for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json")
    ]
    return {
        "passed": all(result["passed"] for result in results.values()),
        "datasets": results,
        "template": "qwen2_vl",
        "processor": type(tokenizer_module["processor"]).__name__,
        "response_token_limit": response_limit,
        "total_token_limit": total_limit,
        "image_max_pixels": image_max_pixels,
        "llamafactory_commit": commit,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("llamafactory", "torch", "transformers", "datasets", "accelerate", "peft", "trl")
        },
        "model_path": str(model_path.resolve()),
        "processor_files_sha256": {path.name: sha256(path) for path in processor_files},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--rl-data", type=Path, required=True)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--trajectory-stats", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--p1-audit", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--llamafactory-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=18100)
    parser.add_argument("--expected-scenes", type=int, default=4525)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--pair-count", type=int, default=960)
    parser.add_argument("--audit-sample-size", type=int, default=30)
    parser.add_argument("--response-token-limit", type=int, default=512)
    parser.add_argument("--total-token-limit", type=int, default=4096)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--processor-batch-size", type=int, default=30)
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
        args.p1_audit,
    ]
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    p1_audit = json.loads(args.p1_audit.read_text(encoding="utf-8"))
    if not p1_audit.get("all_gates_passed"):
        raise SystemExit("P1-S audit did not pass")

    stats = json.loads(args.trajectory_stats.read_text(encoding="utf-8"))
    means = np.asarray(stats["mean"], dtype=float)
    stds = np.asarray(stats["std"], dtype=float)
    if means.shape != (8, 3) or stds.shape != (8, 3) or np.any(stds <= 0):
        raise SystemExit("Trajectory stats must contain positive 8x3 mean/std arrays")
    parse_response = load_current_parser(
        repo_root / "EasyR1/verl/utils/reward_score/navsim/helper.py", args.trajectory_stats
    )
    rows = read_jsonl(args.rollouts)
    if len(rows) != args.expected_rows or len({row.get("token") for row in rows}) != args.expected_scenes:
        raise SystemExit("D0 row or scene count does not match the preregistration")
    train_tokens = set(read_manifest(args.train_manifest))
    dev_tokens = set(read_manifest(args.dev_manifest))
    heldout_tokens = set(read_manifest(args.heldout_manifest))
    selected, selection_stats = select_matched_scenes(
        rows, expected_rollouts=args.expected_rollouts, pair_count=args.pair_count
    )
    rl_rows = load_rl_rows(args.rl_data)
    sft_rows = json.loads(args.sft_data.read_text(encoding="utf-8"))
    datasets, pair_manifest = build_matched_datasets(
        selected=selected,
        rl_rows=rl_rows,
        sft_rows=sft_rows,
        means=means,
        stds=stds,
        parse_response=parse_response,
    )
    repeated, repeated_manifest = build_matched_datasets(
        selected=selected,
        rl_rows=rl_rows,
        sft_rows=sft_rows,
        means=means,
        stds=stds,
        parse_response=parse_response,
    )
    deterministic = all(
        serialize_json(datasets[name]) == serialize_json(repeated[name]) for name in datasets
    ) and serialize_json(pair_manifest) == serialize_json(repeated_manifest)
    core_audit = audit_datasets(
        datasets=datasets,
        selected=selected,
        train_tokens=train_tokens,
        dev_tokens=dev_tokens,
        heldout_tokens=heldout_tokens,
        means=means,
        stds=stds,
        parse_response=parse_response,
        pair_count=args.pair_count,
    )
    core_audit["gates"]["repeat_build_byte_identical"] = deterministic
    core_audit["all_core_gates_passed"] = all(core_audit["gates"].values())
    if not core_audit["all_core_gates_passed"]:
        raise SystemExit("M1 core dataset gate failed before processor audit")

    processor_audit = run_llamafactory_audit(
        datasets=datasets,
        llamafactory_root=args.llamafactory_root,
        model_path=args.model_path,
        image_root=args.image_root,
        response_limit=args.response_token_limit,
        total_limit=args.total_token_limit,
        image_max_pixels=args.image_max_pixels,
        batch_size=args.processor_batch_size,
    )
    report = {
        "status": "passed" if processor_audit["passed"] else "processor_failed",
        "all_gates_passed": core_audit["all_core_gates_passed"] and processor_audit["passed"],
        "selection": selection_stats,
        "core_audit": core_audit,
        "processor_audit": processor_audit,
        "seed": args.seed,
    }

    output_files = {
        "m2_rsft.json": serialize_json(datasets["m2"]),
        "m3_pdms_easyneg_dpo.json": serialize_json(datasets["m3"]),
        "m4_safety_hardneg_dpo.json": serialize_json(datasets["m4"]),
        "pair_manifest.json": serialize_json(pair_manifest),
        "pair_stats.json": serialize_json(report),
    }
    sample = sorted(
        range(len(pair_manifest)),
        key=lambda index: hashlib.sha256(
            f"{args.seed}:{pair_manifest[index]['token']}".encode()
        ).hexdigest(),
    )[: args.audit_sample_size]
    audit_sample = [
        {
            "selection": pair_manifest[index],
            "m2": datasets["m2"][index],
            "m3": datasets["m3"][index],
            "m4": datasets["m4"][index],
        }
        for index in sample
    ]
    output_files["audit_sample_30.json"] = serialize_json(audit_sample)
    output_files["train_tokens.txt"] = "".join(f"{item['token']}\n" for item in pair_manifest).encode()

    args.output_dir.mkdir(parents=True)
    for name, content in output_files.items():
        (args.output_dir / name).write_bytes(content)
    (args.output_dir / "input_sha256.txt").write_text(
        "".join(f"{sha256(path)}  {path.resolve()}\n" for path in input_paths), encoding="utf-8"
    )
    (args.output_dir / "dataset_sha256.txt").write_text(
        "".join(
            f"{hashlib.sha256(content).hexdigest()}  {name}\n"
            for name, content in sorted(output_files.items())
        ),
        encoding="utf-8",
    )
    (args.output_dir / "source_commit.txt").write_text(source_commit + "\n", encoding="utf-8")
    (args.output_dir / "llamafactory_commit.txt").write_text(
        processor_audit["llamafactory_commit"] + "\n", encoding="utf-8"
    )
    if not report["all_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
