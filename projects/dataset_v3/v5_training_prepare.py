from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from projects.dataset_v3.inventory import sha256_file
from projects.dataset_v3.s1_pipeline import read_manifest
from projects.dataset_v3.v4_training_prepare import (
    config_differences,
    resolve_with_current_runtime,
    runtime_input_config,
    validate_rr_contract,
)


EXPECTED_GROUPS = 2000
EXPECTED_MONITOR = 256
ALLOWED_CONFIG_DIFFERENCES = {
    "data.train_files",
    "trainer.experiment_name",
    "trainer.save_checkpoint_path",
}
EXPERIMENT_NAMES = {
    "risk50": "v5_risk50_raw_g4_b4_seed20260827",
    "risk50_fals": "v5_risk50_fals_raw_g4_b4_seed20260827",
}
DATASET_FILENAMES = {
    "risk50": ("v5_risk50_2000.txt", "v5_risk50_2000.parquet"),
    "risk50_fals": ("v5_risk50_fals_2000.txt", "v5_risk50_fals_2000.parquet"),
}


def build_aligned_config(
    reference: dict[str, Any],
    *,
    dataset: str,
    train_parquet: Path,
    future_run_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    aligned = json.loads(json.dumps(reference))
    aligned["data"]["train_files"] = f"{train_parquet}@train"
    aligned["trainer"]["experiment_name"] = EXPERIMENT_NAMES[dataset]
    aligned["trainer"]["save_checkpoint_path"] = str(future_run_dir / "checkpoints")
    differences = config_differences(reference, aligned)
    if set(differences) != ALLOWED_CONFIG_DIFFERENCES:
        raise ValueError(f"Unexpected V5 aligned config differences: {differences}")
    validate_rr_contract(aligned)
    return aligned, differences


def validate_materialized_dataset(
    manifest: Path,
    parquet: Path,
    reference_parquet: Path,
    data_root: Path,
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    if expected_rows is None:
        expected_rows = EXPECTED_GROUPS
    tokens = read_manifest(manifest)
    if len(tokens) != expected_rows:
        raise ValueError(f"{manifest.name} must contain {expected_rows} unique tokens")
    table = pq.read_table(parquet)
    reference = pq.read_table(reference_parquet)
    if table.num_rows != expected_rows or table.schema != reference.schema:
        raise ValueError(f"{parquet.name} row count or schema differs from the GRPO reference")
    parquet_tokens = [str(answer["token"]) for answer in table.column("answer").to_pylist()]
    if parquet_tokens != tokens:
        raise ValueError(f"{parquet.name} answer order differs from its manifest")
    if len(parquet_tokens) != len(set(parquet_tokens)):
        raise ValueError(f"{parquet.name} contains duplicate answer tokens")
    image_paths = [path for paths in table.column("images").to_pylist() for path in paths]
    missing_images = [path for path in image_paths if not (data_root / path).is_file()]
    if missing_images:
        raise ValueError(f"{parquet.name} references {len(missing_images)} missing images")
    return {
        "rows": table.num_rows,
        "columns": table.column_names,
        "schema_matches_reference": True,
        "manifest_order_exact": True,
        "image_references": len(image_paths),
        "missing_images": 0,
    }


def validate_dataset_run(dataset_run: Path) -> dict[str, Any]:
    if not (dataset_run / "COMPLETE").is_file():
        raise ValueError("Corrected V5 dataset run is not COMPLETE")
    if (dataset_run / "exit_code").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("Corrected V5 dataset run did not exit successfully")
    results = dataset_run / "results"
    report_path = results / "v5_risk_fals_dataset_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "V5_RISK_FALS_DATASETS_READY":
        raise ValueError("Corrected V5 dataset report is not ready")
    if any(report.get(key) for key in ("dev_accessed", "final_accessed", "gpu_used", "training_launched")):
        raise ValueError("V5 dataset preparation crossed its CPU-only data boundary")
    output_paths = {
        "risk50_manifest": results / "v5_risk50_2000.txt",
        "risk50_parquet": results / "v5_risk50_2000.parquet",
        "risk50_fals_manifest": results / "v5_risk50_fals_2000.txt",
        "risk50_fals_parquet": results / "v5_risk50_fals_2000.parquet",
        "membership": results / "v5_scene_fals_membership.csv",
    }
    for name, path in output_paths.items():
        if sha256_file(path) != report["output_sha256"][name]:
            raise ValueError(f"Corrected V5 dataset output hash mismatch: {name}")
    return report


def _model_hash_check(path: Path) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 4 or any(not line.endswith(": OK") for line in lines):
        raise ValueError("SFT Stage-2 model hash check does not match the RR reference")


def prepare(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite V5 training preparation: {args.output_dir}")
    future_paths = (
        args.risk50_future_run,
        args.risk50_fals_future_run,
        args.risk50_debug_dir,
        args.risk50_fals_debug_dir,
    )
    if any(path.exists() for path in future_paths):
        raise FileExistsError("A future V5 formal-run or debug directory already exists")
    if args.source_status.read_text(encoding="utf-8").strip():
        raise ValueError("V5 preparation source tree is not clean")
    _model_hash_check(args.model_hash_check)
    dataset_report = validate_dataset_run(args.dataset_run)
    results = args.dataset_run / "results"

    monitor_report = validate_materialized_dataset(
        args.monitor_manifest,
        args.monitor_parquet,
        args.rr_parquet,
        args.data_root,
        expected_rows=EXPECTED_MONITOR,
    )
    monitor_tokens = read_manifest(args.monitor_manifest)
    dataset_reports = {}
    dataset_tokens = {}
    for dataset, (manifest_name, parquet_name) in DATASET_FILENAMES.items():
        manifest = results / manifest_name
        parquet = results / parquet_name
        dataset_reports[dataset] = validate_materialized_dataset(
            manifest, parquet, args.rr_parquet, args.data_root
        )
        dataset_tokens[dataset] = read_manifest(manifest)
        if set(dataset_tokens[dataset]) & set(monitor_tokens):
            raise ValueError(f"{dataset} overlaps the frozen Train Monitor")

    m0 = json.loads(args.m0_protocol.read_text(encoding="utf-8"))
    if m0.get("status") != "M0_FROZEN":
        raise ValueError("V5 preparation requires the frozen M0 protocol")
    reference = json.loads(args.rr_config.read_text(encoding="utf-8"))
    reference_checks = validate_rr_contract(reference)
    if reference["data"]["val_files"] != f"{args.monitor_parquet}@train":
        raise ValueError("RR reference config does not use the frozen Train Monitor parquet")
    if Path(reference["data"]["image_dir"]) != args.data_root:
        raise ValueError("RR reference config does not use the frozen data root")
    if config_differences(reference, resolve_with_current_runtime(reference)):
        raise ValueError("Current runtime changes historical RR config semantics")

    args.output_dir.mkdir(parents=True)
    configs = {}
    config_diffs = {}
    future_runs = {
        "risk50": args.risk50_future_run,
        "risk50_fals": args.risk50_fals_future_run,
    }
    for dataset, (_, parquet_name) in DATASET_FILENAMES.items():
        aligned, differences = build_aligned_config(
            reference,
            dataset=dataset,
            train_parquet=results / parquet_name,
            future_run_dir=future_runs[dataset],
        )
        if config_differences(aligned, resolve_with_current_runtime(aligned)):
            raise ValueError(f"Current runtime changes the generated {dataset} config")
        config_path = args.output_dir / f"v5_{dataset}_raw_config.json"
        config_path.write_text(
            json.dumps(runtime_input_config(aligned), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        configs[dataset] = (aligned, config_path)
        config_diffs[dataset] = differences
    between_differences = config_differences(configs["risk50"][0], configs["risk50_fals"][0])
    if set(between_differences) != ALLOWED_CONFIG_DIFFERENCES:
        raise ValueError(f"Unexpected difference between V5 configs: {between_differences}")

    free_bytes = shutil.disk_usage(args.output_dir).free
    report = {
        "status": "V5_GPU_PREPARATION_READY",
        "training_order": ["V5-RISK50", "V5-RISK50-FALS"],
        "second_round_gate": {
            "required_first_cell": "V5-RISK50",
            "required_status": "COMPLETE",
            "required_exit_code": 0,
            "verify_first_run_result_sha256": True,
        },
        "training_evidence": {
            "export_after_each_v5_run": True,
            "files": [
                "training_history.csv",
                "training_curves.svg",
                "training_curve_summary.json",
                "representative_train_samples.jsonl",
                "training_evidence_manifest.json",
            ],
            "included_in_result_sha256": True,
        },
        "datasets": dataset_reports,
        "train_monitor": monitor_report,
        "dataset_lexicographic_optima": dataset_report["datasets"]["risk50_fals"][
            "lexicographic_optima"
        ],
        "train_monitor_overlap": {dataset: 0 for dataset in DATASET_FILENAMES},
        "rr_config_checks": reference_checks,
        "allowed_config_differences": config_diffs,
        "between_v5_config_differences": between_differences,
        "launch_gate": {
            "source_tree_clean": True,
            "model_hash_matches_rr": True,
            "future_run_dirs_absent": True,
            "future_debug_dirs_absent": True,
            "free_bytes_at_prepare": free_bytes,
            "free_space_30_gib": free_bytes >= 30 * 1024**3,
            "gpu_idle_check": "DEFERRED_UNTIL_GPU_INSTANCE",
            "reward_port_check": "DEFERRED_UNTIL_GPU_INSTANCE",
        },
        "input_sha256": {
            "dataset_report": sha256_file(results / "v5_risk_fals_dataset_report.json"),
            "risk50_manifest": sha256_file(results / DATASET_FILENAMES["risk50"][0]),
            "risk50_parquet": sha256_file(results / DATASET_FILENAMES["risk50"][1]),
            "risk50_fals_manifest": sha256_file(results / DATASET_FILENAMES["risk50_fals"][0]),
            "risk50_fals_parquet": sha256_file(results / DATASET_FILENAMES["risk50_fals"][1]),
            "rr_parquet": sha256_file(args.rr_parquet),
            "rr_config": sha256_file(args.rr_config),
            "monitor_manifest": sha256_file(args.monitor_manifest),
            "monitor_parquet": sha256_file(args.monitor_parquet),
            "m0_protocol": sha256_file(args.m0_protocol),
        },
        "output_sha256": {
            dataset: sha256_file(config_path) for dataset, (_, config_path) in configs.items()
        },
        "dev_accessed": False,
        "final_accessed": False,
        "gpu_used": False,
        "training_launched": False,
    }
    (args.output_dir / "v5_training_prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def smoke_loaders(args: argparse.Namespace) -> None:
    from EasyR1.verl.utils.dataset import RLHFDataset, collate_fn
    from EasyR1.verl.utils.tokenizer import get_processor, get_tokenizer

    configs = {
        "risk50": resolve_with_current_runtime(
            json.loads(args.risk50_config.read_text(encoding="utf-8"))
        ),
        "risk50_fals": resolve_with_current_runtime(
            json.loads(args.risk50_fals_config.read_text(encoding="utf-8"))
        ),
    }
    model_paths = {config["worker"]["actor"]["model"]["model_path"] for config in configs.values()}
    if len(model_paths) != 1:
        raise ValueError("V5 configs do not use the same Stage-2 model")
    model_path = model_paths.pop()
    tokenizer = get_tokenizer(model_path, trust_remote_code=True, use_fast=True)
    processor = get_processor(model_path, trust_remote_code=True, use_fast=True)
    reports = {}
    indices = [0, EXPECTED_GROUPS // 3, EXPECTED_GROUPS * 2 // 3, EXPECTED_GROUPS - 1]
    for dataset, config in configs.items():
        data = config["data"]
        loaded = RLHFDataset(
            data_path=data["train_files"],
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=data["prompt_key"],
            answer_key=data["answer_key"],
            image_key=data["image_key"],
            video_key=data["video_key"],
            image_dir=data["image_dir"],
            max_prompt_length=data["max_prompt_length"],
            truncation="right",
            min_pixels=data["min_pixels"],
            max_pixels=data["max_pixels"],
            filter_overlong_prompts=False,
        )
        if len(loaded) != EXPECTED_GROUPS:
            raise ValueError(f"{dataset} loader returned {len(loaded)} rows")
        examples = [loaded[index] for index in indices]
        batch = collate_fn(examples)
        reports[dataset] = {
            "dataset_rows": len(loaded),
            "sampled_indices": indices,
            "sampled_tokens": [str(example["ground_truth"]["token"]) for example in examples],
            "batch_rows": len(examples),
            "input_shape": list(batch["input_ids"].shape),
        }
    args.output.write_text(
        json.dumps(
            {
                "status": "V5_DATALOADER_SMOKE_PASS",
                "datasets": reports,
                "dev_accessed": False,
                "final_accessed": False,
                "gpu_used": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset-run", type=Path, required=True)
    prep.add_argument("--rr-parquet", type=Path, required=True)
    prep.add_argument("--rr-config", type=Path, required=True)
    prep.add_argument("--monitor-manifest", type=Path, required=True)
    prep.add_argument("--monitor-parquet", type=Path, required=True)
    prep.add_argument("--m0-protocol", type=Path, required=True)
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--risk50-future-run", type=Path, required=True)
    prep.add_argument("--risk50-fals-future-run", type=Path, required=True)
    prep.add_argument("--risk50-debug-dir", type=Path, required=True)
    prep.add_argument("--risk50-fals-debug-dir", type=Path, required=True)
    prep.add_argument("--source-status", type=Path, required=True)
    prep.add_argument("--model-hash-check", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.set_defaults(function=prepare)
    smoke = commands.add_parser("smoke-loaders")
    smoke.add_argument("--risk50-config", type=Path, required=True)
    smoke.add_argument("--risk50-fals-config", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.set_defaults(function=smoke_loaders)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
