from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from projects.dataset_v3.inventory import sha256_file
from projects.dataset_v3.s1_pipeline import read_manifest


EXPECTED_GROUPS = 2000
EXPERIMENT_NAME = "v4_risk50_raw_g4_b4_seed20260827"
ALLOWED_RR_CONFIG_DIFFERENCES = {
    "data.train_files",
    "trainer.experiment_name",
    "trainer.save_checkpoint_path",
}


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value))


def config_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(config_differences(left[key], right[key], path))
        return differences
    return [] if left == right else [prefix]


def build_aligned_config(
    rr_config: dict[str, Any], *, train_parquet: Path, future_run_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    aligned = copy.deepcopy(rr_config)
    aligned["data"]["train_files"] = f"{train_parquet}@train"
    aligned["trainer"]["experiment_name"] = EXPERIMENT_NAME
    aligned["trainer"]["save_checkpoint_path"] = str(future_run_dir / "checkpoints")
    differences = config_differences(rr_config, aligned)
    if set(differences) != ALLOWED_RR_CONFIG_DIFFERENCES:
        raise ValueError(f"Unexpected RR-aligned config differences: {differences}")
    return aligned, differences


def runtime_input_config(resolved_config: dict[str, Any]) -> dict[str, Any]:
    runnable = copy.deepcopy(resolved_config)
    reward = runnable["worker"]["reward"]
    reward_name = reward.pop("reward_function_name", None)
    if reward_name is not None:
        reward["reward_function"] = f"{reward['reward_function']}:{reward_name}"
    return runnable


def resolve_with_current_runtime(config: dict[str, Any]) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from EasyR1.verl.trainer.config import PPOConfig

    merged = OmegaConf.merge(
        OmegaConf.structured(PPOConfig()), OmegaConf.create(runtime_input_config(config))
    )
    resolved: PPOConfig = OmegaConf.to_object(merged)
    resolved.deep_post_init()
    return _normalized(resolved.to_dict())


def validate_rr_contract(config: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "seed": config["data"]["seed"] == 20260827
        and config["worker"]["rollout"]["seed"] == 20260827,
        "raw_pdms": config["worker"]["reward"]["reward_function_name"]
        == "compute_score_raw_pdms",
        "groups": config["worker"]["rollout"]["n"] == 4
        and config["data"]["rollout_batch_size"] == 4
        and config["worker"]["actor"]["global_batch_size"] == 4,
        "optimizer": config["worker"]["actor"]["optim"]["lr"] == 1e-6
        and config["worker"]["actor"]["ppo_epochs"] == 1
        and config["worker"]["actor"]["optim"]["lr_scheduler_type"] == "constant",
        "kl": config["algorithm"]["use_kl_loss"] is True
        and config["algorithm"]["kl_penalty"] == "low_var_kl"
        and config["algorithm"]["kl_coef"] == 0.01,
        "lora": config["worker"]["actor"]["model"]["lora"]
        == {
            "rank": 8,
            "alpha": 16,
            "target_modules": "q_proj,k_proj,v_proj,o_proj",
            "exclude_modules": ".*visual.*",
        },
        "budget": config["trainer"]["max_steps"] == 500
        and config["trainer"]["val_steps"] == [100, 200, 300, 400, 500],
        "sampling": config["worker"]["rollout"]["temperature"] == 1.0
        and config["worker"]["rollout"]["top_p"] == 1.0,
    }
    if not all(checks.values()):
        raise ValueError(f"Historical RR config does not match its frozen contract: {checks}")
    return checks


def materialize_parquet(
    manifest: Path,
    screen_parquet: Path,
    rr_parquet: Path,
    output_manifest: Path,
    output_parquet: Path,
    data_root: Path,
) -> dict[str, Any]:
    tokens = read_manifest(manifest)
    if len(tokens) != EXPECTED_GROUPS or len(set(tokens)) != EXPECTED_GROUPS:
        raise ValueError("Risk50 manifest must contain 2,000 unique tokens")
    screen = pq.read_table(screen_parquet)
    rr = pq.read_table(rr_parquet)
    if screen.schema != rr.schema:
        raise ValueError("RR parquet schema differs from the Screen source schema")
    answers = screen.column("answer").to_pylist()
    screen_tokens = [str(answer["token"]) for answer in answers]
    if len(screen_tokens) != len(set(screen_tokens)):
        raise ValueError("Screen parquet contains duplicate answer tokens")
    index_by_token = {token: index for index, token in enumerate(screen_tokens)}
    missing = set(tokens) - set(index_by_token)
    if missing:
        raise ValueError(f"Risk50 manifest contains {len(missing)} tokens outside Screen parquet")
    selected = screen.take(pa.array([index_by_token[token] for token in tokens]))
    selected_tokens = [str(answer["token"]) for answer in selected.column("answer").to_pylist()]
    if selected_tokens != tokens:
        raise ValueError("Materialized Risk50 parquet order differs from its manifest")
    image_paths = [path for paths in selected.column("images").to_pylist() for path in paths]
    missing_images = [path for path in image_paths if not (data_root / path).is_file()]
    if missing_images:
        raise ValueError(f"Risk50 parquet references {len(missing_images)} missing images")
    output_manifest.write_bytes(manifest.read_bytes())
    pq.write_table(selected, output_parquet)
    written = pq.read_table(output_parquet)
    if written.schema != rr.schema or written.num_rows != EXPECTED_GROUPS:
        raise ValueError("Written Risk50 parquet failed schema or row-count verification")
    return {
        "rows": written.num_rows,
        "columns": written.column_names,
        "schema_matches_rr": written.schema == rr.schema,
        "manifest_order_exact": selected_tokens == tokens,
        "image_references": len(image_paths),
        "missing_images": 0,
    }


def rr_runtime_summary(rr_run_dir: Path) -> dict[str, Any]:
    start = int((rr_run_dir / "start_epoch.txt").read_text().strip())
    end = int((rr_run_dir / "end_epoch.txt").read_text().strip())
    with (rr_run_dir / "gpu_memory.csv").open(encoding="utf-8", newline="") as handle:
        memory_rows = list(csv.DictReader(handle))
    return {
        "wall_seconds": end - start,
        "max_gpu_memory_used_mib": max(int(row["memory_used_mib"]) for row in memory_rows),
        "gpu_memory_total_mib": max(int(row["memory_total_mib"]) for row in memory_rows),
    }


def prepare(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite V4 training preparation: {args.output_dir}")
    if args.future_run_dir.exists():
        raise FileExistsError(f"Future V4 training run already exists: {args.future_run_dir}")
    if args.source_status.read_text(encoding="utf-8").strip():
        raise ValueError("V4 preparation source tree is not clean")
    model_hash_lines = [
        line for line in args.model_hash_check.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(model_hash_lines) != 4 or any(not line.endswith(": OK") for line in model_hash_lines):
        raise ValueError("SFT Stage-2 model hash check does not match RR")
    args.output_dir.mkdir(parents=True)
    output_manifest = args.output_dir / "risk50_train_2000.txt"
    output_parquet = args.output_dir / "risk50_train_2000.parquet"
    parquet_report = materialize_parquet(
        args.frozen_manifest,
        args.screen_parquet,
        args.rr_parquet,
        output_manifest,
        output_parquet,
        args.data_root,
    )

    rr_config = json.loads(args.rr_config.read_text(encoding="utf-8"))
    rr_checks = validate_rr_contract(rr_config)
    current_rr = resolve_with_current_runtime(rr_config)
    runtime_differences = config_differences(rr_config, current_rr)
    if runtime_differences:
        raise ValueError(f"Current runtime changes historical RR config semantics: {runtime_differences}")
    aligned_config, allowed_differences = build_aligned_config(
        rr_config,
        train_parquet=output_parquet,
        future_run_dir=args.future_run_dir,
    )
    aligned_runtime = resolve_with_current_runtime(aligned_config)
    if config_differences(aligned_config, aligned_runtime):
        raise ValueError("Current runtime changes the generated V4 aligned config")
    config_path = args.output_dir / "risk50_rr_aligned_config.json"
    config_path.write_text(
        json.dumps(runtime_input_config(aligned_config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    monitor_tokens = read_manifest(args.monitor_manifest)
    risk_tokens = read_manifest(output_manifest)
    if set(risk_tokens) & set(monitor_tokens):
        raise ValueError("Risk50 optimizer data overlaps the frozen Train Monitor")
    m0 = json.loads(args.m0_protocol.read_text(encoding="utf-8"))
    if m0["status"] != "M0_FROZEN":
        raise ValueError("V4 preparation requires the frozen M0 protocol")
    free_bytes = shutil.disk_usage(args.output_dir).free
    report = {
        "status": "V4_RISK50_RR_ALIGNED_READY",
        "experiment_name": EXPERIMENT_NAME,
        "parquet": parquet_report,
        "rr_config_checks": rr_checks,
        "rr_runtime_config_drift": runtime_differences,
        "allowed_config_differences": allowed_differences,
        "train_monitor_overlap": 0,
        "rr_runtime_reference": rr_runtime_summary(args.rr_run_dir),
        "launch_gate": {
            "source_tree_clean": True,
            "model_hash_matches_rr": True,
            "free_bytes_at_prepare": free_bytes,
            "free_space_30_gib": free_bytes >= 30 * 1024**3,
            "gpu_idle_check": "DEFERRED_UNTIL_GPU_INSTANCE",
            "reward_port_check": "DEFERRED_UNTIL_GPU_INSTANCE",
            "future_run_absent": True,
        },
        "input_sha256": {
            "frozen_manifest": sha256_file(args.frozen_manifest),
            "screen_parquet": sha256_file(args.screen_parquet),
            "rr_parquet": sha256_file(args.rr_parquet),
            "rr_config": sha256_file(args.rr_config),
            "monitor_manifest": sha256_file(args.monitor_manifest),
            "monitor_parquet": sha256_file(args.monitor_parquet),
            "m0_protocol": sha256_file(args.m0_protocol),
        },
        "output_sha256": {
            "manifest": sha256_file(output_manifest),
            "parquet": sha256_file(output_parquet),
            "config": sha256_file(config_path),
        },
        "dev_accessed": False,
        "final_accessed": False,
        "gpu_training_authorized": False,
    }
    (args.output_dir / "v4_risk50_training_prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def smoke_loader(args: argparse.Namespace) -> None:
    from EasyR1.verl.utils.dataset import RLHFDataset, collate_fn
    from EasyR1.verl.utils.tokenizer import get_processor, get_tokenizer

    config = resolve_with_current_runtime(json.loads(args.config.read_text(encoding="utf-8")))
    data = config["data"]
    model_path = config["worker"]["actor"]["model"]["model_path"]
    tokenizer = get_tokenizer(model_path, trust_remote_code=True, use_fast=True)
    processor = get_processor(model_path, trust_remote_code=True, use_fast=True)
    dataset = RLHFDataset(
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
    if len(dataset) != EXPECTED_GROUPS:
        raise ValueError(f"Risk50 loader returned {len(dataset)} rows")
    indices = [0, EXPECTED_GROUPS // 3, EXPECTED_GROUPS * 2 // 3, EXPECTED_GROUPS - 1]
    examples = [dataset[index] for index in indices]
    batch = collate_fn(examples)
    sampled_tokens = [str(example["ground_truth"]["token"]) for example in examples]
    report = {
        "status": "V4_RISK50_DATALOADER_SMOKE_PASS",
        "dataset_rows": len(dataset),
        "sampled_indices": indices,
        "sampled_tokens": sampled_tokens,
        "batch_rows": len(examples),
        "input_shape": list(batch["input_ids"].shape),
        "dev_accessed": False,
        "final_accessed": False,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--frozen-manifest", type=Path, required=True)
    prep.add_argument("--screen-parquet", type=Path, required=True)
    prep.add_argument("--rr-parquet", type=Path, required=True)
    prep.add_argument("--rr-config", type=Path, required=True)
    prep.add_argument("--rr-run-dir", type=Path, required=True)
    prep.add_argument("--monitor-manifest", type=Path, required=True)
    prep.add_argument("--monitor-parquet", type=Path, required=True)
    prep.add_argument("--m0-protocol", type=Path, required=True)
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--future-run-dir", type=Path, required=True)
    prep.add_argument("--source-status", type=Path, required=True)
    prep.add_argument("--model-hash-check", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.set_defaults(function=prepare)
    smoke = commands.add_parser("smoke-loader")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.set_defaults(function=smoke_loader)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
