from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_composed_state(base: Path, adapter: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    config = AutoConfig.from_pretrained(base, trust_remote_code=True)
    auto_class = (
        AutoModelForImageTextToText
        if type(config) in AutoModelForImageTextToText._model_mapping.keys()
        else AutoModelForCausalLM
    )
    base_model = auto_class.from_pretrained(
        base,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_model.tie_weights()
    composed = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
    return composed.state_dict(), load_file(adapter / "adapter_model.safetensors", device="cpu")


def mapped_adapter_state(adapter_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        ): value
        for key, value in adapter_state.items()
    }


def verify_adapter_tensors(
    composed_state: dict[str, torch.Tensor], adapter_state: dict[str, torch.Tensor]
) -> int:
    mapped = mapped_adapter_state(adapter_state)
    lora_keys = {key for key in composed_state if "lora_" in key}
    if set(mapped) != lora_keys:
        raise ValueError("Adapter keys do not exactly cover composed LoRA tensors")
    for key, expected in mapped.items():
        if not torch.equal(composed_state[key].detach().cpu(), expected):
            raise ValueError(f"Adapter tensor differs from composed tensor: {key}")
    return len(lora_keys)


def compare_states(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    if set(actual) != set(expected):
        raise ValueError("Composed and reference actor keys differ")
    for key, expected_tensor in expected.items():
        if not torch.equal(actual[key].detach().cpu(), expected_tensor):
            raise ValueError(f"Composed tensor differs from reference actor: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--mode", choices=("verify", "materialize"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    full_path = args.actor / "model_world_size_1_rank_0.pt"
    adapter_path = args.actor / "lora_adapter"
    adapter_weights = adapter_path / "adapter_model.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    if args.report.exists():
        raise FileExistsError(args.report)
    for path in (args.base, adapter_weights, adapter_config):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.mode == "verify" and not full_path.exists():
        raise FileNotFoundError(full_path)
    if args.mode == "materialize" and full_path.exists():
        raise FileExistsError(full_path)

    started = time.time()
    composed_state, adapter_state = load_composed_state(args.base, adapter_path)
    lora_tensors = verify_adapter_tensors(composed_state, adapter_state)
    if args.mode == "verify":
        reference = torch.load(full_path, map_location="cpu", weights_only=True, mmap=True)
        compare_states(composed_state, reference)
        status = "BASE_PLUS_LORA_EXACT_RECOVERY_PASS"
    else:
        temporary = full_path.with_name(f".{full_path.name}.materializing")
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            cpu_state = {key: value.detach().cpu() for key, value in composed_state.items()}
            torch.save(cpu_state, temporary)
            saved = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
            compare_states(composed_state, saved)
            os.replace(temporary, full_path)
        finally:
            temporary.unlink(missing_ok=True)
        status = "BASE_PLUS_LORA_ACTOR_MATERIALIZED"

    report = {
        "status": status,
        "mode": args.mode,
        "base_model": str(args.base),
        "actor": str(args.actor),
        "full_actor": str(full_path),
        "full_actor_bytes": full_path.stat().st_size,
        "state_tensors": len(composed_state),
        "lora_tensors": lora_tensors,
        "adapter_to_composed_lora_exact": True,
        "composed_to_full_actor_exact": True,
        "adapter_sha256": sha256_file(adapter_weights),
        "adapter_config_sha256": sha256_file(adapter_config),
        "wall_seconds": time.time() - started,
        "max_gpu_memory_allocated_mib": torch.cuda.max_memory_allocated() // (1024 * 1024),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
