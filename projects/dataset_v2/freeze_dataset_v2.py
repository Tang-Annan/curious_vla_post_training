#!/usr/bin/env python3
"""Verify and freeze the completed Dataset V2 assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_cache_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"token", "log_name", "split", "image_path"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Cache manifest must contain columns {sorted(required)}")
    return rows


def validate_assets(
    data_root: Path,
    cache_manifest: Path,
    cache_dir: Path,
    final_manifest: Path,
    expected_active: int,
    expected_final: int,
) -> dict:
    rows = load_cache_manifest(cache_manifest)
    tokens = [row["token"] for row in rows]
    if len(rows) != expected_active or len(set(tokens)) != expected_active:
        raise ValueError(
            f"Active manifest coverage mismatch: rows={len(rows)} unique_tokens={len(set(tokens))} "
            f"expected={expected_active}"
        )

    missing_images = []
    for row in rows:
        image = data_root / row["image_path"]
        try:
            with image.open("rb") as handle:
                handle.read(1)
        except OSError:
            missing_images.append(str(image))
    if missing_images:
        raise ValueError(f"Unreadable active images: {len(missing_images)}")

    cache_files = list(cache_dir.rglob("metric_cache.pkl"))
    cache_tokens = [path.parent.name for path in cache_files]
    if len(cache_files) != expected_active or set(cache_tokens) != set(tokens):
        raise ValueError(
            f"Metric cache coverage mismatch: files={len(cache_files)} unique_tokens={len(set(cache_tokens))} "
            f"missing={len(set(tokens) - set(cache_tokens))} unexpected={len(set(cache_tokens) - set(tokens))}"
        )

    final_tokens = [line.strip() for line in final_manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(final_tokens) != expected_final or len(set(final_tokens)) != expected_final:
        raise ValueError(
            f"Final reserve coverage mismatch: rows={len(final_tokens)} unique_tokens={len(set(final_tokens))} "
            f"expected={expected_final}"
        )
    if set(final_tokens) & set(tokens):
        raise ValueError("Final reserve overlaps the active image/cache set")

    return {
        "active_tokens": expected_active,
        "readable_images": expected_active,
        "metric_caches": expected_active,
        "final_reserve_tokens": expected_final,
        "final_reserve_state": "manifest_only",
    }


def model_hashes(model_dir: Path) -> dict[str, str]:
    weights = sorted(model_dir.glob("model-*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"Stage-2 model weights missing from {model_dir}")
    paths = weights
    paths += [model_dir / "config.json", model_dir / "model.safetensors.index.json"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage-2 model files missing: {missing}")
    return {path.name: sha256(path) for path in paths}


def refresh_hash_manifest(dataset_dir: Path, manifest_dir: Path) -> Path:
    output = manifest_dir / "sha256sum.txt"
    paths = sorted(
        path
        for root in (dataset_dir, manifest_dir)
        for path in root.rglob("*")
        if path.is_file() and path != output and path.name != "V2_DATA_FROZEN"
    )
    base = manifest_dir.parent.parent
    output.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(base)}\n" for path in paths),
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--dev-parquet", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--stage2-model", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rebind-source", action="store_true")
    parser.add_argument("--expected-active", type=int, default=10000)
    parser.add_argument("--expected-final", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    marker = args.dataset_dir / "V2_DATA_FROZEN"
    if marker.exists() and not args.rebind_source:
        raise SystemExit(f"Dataset is already frozen: {marker}")
    if args.rebind_source and not marker.exists():
        raise SystemExit(f"Source rebind requires an existing freeze marker: {marker}")
    if len(args.source_commit) != 40 or any(character not in "0123456789abcdef" for character in args.source_commit):
        raise ValueError("source-commit must be a full lowercase Git SHA")
    for path in (
        args.data_root,
        args.dataset_dir,
        args.manifest_dir,
        args.train_parquet,
        args.dev_parquet,
        args.cache_manifest,
        args.cache_dir,
        args.final_manifest,
        args.stage2_model,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    assets = validate_assets(
        args.data_root,
        args.cache_manifest,
        args.cache_dir,
        args.final_manifest,
        args.expected_active,
        args.expected_final,
    )
    input_hashes = {
        "train_parquet": sha256(args.train_parquet),
        "dev_parquet": sha256(args.dev_parquet),
        "cache_manifest": sha256(args.cache_manifest),
        "final_manifest": sha256(args.final_manifest),
    }
    stage2_hashes = model_hashes(args.stage2_model)

    card_path = args.dataset_dir / "dataset_card.json"
    acceptance_path = args.manifest_dir / "acceptance_report.json"
    previous_marker = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if previous_marker is not None:
        shutil.copy2(marker, args.output.parent / "previous_V2_DATA_FROZEN.json")
        shutil.copy2(card_path, args.output.parent / "previous_dataset_card.json")
        shutil.copy2(acceptance_path, args.output.parent / "previous_acceptance_report.json")
        shutil.copy2(args.manifest_dir / "sha256sum.txt", args.output.parent / "previous_sha256sum.txt")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    card["artifacts"]["image_stage"] = {"state": "complete", "files": assets["readable_images"]}
    card["artifacts"]["metric_cache_stage"] = {"state": "complete", "files": assets["metric_caches"]}
    card["artifacts"]["final_reserve"] = "manifest_only"
    card["data_freeze"] = {"source_commit": args.source_commit, **assets}
    acceptance["image_stage_deferred"] = False
    acceptance["cache_stage_deferred"] = False
    acceptance["gates"].update(
        {
            "active_images_complete_and_readable": True,
            "active_metric_cache_exact": True,
            "final_reserve_manifest_only": True,
        }
    )
    acceptance["all_gates_passed"] = all(acceptance["gates"].values())
    acceptance["data_freeze"] = {"source_commit": args.source_commit, **assets}
    atomic_json(card_path, card)
    atomic_json(acceptance_path, acceptance)
    hash_manifest = refresh_hash_manifest(args.dataset_dir, args.manifest_dir)

    usage = shutil.disk_usage(args.data_root)
    report = {
        "id": "V2-D0-SOURCE-REBIND" if args.rebind_source else "V2-D0",
        "status": "COMPLETE",
        "source_commit": args.source_commit,
        "previous_source_commit": previous_marker.get("source_commit") if previous_marker else None,
        "assets": assets,
        "input_hashes": input_hashes,
        "stage2_model_hashes": stage2_hashes,
        "dataset_card_sha256": sha256(card_path),
        "acceptance_report_sha256": sha256(acceptance_path),
        "hash_manifest_sha256": sha256(hash_manifest),
        "disk_free_bytes": usage.free,
    }
    atomic_json(args.output, report)
    atomic_json(
        marker,
        {
            "dataset_version": card["dataset_version"],
            "source_commit": args.source_commit,
            "freeze_report": str(args.output),
            "freeze_report_sha256": sha256(args.output),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
