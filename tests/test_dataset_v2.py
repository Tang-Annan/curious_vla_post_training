import csv
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from projects.dataset_v2.build_dataset_v2 import (
    HORIZON_4,
    HORIZON_5,
    largest_remainder,
    log_from_image,
    normalize_intent,
    v2_image_path,
)
from projects.dataset_v2.freeze_dataset_v2 import load_cache_manifest, validate_assets
from projects.dataset_v2.experiment_pipeline import adas_eligible, analyze_s0, build_manifests, fals_score, spearman


def test_intent_normalization_and_image_namespace() -> None:
    prompt = "Current high-level intent (string): turn left\n"
    assert normalize_intent(prompt) == "left"
    source = "navsim/trainval_sensor_blobs/trainval/log-a/CAM_F0/frame.jpg"
    assert log_from_image(source) == "log-a"
    assert v2_image_path(source, "dataset_v2_20260825") == "dataset_v2_20260825/sensor_blobs/trainval/log-a/CAM_F0/frame.jpg"


def test_largest_remainder_is_exact() -> None:
    quota = largest_remainder({"straight": 65, "left": 26, "right": 9}, 1000)
    assert quota == {"straight": 650, "left": 260, "right": 90}
    assert sum(quota.values()) == 1000


def test_parquet_schema_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    table = pa.table(
        {
            "images": [["navsim/trainval_sensor_blobs/trainval/log-a/CAM_F0/frame.jpg"]],
            "problem": [f"Current high-level intent (string): go straight\n{HORIZON_5}"],
            "answer": [{"gt": [], "token": "token-a"}],
        }
    )
    pq.write_table(table, path)
    result = pq.read_table(path).to_pylist()[0]
    assert result["answer"]["token"] == "token-a"
    assert HORIZON_4 not in result["problem"]


def test_validate_completed_assets(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    manifest = tmp_path / "cache.csv"
    final_manifest = tmp_path / "final.txt"
    rows = []
    for token in ("active-a", "active-b"):
        image_path = Path("dataset_v2") / "sensor_blobs" / token / "CAM_F0" / "frame.jpg"
        image = data_root / image_path
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        cache = cache_dir / token / "metric_cache.pkl"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"cache")
        rows.append({"token": token, "log_name": token, "split": "candidate", "image_path": image_path.as_posix()})
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", "log_name", "split", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
    final_manifest.write_text("final-a\n", encoding="utf-8")

    assert len(load_cache_manifest(manifest)) == 2
    report = validate_assets(data_root, manifest, cache_dir, final_manifest, 2, 1)
    assert report == {
        "active_tokens": 2,
        "readable_images": 2,
        "metric_caches": 2,
        "final_reserve_tokens": 1,
        "final_reserve_state": "manifest_only",
    }


def test_dataset_v2_launcher_has_no_legacy_fallbacks() -> None:
    launcher = (Path(__file__).parents[1] / "scripts" / "run_dataset_v2_experiment.sh").read_text(encoding="utf-8")
    assert "metric_cache_released_5656" not in launcher
    assert "566" not in launcher
    for required in ("--train-parquet", "--dev-parquet", "--cache-manifest", "--cache-dir", "--experiment-root"):
        assert required in launcher
    assert '^(d0|rollout|train)$' in launcher
    assert 'TENSORBOARD_DIR="$RUN_DIR/tensorboard"' in launcher


def test_selector_formulas_match_preregistered_g4_rules() -> None:
    eligible = {
        "pdms_mean": 0.5,
        "pdms_std": 1 / 3**0.5,
        "pdms_min": 0.0,
        "pdms_max": 1.0,
        "scaled_mean": 0.4,
        "scaled_std": 0.2,
        "scaled_max": 0.8,
    }
    assert adas_eligible(eligible)
    assert fals_score(eligible) == (1.0 - 0.4) * (0.8 - 0.4)
    assert not adas_eligible({**eligible, "pdms_std": 0.0})
    assert spearman({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 2.0, "b": 4.0, "c": 6.0}) == 1.0


def test_s0_analyzes_four_shared_g4_blocks(tmp_path: Path) -> None:
    tokens = ["a", "b", "c", "d"]
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    blocks = []
    for block_index in range(4):
        block = tmp_path / f"block_{block_index}.jsonl"
        rows = []
        for token_index, token in enumerate(tokens):
            scaled = [0.1, 0.2, 0.3, 0.9 - token_index * 0.1]
            rows.extend({"token": token, "pdms": value, "pdms_scaled": scaled[index]} for index, value in enumerate([0.0, 0.0, 1.0, 1.0]))
        block.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        blocks.append(block)
    output = tmp_path / "report.json"
    report = analyze_s0(Namespace(manifest=manifest, block=blocks, output=output))
    assert report["blocks"] == 4
    assert report["gates"]["adas_passed"]
    assert report["gates"]["fals_passed"]


def test_m0_skips_adas_closed_by_s0(tmp_path: Path) -> None:
    intents = ["straight"] * 634 + ["left"] * 251 + ["right"] * 115
    tokens = [f"token-{index:04d}" for index in range(1000)]
    candidate = tmp_path / "candidate.txt"
    random_manifest = tmp_path / "random.txt"
    candidate.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    random_manifest.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    master = tmp_path / "master.csv"
    with master.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", "intent", "log_name"])
        writer.writeheader()
        writer.writerows(
            {"token": token, "intent": intent, "log_name": f"log-{index:04d}"}
            for index, (token, intent) in enumerate(zip(tokens, intents))
        )
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        "".join(
            json.dumps({"token": token, "pdms": value, "pdms_scaled": value}) + "\n"
            for token in tokens
            for value in (0.2, 0.4, 0.6, 0.8)
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "manifests"
    report = build_manifests(
        Namespace(
            master_index=master,
            candidate_manifest=candidate,
            rollouts=rollouts,
            random_manifest=random_manifest,
            output_dir=output_dir,
            report=tmp_path / "report.json",
            seed=20260825,
            skip_adas=True,
        )
    )
    assert report["decision"] == "freeze_manifests"
    assert not report["extension_required"]
    assert report["adas_status"] == "skipped_by_s0"
    assert not (output_dir / "adas_1k.txt").exists()
    assert (output_dir / "fals_1k.txt").is_file()


def test_pipeline_cli_runs_outside_repository_cwd(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "projects" / "dataset_v2" / "experiment_pipeline.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
