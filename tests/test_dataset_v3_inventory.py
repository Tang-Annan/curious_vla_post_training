import argparse
import csv
import io
import os
import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from projects.dataset_v3.inventory import build_inventory, eligible_centers, load_navsim_log


def frames(log_name: str, token: str, *, has_route: bool = True) -> list[dict]:
    result = []
    for index in range(14):
        result.append(
            {
                "token": token if index == 3 else f"frame-{index}",
                "log_name": log_name,
                "roadblock_ids": ["route"] if has_route else [],
                "annotations": {"boxes": [index]},
                "ego_status": {"velocity": [0.0, 0.0]},
            }
        )
    return result


def test_restricted_unpickler_rejects_arbitrary_globals() -> None:
    class Unsafe:
        def __reduce__(self):
            return os.system, ("echo unsafe",)

    with pytest.raises(pickle.UnpicklingError, match="Disallowed pickle global"):
        load_navsim_log(io.BytesIO(pickle.dumps(Unsafe())))


def test_eligible_centers_require_complete_window_and_route() -> None:
    candidates = frames("log-a", "token-a") + frames("log-a", "token-b", has_route=False) + [{}]

    assert [center["token"] for center in eligible_centers(candidates)] == ["token-a"]


def test_build_inventory_separates_sft_logs(tmp_path: Path) -> None:
    sft_parquet = tmp_path / "sft.parquet"
    pq.write_table(
        pa.table({"answer": pa.array([{"gt": [], "token": "token-a"}])}),
        sft_parquet,
    )
    master_index = tmp_path / "master.csv"
    with master_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", "log_name", "intent", "split"])
        writer.writeheader()
        writer.writerow({"token": "token-a", "log_name": "log-a", "intent": "straight", "split": "source"})

    navsim_logs = tmp_path / "logs"
    navsim_logs.mkdir()
    (navsim_logs / "log-a.pkl").write_bytes(pickle.dumps(frames("log-a", "token-a")))
    (navsim_logs / "log-b.pkl").write_bytes(pickle.dumps(frames("log-b", "token-b")))
    sensor_root = tmp_path / "sensors"
    metric_cache_root = tmp_path / "metric_cache"
    sensor_root.mkdir()
    metric_cache_root.mkdir()
    model_hash_record = tmp_path / "model_sha256.txt"
    model_hash_record.write_text(f"{'0' * 64}  /models/model.safetensors\n", encoding="utf-8")

    report = build_inventory(
        argparse.Namespace(
            sft_parquet=sft_parquet,
            master_index=master_index,
            navsim_logs=navsim_logs,
            sensor_root=sensor_root,
            metric_cache_root=metric_cache_root,
            model_hash_record=model_hash_record,
            source_commit="test-commit",
            output_dir=tmp_path / "output",
        )
    )

    assert report["sft_provenance"]["master_unique_logs"] == 1
    assert report["raw_logs"]["sft_unseen_logs"] == 1
    assert report["sft_unseen_capacity"]["unique_tokens"] == 1
    assert report["sft_unseen_capacity"]["sft_token_overlap"] == 0
    assert (tmp_path / "output/COMPLETE").exists()
