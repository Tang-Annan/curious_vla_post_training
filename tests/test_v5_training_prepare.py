from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from projects.dataset_v3.formal_pipeline import CELL_METADATA, CELL_REWARD
import projects.dataset_v3.v5_training_prepare as preparation
from projects.dataset_v3.v4_training_prepare import config_differences, validate_rr_contract


def rr_config() -> dict:
    return {
        "data": {
            "train_files": "/rr.parquet@train",
            "val_files": "/monitor.parquet@train",
            "image_dir": "/data",
            "seed": 20260827,
            "rollout_batch_size": 4,
        },
        "worker": {
            "reward": {"reward_function_name": "compute_score_raw_pdms"},
            "rollout": {"seed": 20260827, "n": 4, "temperature": 1.0, "top_p": 1.0},
            "actor": {
                "global_batch_size": 4,
                "ppo_epochs": 1,
                "optim": {"lr": 1e-6, "lr_scheduler_type": "constant"},
                "model": {
                    "lora": {
                        "rank": 8,
                        "alpha": 16,
                        "target_modules": "q_proj,k_proj,v_proj,o_proj",
                        "exclude_modules": ".*visual.*",
                    }
                },
            },
        },
        "algorithm": {"use_kl_loss": True, "kl_penalty": "low_var_kl", "kl_coef": 0.01},
        "trainer": {
            "experiment_name": "v3_rr_random_raw_g4_b4_seed20260827",
            "save_checkpoint_path": "/rr/checkpoints",
            "max_steps": 500,
            "val_steps": [100, 200, 300, 400, 500],
        },
    }


@pytest.mark.parametrize("dataset", ["risk50", "risk50_fals"])
def test_v5_config_only_changes_dataset_and_run_identity(tmp_path: Path, dataset: str) -> None:
    reference = rr_config()
    aligned, differences = preparation.build_aligned_config(
        reference,
        dataset=dataset,
        train_parquet=tmp_path / f"{dataset}.parquet",
        future_run_dir=tmp_path / dataset,
    )

    assert set(differences) == preparation.ALLOWED_CONFIG_DIFFERENCES
    assert aligned["trainer"]["experiment_name"] == preparation.EXPERIMENT_NAMES[dataset]
    assert all(validate_rr_contract(aligned).values())


def test_two_v5_configs_differ_only_by_dataset_and_run_identity(tmp_path: Path) -> None:
    reference = rr_config()
    risk, _ = preparation.build_aligned_config(
        reference,
        dataset="risk50",
        train_parquet=tmp_path / "risk.parquet",
        future_run_dir=tmp_path / "risk",
    )
    fals, _ = preparation.build_aligned_config(
        reference,
        dataset="risk50_fals",
        train_parquet=tmp_path / "fals.parquet",
        future_run_dir=tmp_path / "fals",
    )

    assert set(config_differences(risk, fals)) == preparation.ALLOWED_CONFIG_DIFFERENCES


def test_materialized_dataset_validates_schema_order_and_images(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preparation, "EXPECTED_GROUPS", 2)
    data_root = tmp_path / "data"
    data_root.mkdir()
    for token in ("a", "b"):
        (data_root / f"{token}.jpg").write_bytes(b"image")
    table = pa.table(
        {
            "problem": ["p-a", "p-b"],
            "answer": [{"token": "a"}, {"token": "b"}],
            "images": [["a.jpg"], ["b.jpg"]],
        }
    )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("a\nb\n", encoding="utf-8")
    parquet = tmp_path / "dataset.parquet"
    reference = tmp_path / "reference.parquet"
    pq.write_table(table, parquet)
    pq.write_table(table, reference)

    report = preparation.validate_materialized_dataset(manifest, parquet, reference, data_root)

    assert report["rows"] == 2
    assert report["manifest_order_exact"] is True
    assert report["missing_images"] == 0


def test_materialized_dataset_rejects_manifest_order_mismatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preparation, "EXPECTED_GROUPS", 2)
    data_root = tmp_path / "data"
    data_root.mkdir()
    table = pa.table(
        {
            "problem": ["p-a", "p-b"],
            "answer": [{"token": "a"}, {"token": "b"}],
            "images": [[], []],
        }
    )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("b\na\n", encoding="utf-8")
    parquet = tmp_path / "dataset.parquet"
    pq.write_table(table, parquet)

    with pytest.raises(ValueError, match="answer order"):
        preparation.validate_materialized_dataset(manifest, parquet, parquet, data_root)


def test_v5_formal_cells_are_registered_for_raw_pdms() -> None:
    assert CELL_REWARD["V5-RISK50"] == "compute_score_raw_pdms"
    assert CELL_REWARD["V5-RISK50-FALS"] == "compute_score_raw_pdms"
    assert CELL_METADATA["V5-RISK50"] == {"selector": "Risk50", "reward": "Raw-PDMS"}
    assert CELL_METADATA["V5-RISK50-FALS"] == {
        "selector": "Risk50+FALS",
        "reward": "Raw-PDMS",
    }
