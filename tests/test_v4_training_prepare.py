from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from projects.dataset_v3.formal_pipeline import CELL_METADATA, CELL_REWARD
from projects.dataset_v3.v4_training_prepare import (
    ALLOWED_RR_CONFIG_DIFFERENCES,
    EXPERIMENT_NAME,
    build_aligned_config,
    config_differences,
    materialize_parquet,
    validate_rr_contract,
)


def rr_config() -> dict:
    return {
        "data": {
            "train_files": "/rr/random.parquet@train",
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


def test_aligned_config_only_changes_the_three_data_identity_fields(tmp_path: Path) -> None:
    baseline = rr_config()
    aligned, differences = build_aligned_config(
        baseline,
        train_parquet=tmp_path / "risk50.parquet",
        future_run_dir=tmp_path / "future",
    )

    assert set(differences) == ALLOWED_RR_CONFIG_DIFFERENCES
    assert aligned["trainer"]["experiment_name"] == EXPERIMENT_NAME
    assert validate_rr_contract(aligned) == {
        "seed": True,
        "raw_pdms": True,
        "groups": True,
        "optimizer": True,
        "kl": True,
        "lora": True,
        "budget": True,
        "sampling": True,
    }


def test_config_differences_reports_nested_paths() -> None:
    assert config_differences({"a": {"b": 1}, "c": 2}, {"a": {"b": 3}, "c": 2}) == ["a.b"]


def test_materialized_parquet_matches_rr_schema_and_manifest_order(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for token in ("a", "b", "c"):
        (data_root / f"{token}.jpg").write_bytes(b"image")
    table = pa.table(
        {
            "problem": ["p-a", "p-b", "p-c"],
            "answer": [{"token": "a"}, {"token": "b"}, {"token": "c"}],
            "images": [["a.jpg"], ["b.jpg"], ["c.jpg"]],
        }
    )
    screen = tmp_path / "screen.parquet"
    rr = tmp_path / "rr.parquet"
    pq.write_table(table, screen)
    pq.write_table(table.take(pa.array([0, 1])), rr)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("c\na\n", encoding="utf-8")
    output_manifest = tmp_path / "output.txt"
    output_parquet = tmp_path / "output.parquet"

    import projects.dataset_v3.v4_training_prepare as preparation

    monkeypatch.setattr(preparation, "EXPECTED_GROUPS", 2)
    report = materialize_parquet(
        manifest, screen, rr, output_manifest, output_parquet, data_root
    )

    assert report["rows"] == 2
    assert report["schema_matches_rr"] is True
    assert [row["token"] for row in pq.read_table(output_parquet)["answer"].to_pylist()] == ["c", "a"]


def test_v4_formal_cell_uses_risk50_and_raw_pdms() -> None:
    assert CELL_REWARD["V4-RISK50"] == "compute_score_raw_pdms"
    assert CELL_METADATA["V4-RISK50"] == {"selector": "Risk50", "reward": "Raw-PDMS"}
