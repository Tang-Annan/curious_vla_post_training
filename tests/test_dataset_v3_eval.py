import csv
import json
from pathlib import Path
from types import SimpleNamespace

from projects.dataset_v3.eval_pipeline import summarize_eval


def test_dev_eval_requires_exact_unseen_coverage_and_summarizes_tail(tmp_path: Path) -> None:
    natural = ["n0", "n1"]
    tail = ["t0", "t1"]
    (tmp_path / "natural.txt").write_text("\n".join(natural) + "\n", encoding="utf-8")
    (tmp_path / "tail.txt").write_text("\n".join(tail) + "\n", encoding="utf-8")
    with (tmp_path / "master.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", "log_name", "source_universe", "split", "intent"])
        writer.writeheader()
        for token in natural + tail:
            writer.writerow(
                {
                    "token": token,
                    "log_name": f"log-{token}",
                    "source_universe": "sft_unseen",
                    "split": "dev_natural" if token in natural else "dev_tail",
                    "intent": "straight",
                }
            )
    protocol = {
        "status": "M0_FROZEN",
        "tail_evaluation": {"dev_counts": {"natural": 2, "tail": 2}},
        "final_accessed": False,
    }
    (tmp_path / "m0.json").write_text(json.dumps(protocol), encoding="utf-8")
    metrics = {
        "n0": (1, 1, 1, 1, 1, 0.8, 0.6),
        "n1": (1, 1, 1, 1, 1, 0.6, 0.4),
        "t0": (1, 1, 1, 1, 1, 0.9, 0.7),
        "t1": (1, 1, 0.5, 0, 1, 0.2, 0.1),
    }
    fields = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "history_comfort",
        "pdms",
        "pdms_scaled",
    )
    rows = []
    for token in natural + tail:
        rows.append(
            {
                "token": token,
                "evidence_phase": "dev_eval",
                "raw_response": "{}",
                "poses": [[0, 0, 0]] * 8,
                "parsed_ok": True,
                "response_length": 100,
                **dict(zip(fields, metrics[token])),
            }
        )
    (tmp_path / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "results"
    summarize_eval(
        SimpleNamespace(
            rollouts=tmp_path / "rollouts.jsonl",
            master_index=tmp_path / "master.csv",
            natural_manifest=tmp_path / "natural.txt",
            tail_manifest=tmp_path / "tail.txt",
            m0_protocol=tmp_path / "m0.json",
            run_id="e0",
            model_id="V3-E0-SFT",
            evidence_phase="dev_eval",
            evaluation_seed=20260827,
            output_dir=output,
        )
    )
    report = json.loads((output / "eval_summary.json").read_text(encoding="utf-8"))
    assert report["combined"]["scenes"] == 4
    assert report["tail"]["strict_clear_rate"] == 0.5
    assert report["tail"]["tier_rates"]["L2"] == 0.5
    assert report["tail"]["pdms_cvar20"] == 0.2
    assert report["final_accessed"] is False


def test_adas_runner_propagates_evaluation_evidence_phase() -> None:
    root = Path(__file__).parents[1]
    source = (root / "EasyR1/verl/trainer/main_adas.py").read_text(encoding="utf-8")
    checkpoint = (root / "EasyR1/verl/utils/checkpoint/fsdp_checkpoint_manager.py").read_text(encoding="utf-8")
    assert 'os.environ.get("EVIDENCE_PHASE")' in source
    assert 'batch.non_tensor_batch["evidence_phase"]' in source
    assert "actor_wg.load_checkpoint(actor_path, load_model_only=True)" in source
    assert "set_model_state_dict(self.model, model_state_dict" in checkpoint
