import json
from pathlib import Path

from projects.safe_grpo.export_training_evidence import export_samples


def test_export_samples_uses_only_train_rows_and_detects_raw_response(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    rows = [
        {
            "token": "train",
            "evidence_phase": "train",
            "training_reward": 0.5,
            "response_length": 10,
            "parsed_ok": True,
            "raw_response": "trajectory",
        },
        {
            "token": "monitor",
            "evidence_phase": "train_monitor",
            "training_reward": 1.0,
            "response_length": 20,
            "parsed_ok": True,
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "samples.jsonl"

    report = export_samples(source, output)

    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["source_rollouts"] == 1
    assert report["raw_response_available"] is True
    assert {row["token"] for row in exported} == {"train"}
