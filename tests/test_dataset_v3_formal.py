import csv
import json
from pathlib import Path
from types import SimpleNamespace

from projects.dataset_v3.formal_pipeline import compare_dev


def _write_metrics(path: Path, strict_tail: tuple[bool, bool], natural_shift: float = 0.0) -> None:
    fields = [
        "token",
        "log_name",
        "split",
        "intent",
        "parsed_ok",
        "response_length",
        "clipped",
        "tier",
        "strict_clear",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "history_comfort",
        "pdms",
        "pdms_scaled",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, token in enumerate(("n0", "n1", "t0", "t1")):
            tail = token.startswith("t")
            clear = strict_tail[index - 2] if tail else True
            writer.writerow(
                {
                    "token": token,
                    "log_name": f"log-{token}",
                    "split": "dev_tail" if tail else "dev_natural",
                    "intent": "straight",
                    "parsed_ok": True,
                    "response_length": 100,
                    "clipped": False,
                    "tier": "L3" if clear else "L2",
                    "strict_clear": clear,
                    "no_at_fault_collisions": 1,
                    "drivable_area_compliance": 1,
                    "ego_progress": 1,
                    "time_to_collision_within_bound": 1 if clear else 0,
                    "history_comfort": 1,
                    "pdms": 0.8 + (natural_shift if not tail else 0),
                    "pdms_scaled": 0.7 + (natural_shift if not tail else 0),
                }
            )


def test_paired_dev_comparison_applies_frozen_discovery_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_metrics(baseline, (True, False))
    _write_metrics(candidate, (True, True))
    protocol = {
        "tail_evaluation": {"dev_counts": {"natural": 2, "tail": 2}},
        "evaluation": {"bootstrap": {"resamples": 2000}},
        "promotion": {
            "discovery": {
                "tail_primary_delta_min": 0.01,
                "tail_primary_ci_upper_gt": 0.0,
                "natural_primary_point_delta_min": -0.01,
                "natural_primary_ci_lower_severe_harm_floor": -0.03,
                "safety_component_point_drop_max": 0.005,
            }
        },
    }
    m0 = tmp_path / "m0.json"
    m0.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "comparison.json"
    compare_dev(
        SimpleNamespace(
            m0_protocol=m0,
            baseline=baseline,
            candidate=candidate,
            contrast="candidate-baseline",
            seed=20260827,
            bootstrap_seed=20260827,
            output=output,
        )
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["tail"]["strict_clear"]["point_delta"] == 0.5
    assert report["natural"]["pdms_scaled"]["point_delta"] == 0.0
    assert report["status"] == "PROMOTE_TO_CONFIRMATION"
    assert report["final_accessed"] is False
