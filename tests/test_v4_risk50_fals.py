import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import projects.dataset_v3.v4_risk50_fals as selector


def rollout(
    token: str,
    pdms: float,
    *,
    safe: bool,
    parsed_ok: bool = True,
) -> dict:
    return {
        "token": token,
        "pdms": pdms,
        "parsed_ok": parsed_ok,
        "no_at_fault_collisions": float(safe),
        "drivable_area_compliance": 1.0,
        "time_to_collision_within_bound": float(safe),
    }


def feature(
    token: str,
    *,
    family: str,
    intent: str,
    log_name: str,
    mixed: bool,
    fals: float,
    headroom: float,
) -> dict:
    return {
        "token": token,
        "exclusive_family": family,
        "intent": intent,
        "log_name": log_name,
        "strict_clear_mixed": int(mixed),
        "fals": fals,
        "headroom": headroom,
        "selector_role": "direct_safety_contrast" if mixed else "fals_learnable",
    }


def test_raw_pdms_features_partition_selector_roles() -> None:
    tokens = ("A", "B", "CS", "CU", "D", "P")
    labels = [
        {
            "token": token,
            "log_name": f"log-{token}",
            "intent": "straight",
            "exclusive_family": "proximity",
        }
        for token in tokens
    ]
    rows = [
        *[rollout("A", pdms, safe=index >= 2) for index, pdms in enumerate((0.1, 0.2, 0.8, 0.9))],
        *[rollout("B", 1.0, safe=True) for _ in range(4)],
        *[rollout("CS", pdms, safe=True) for pdms in (0.4, 0.5, 0.8, 0.9)],
        *[rollout("CU", pdms, safe=False) for pdms in (0.1, 0.2, 0.3, 0.4)],
        *[rollout("D", 0.0, safe=False) for _ in range(4)],
        rollout("P", 0.7, safe=False, parsed_ok=False),
        *[rollout("P", 0.7, safe=True) for _ in range(3)],
    ]

    features = {row["token"]: row for row in selector.build_features(labels, rows)}

    assert {token: row["semantic_bucket"] for token, row in features.items()} == {
        "A": "A",
        "B": "B",
        "CS": "C-safe",
        "CU": "C-unsafe",
        "D": "D",
        "P": "A",
    }
    assert features["A"]["selector_role"] == "direct_safety_contrast"
    assert features["CS"]["selector_role"] == "fals_learnable"
    assert features["D"]["selector_role"] == "random_anchor"
    assert features["P"]["parse_induced_mixed"] == 1
    assert features["P"]["mean_raw_pdms"] == pytest.approx(0.525)


def test_constrained_greedy_keeps_intent_and_excludes_lowest_ranked_mixed_at_log_cap(
    monkeypatch,
) -> None:
    monkeypatch.setattr(selector, "TOTAL_SCENES", 4)
    monkeypatch.setattr(selector, "FAMILY_QUOTAS", {"proximity": 3, "signal": 1})
    monkeypatch.setattr(selector, "INTENT_QUOTAS", {"straight": 2, "left": 2})
    monkeypatch.setattr(selector, "LOG_CAP", 2)
    rows = [
        feature("a1", family="proximity", intent="straight", log_name="shared", mixed=True, fals=0.9, headroom=0.9),
        feature("a2", family="proximity", intent="left", log_name="shared", mixed=True, fals=0.8, headroom=0.8),
        feature("a3", family="signal", intent="straight", log_name="shared", mixed=True, fals=0.7, headroom=0.7),
        feature("s1", family="signal", intent="left", log_name="signal", mixed=False, fals=0.6, headroom=0.6),
        feature("p1", family="proximity", intent="straight", log_name="prox-1", mixed=False, fals=0.5, headroom=0.5),
        feature("p2", family="proximity", intent="left", log_name="prox-2", mixed=False, fals=0.4, headroom=0.4),
    ]

    trial = selector.select_with_intent_fallback(rows)
    selected = trial["selected"]

    assert trial["intent_trial"]["status"] == "EXACT_FEASIBLE"
    assert Counter(row["exclusive_family"] for row in selected) == {"proximity": 3, "signal": 1}
    assert Counter(row["intent"] for row in selected) == {"straight": 2, "left": 2}
    assert {row["token"] for row in selected} == {"a1", "a2", "s1", "p1"}
    assert trial["mixed_capacity_excluded"] == ["a3"]


def test_selector_falls_back_without_intent_and_materializes_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(selector, "TOTAL_SCENES", 2)
    monkeypatch.setattr(selector, "EXPECTED_CANDIDATES", 3)
    monkeypatch.setattr(selector, "FAMILY_QUOTAS", {"proximity": 1, "signal": 1})
    monkeypatch.setattr(selector, "INTENT_QUOTAS", {"straight": 1, "left": 1})
    monkeypatch.setattr(selector, "LOG_CAP", 2)

    data_root = tmp_path / "data"
    data_root.mkdir()
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "token,log_name,intent,exclusive_family\n"
        "p,log-p,straight,proximity\n"
        "s,log-s,straight,signal\n"
        "x,log-x,straight,proximity\n",
        encoding="utf-8",
    )
    enriched = tmp_path / "enriched.jsonl"
    with enriched.open("w", encoding="utf-8") as handle:
        for token, safe, values in (
            ("p", True, (0.2, 0.4, 0.6, 0.8)),
            ("s", False, (0.1, 0.2, 0.3, 0.4)),
            ("x", False, (0.0, 0.0, 0.0, 0.0)),
        ):
            for value in values:
                handle.write(json.dumps(rollout(token, value, safe=safe)) + "\n")
    risk50 = tmp_path / "risk50.txt"
    random = tmp_path / "random.txt"
    risk50.write_text("p\ns\n", encoding="utf-8")
    random.write_text("p\nx\n", encoding="utf-8")
    for token in ("p", "s", "x"):
        (data_root / f"{token}.jpg").write_bytes(b"image")
    parquet = tmp_path / "screen.parquet"
    pq.write_table(
        pa.table(
            {
                "problem": ["p", "s", "x"],
                "answer": [{"token": token} for token in ("p", "s", "x")],
                "images": [[f"{token}.jpg"] for token in ("p", "s", "x")],
            }
        ),
        parquet,
    )
    output = tmp_path / "results"

    selector.run(
        argparse.Namespace(
            risk_labels=labels,
            screen_enriched=enriched,
            baseline_manifest=risk50,
            random_manifest=random,
            screen_parquet=parquet,
            data_root=data_root,
            output_dir=output,
        )
    )

    report = json.loads((output / "v4_risk50_fals_n1_report.json").read_text(encoding="utf-8"))
    assert report["intent_trial"]["status"] == "INTENT_GREEDY_INFEASIBLE"
    assert report["intent_constraint_used"] is False
    assert report["chosen_dataset"]["summary"]["family_counts"] == {
        "proximity": 1,
        "signal": 1,
    }
    assert report["chosen_dataset"]["parquet"]["rows"] == 2
    assert (output / "risk50_fals_n1_2000.txt").read_text(encoding="utf-8").count("\n") == 2
    assert pq.read_table(output / "risk50_fals_n1_2000.parquet").num_rows == 2
