from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from projects.dataset_v3.data_prep import stable_key
from projects.dataset_v3.inventory import sha256_file
from projects.dataset_v3.v4_experiment_closure import (
    INTENT_QUOTAS,
    build_train_selection,
    exclusive_family,
    read_csv,
    revised_train_tiers,
    write_csv,
)


RATIO_FAMILY_QUOTAS = {
    "40": {"proximity": 800, "construction": 600, "signal": 600},
    "50": {"proximity": 1000, "construction": 500, "signal": 500},
    "60": {"proximity": 1200, "construction": 400, "signal": 400},
}
TOTAL_SCENES = 2000
LOG_CAP = 4
MATERIAL_RISK_GAP = 0.20


def read_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != TOTAL_SCENES or len(set(tokens)) != TOTAL_SCENES:
        raise ValueError(f"Manifest must contain {TOTAL_SCENES} unique tokens: {path}")
    return tokens


def candidate_rows(
    scene_rows: list[dict[str, str]], tier_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    scenes = {row["token"]: row for row in scene_rows}
    tiers = {row["token"]: row for row in tier_rows}
    if len(scenes) != len(scene_rows) or len(tiers) != len(tier_rows) or set(scenes) != set(tiers):
        raise ValueError("Train scene and tier labels must have identical unique token coverage")
    classified = {}
    candidates = []
    for token, source in scenes.items():
        family = exclusive_family(tiers[token])
        classified[token] = {
            "token": token,
            "log_name": source["log_name"],
            "intent": source["intent"],
            "exclusive_family": family,
        }
        if family != "control":
            candidates.append(classified[token])
    return candidates, classified


def maximize_primary_risk(
    rows: list[dict[str, str]],
    *,
    total: int = TOTAL_SCENES,
    intent_quotas: dict[str, int] = INTENT_QUOTAS,
    log_cap: int = LOG_CAP,
    seed: int = 20260831,
) -> list[dict[str, str]]:
    if sum(intent_quotas.values()) != total:
        raise ValueError("Intent quotas must sum to the requested total")
    ordered = sorted(rows, key=lambda row: stable_key(seed, "v4-risk-max-order", row["token"]))
    constraints: list[tuple[list[int], float, float]] = [
        (list(range(len(ordered))), total, total)
    ]
    for intent, quota in intent_quotas.items():
        constraints.append(
            ([index for index, row in enumerate(ordered) if row["intent"] == intent], quota, quota)
        )
    by_log: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        by_log[row["log_name"]].append(index)
    constraints.extend((indices, -math.inf, log_cap) for indices in by_log.values())

    matrix_row = []
    matrix_column = []
    lower = []
    upper = []
    for row_index, (indices, minimum, maximum) in enumerate(constraints):
        matrix_row.extend([row_index] * len(indices))
        matrix_column.extend(indices)
        lower.append(minimum)
        upper.append(maximum)
    matrix = coo_matrix(
        (np.ones(len(matrix_row)), (matrix_row, matrix_column)),
        shape=(len(constraints), len(ordered)),
    ).tocsr()
    tie_break = np.asarray(
        [int(stable_key(seed, "v4-risk-max-cost", row["token"])[:13], 16) / 16**13 for row in ordered]
    )
    primary_risk = np.asarray([row["exclusive_family"] == "proximity" for row in ordered])
    result = milp(
        c=-primary_risk.astype(float) + tie_break * 1e-9,
        integrality=np.ones(len(ordered)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise ValueError(f"Maximum-risk MILP did not reach an exact optimum: {result.message}")
    selected = [row for row, value in zip(ordered, result.x) if value > 0.5]
    if len(selected) != total:
        raise ValueError(f"Maximum-risk MILP returned {len(selected)} rows, expected {total}")
    return sorted(selected, key=lambda row: stable_key(seed, "v4-risk-max-output", row["token"]))


def composition(
    tokens: list[str], classified: dict[str, dict[str, str]], *, name: str
) -> dict[str, Any]:
    unknown = set(tokens) - set(classified)
    if unknown:
        raise ValueError(f"{name} contains {len(unknown)} tokens outside Train labels")
    rows = [classified[token] for token in tokens]
    families = Counter(row["exclusive_family"] for row in rows)
    logs = Counter(row["log_name"] for row in rows)
    intents = Counter(row["intent"] for row in rows)
    result: dict[str, Any] = {
        "dataset": name,
        "scenes": len(rows),
        "unique_logs": len(logs),
        "max_per_log": max(logs.values()),
        "straight": intents["straight"],
        "left": intents["left"],
        "right": intents["right"],
    }
    for family in ("proximity", "construction", "signal", "control"):
        result[f"{family}_count"] = families[family]
        result[f"{family}_rate"] = families[family] / len(rows)
    return result


def audit(
    scene_rows: list[dict[str, str]], random_tokens: list[str], *, seed: int
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    tier_rows = revised_train_tiers(scene_rows)
    candidates, classified = candidate_rows(scene_rows, tier_rows)
    selected_by_ratio: dict[str, list[dict[str, str]]] = {}
    ratio_trials: dict[str, Any] = {}
    for ratio, quotas in RATIO_FAMILY_QUOTAS.items():
        try:
            _, selected, selection_report = build_train_selection(
                scene_rows,
                tier_rows,
                family_quotas=quotas,
                intent_quotas=INTENT_QUOTAS,
                cap_candidates=(LOG_CAP,),
                seed=seed + int(ratio),
            )
        except ValueError as error:
            ratio_trials[ratio] = {
                "status": "EXACT_INFEASIBLE",
                "family_quotas": quotas,
                "reason": str(error),
            }
            continue
        selected_by_ratio[ratio] = selected
        ratio_trials[ratio] = {
            "status": "EXACT_FEASIBLE",
            "family_quotas": quotas,
            "selected_unique_logs": selection_report["selected_unique_logs"],
            "selected_max_per_log": selection_report["selected_max_per_log"],
            "selected_intent_counts": selection_report["selected_intent_counts"],
        }

    maximum = maximize_primary_risk(candidates, seed=seed)
    selected_by_ratio["max"] = maximum
    maximum_composition = composition(
        [row["token"] for row in maximum], classified, name="Maximum-risk 2K"
    )

    random_composition = composition(random_tokens, classified, name="Random 2K")
    risk50 = selected_by_ratio.get("50")
    comparison_rows = [random_composition]
    if risk50 is not None:
        risk50_tokens = [row["token"] for row in risk50]
        risk50_composition = composition(risk50_tokens, classified, name="New Risk50 2K")
        comparison_rows.append(risk50_composition)
        risk_gap = risk50_composition["proximity_rate"] - random_composition["proximity_rate"]
        checks = {
            "risk50_exact_feasible": True,
            "intent_quota_exact": all(
                risk50_composition[intent] == quota for intent, quota in INTENT_QUOTAS.items()
            ),
            "log_cap_at_most_4": risk50_composition["max_per_log"] <= LOG_CAP,
            "proximity_rate_exactly_50pct": risk50_composition["proximity_rate"] == 0.5,
            "random_to_risk50_gap_at_least_20pp": risk_gap >= MATERIAL_RISK_GAP,
        }
        overlap = len(set(random_tokens) & set(risk50_tokens))
    else:
        risk50_tokens = []
        risk_gap = None
        overlap = 0
        checks = {
            "risk50_exact_feasible": False,
            "intent_quota_exact": False,
            "log_cap_at_most_4": False,
            "proximity_rate_exactly_50pct": False,
            "random_to_risk50_gap_at_least_20pp": False,
        }
    gate_pass = all(checks.values())
    report = {
        "status": "FROZEN_RISK50_CPU_GATE_PASS" if gate_pass else "RISK50_CPU_GATE_FAILED",
        "semantic_definition": {
            "primary_risk": "current matching-front visible vehicle<=5m or VRU<=10m",
            "exclusive_priority": ["proximity", "construction", "signal", "control"],
            "diversity_split": "remaining non-proximity quota split equally between construction and signal",
        },
        "constraints": {
            "total_scenes": TOTAL_SCENES,
            "intent_quotas": INTENT_QUOTAS,
            "max_per_log": LOG_CAP,
            "seed": seed,
        },
        "candidate_capacity": {
            "scenes": len(candidates),
            "unique_logs": len({row["log_name"] for row in candidates}),
            "family_counts": dict(sorted(Counter(row["exclusive_family"] for row in candidates).items())),
        },
        "ratio_trials": ratio_trials,
        "maximum_primary_risk": maximum_composition,
        "random_vs_risk50": {
            "rows": comparison_rows,
            "proximity_rate_gap": risk_gap,
            "material_gap_threshold": MATERIAL_RISK_GAP,
            "token_overlap": overlap,
            "token_jaccard": overlap / (TOTAL_SCENES * 2 - overlap) if risk50_tokens else None,
        },
        "freeze_gate": {"passed": gate_pass, "checks": checks},
        "decision": {
            "frozen_manifest": "frozen_current_visible_risk50_2000.txt" if gate_pass else None,
            "gpu_training_authorized": False,
            "next_action": (
                "prepare a matched Random-versus-Risk50 GPU launch specification"
                if gate_pass
                else "do not launch GPU; revise the training-pool composition"
            ),
        },
        "dev_accessed": False,
        "final_accessed": False,
    }
    return report, selected_by_ratio, comparison_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(f"{row['token']}\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-scene-labels", type=Path, required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite V4 risk-ratio output: {args.output_dir}")

    scene_rows = read_csv(args.train_scene_labels)
    if len(scene_rows) != 8000:
        raise ValueError("V4 Train scene labels must contain 8,000 rows")
    random_tokens = read_manifest(args.random_manifest)
    report, selected_by_ratio, comparison_rows = audit(scene_rows, random_tokens, seed=args.seed)
    report["input_sha256"] = {
        "train_scene_labels": sha256_file(args.train_scene_labels),
        "random_manifest": sha256_file(args.random_manifest),
    }

    args.output_dir.mkdir(parents=True)
    for ratio in ("40", "50", "60"):
        if ratio in selected_by_ratio:
            write_manifest(args.output_dir / f"candidate_risk_ratio_{ratio}_2000.txt", selected_by_ratio[ratio])
    write_manifest(args.output_dir / "capacity_max_primary_risk_2000.txt", selected_by_ratio["max"])
    if report["freeze_gate"]["passed"]:
        write_manifest(
            args.output_dir / "frozen_current_visible_risk50_2000.txt",
            selected_by_ratio["50"],
        )
    write_csv(args.output_dir / "random_vs_frozen_risk_composition.csv", comparison_rows)
    (args.output_dir / "v4_risk_ratio_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
