from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from projects.dataset_v3.s1_pipeline import sha256_file


SEMANTIC_STATUS = "COMPLETE_NON_SAFETY_REMAINDER_OF_RECORDED_PDMS_COMPONENTS"


def freeze_protocol(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "CANDIDATE_GEOMETRY_ONLY_NOT_REWARD_FREEZE":
        raise ValueError("R0 candidate geometry report has an unexpected status")
    if report.get("task_quality_audit", {}).get("semantic_status") != SEMANTIC_STATUS:
        raise ValueError("Q_task semantic completeness gate failed")

    for selector in ("random", "tailmix"):
        raw = report["cells"][f"{selector}_raw_pdms"]
        task = report["cells"][f"{selector}_cdt_task"]
        if task["cross_tier_inversions"] or task["cross_tier_ties"]:
            raise ValueError(f"{selector} cross-tier ordering gate failed")
        if task["within_tier_quality_inversions_or_ties"]:
            raise ValueError(f"{selector} within-tier ordering gate failed")
        if task["effective_group_rate"] < raw["effective_group_rate"]:
            raise ValueError(f"{selector} EffectiveGroupRate gate failed")

    return {
        "status": "FROZEN",
        "reward_id": "R_TASK_CDT_V3",
        "formula": "(2*tier_value + Q_task)/7",
        "task_quality": "Q_task=(5*ego_progress+2*history_comfort)/7",
        "tier_values": {"L0": 0, "L1": 1, "L2": 2, "L3": 3},
        "intervals": {"L0": [0, 1 / 7], "L1": [2 / 7, 3 / 7], "L2": [4 / 7, 5 / 7], "L3": [6 / 7, 1]},
        "invalid_policy": "parse-invalid is outside L0-L3 and receives technical zero reward",
        "production_module": "EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py",
        "production_function": "compute_score_cdt_task",
        "raw_control_function": "compute_score_raw_pdms",
        "candidate_report_summary": {
            selector: {
                formula: {
                    key: report["cells"][f"{selector}_{formula}"][key]
                    for key in (
                        "effective_group_rate",
                        "exact_zero_rate",
                        "low_nonzero_rate",
                        "cross_tier_inversions",
                        "cross_tier_ties",
                        "within_tier_quality_inversions_or_ties",
                    )
                }
                for formula in ("raw_pdms", "cdt_task")
            }
            for selector in ("random", "tailmix")
        },
        "dev_accessed": False,
        "final_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.geometry_report.read_text(encoding="utf-8"))
    protocol = freeze_protocol(report)
    protocol["geometry_report_sha256"] = sha256_file(args.geometry_report)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
