from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

METRIC_FIELDS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
    "pdms",
    "pdms_scaled",
)
TIERS = ("L0", "L1", "L2", "L3")
AUDITED_EPS = 1e-6


def _canonical(value: float, allowed: tuple[float, ...], name: str) -> float:
    matches = [candidate for candidate in allowed if abs(value - candidate) <= AUDITED_EPS + 1e-12]
    if len(matches) != 1:
        raise ValueError(f"{name}={value} is outside the audited evaluator values")
    return matches[0]


def _classify_tier(parsed_ok: bool, metrics: dict[str, float]) -> str | None:
    if not parsed_ok:
        return None
    collision = _canonical(metrics["no_at_fault_collisions"], (0.0, 0.5, 1.0), "collision")
    drivable = _canonical(metrics["drivable_area_compliance"], (0.0, 1.0), "drivable")
    ttc = _canonical(metrics["time_to_collision_within_bound"], (0.0, 1.0), "ttc")
    if collision == 0.0:
        return "L0"
    if collision < 1.0 or drivable < 1.0:
        return "L1"
    if ttc < 1.0:
        return "L2"
    return "L3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate token in manifest: {path}")
    return tokens


def _cvar20(values: list[float]) -> float:
    count = max(1, math.ceil(len(values) * 0.2))
    return statistics.fmean(sorted(values)[:count])


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tiers = Counter(row["tier"] for row in rows if row["tier"] is not None)
    return {
        "scenes": len(rows),
        "logs": len({row["log_name"] for row in rows}),
        "parse_rate": statistics.fmean(row["parsed_ok"] for row in rows),
        "clip_rate": statistics.fmean(row["clipped"] for row in rows),
        "invalid_rate": statistics.fmean(row["tier"] is None for row in rows),
        "strict_clear_count": tiers["L3"],
        "strict_clear_rate": tiers["L3"] / len(rows),
        "tier_rates": {tier: tiers[tier] / len(rows) for tier in TIERS},
        "pdms_cvar20": _cvar20([row["pdms"] for row in rows]),
        "metric_means": {
            field: statistics.fmean(row[field] for row in rows) for field in METRIC_FIELDS
        },
        "response_length": {
            "mean": statistics.fmean(row["response_length"] for row in rows),
            "max": max(row["response_length"] for row in rows),
        },
    }


def summarize_eval(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output_dir}")
    protocol = json.loads(args.m0_protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "M0_FROZEN" or protocol.get("final_accessed") is not False:
        raise ValueError("M0 is not frozen with Final unaccessed")

    natural_tokens = _manifest(args.natural_manifest)
    tail_tokens = _manifest(args.tail_manifest)
    expected_counts = protocol["tail_evaluation"]["dev_counts"]
    if len(natural_tokens) != int(expected_counts["natural"]) or len(tail_tokens) != int(expected_counts["tail"]):
        raise ValueError("Dev manifest counts differ from M0")
    if set(natural_tokens) & set(tail_tokens):
        raise ValueError("Natural/Tail Dev manifests overlap")
    expected_tokens = set(natural_tokens) | set(tail_tokens)

    with args.master_index.open(encoding="utf-8-sig", newline="") as handle:
        master = {row["token"]: row for row in csv.DictReader(handle)}
    if any(master[token]["source_universe"] != "sft_unseen" for token in expected_tokens):
        raise ValueError("Dev evaluation escaped the SFT-unseen universe")
    expected_split = {**{token: "dev_natural" for token in natural_tokens}, **{token: "dev_tail" for token in tail_tokens}}
    if any(master[token]["split"] != split for token, split in expected_split.items()):
        raise ValueError("Dev split differs from Master Index")

    rollout_by_token: dict[str, dict[str, Any]] = {}
    for line in args.rollouts.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        token = str(row["token"])
        if token in rollout_by_token:
            raise ValueError(f"Duplicate Dev rollout token: {token}")
        rollout_by_token[token] = row
    if set(rollout_by_token) != expected_tokens:
        raise ValueError("Dev rollout coverage is not exact")

    analyzed: list[dict[str, Any]] = []
    for token in natural_tokens + tail_tokens:
        raw = rollout_by_token[token]
        if raw.get("evidence_phase") != args.evidence_phase:
            raise ValueError(f"Unexpected evidence phase for {token}")
        if any(field not in raw for field in ("raw_response", "poses", "parsed_ok", "response_length", *METRIC_FIELDS)):
            raise ValueError(f"Missing evaluation evidence for {token}")
        metrics = {field: float(raw[field]) for field in METRIC_FIELDS}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Non-finite evaluation metric for {token}")
        parsed_ok = bool(raw["parsed_ok"])
        tier = _classify_tier(parsed_ok, metrics)
        response_length = int(raw["response_length"])
        analyzed.append(
            {
                "token": token,
                "log_name": master[token]["log_name"],
                "split": expected_split[token],
                "intent": master[token]["intent"],
                "parsed_ok": parsed_ok,
                "response_length": response_length,
                "clipped": response_length >= 512,
                "tier": tier,
                "strict_clear": tier == "L3",
                **metrics,
            }
        )

    by_split = {
        "natural": [row for row in analyzed if row["split"] == "dev_natural"],
        "tail": [row for row in analyzed if row["split"] == "dev_tail"],
    }
    args.output_dir.mkdir(parents=True)
    fieldnames = list(analyzed[0])
    with (args.output_dir / "scene_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analyzed)

    summary = {
        "status": "EVAL_COMPLETE",
        "run_id": args.run_id,
        "model_id": args.model_id,
        "evidence_phase": args.evidence_phase,
        "evaluation_seed": args.evaluation_seed,
        "decoding": {"n": 1, "temperature": 0.6, "top_p": 0.95, "max_response_length": 512},
        "combined": _summary(analyzed),
        "natural": _summary(by_split["natural"]),
        "tail": _summary(by_split["tail"]),
        "input_sha256": {
            "rollouts": _sha256(args.rollouts),
            "master_index": _sha256(args.master_index),
            "natural_manifest": _sha256(args.natural_manifest),
            "tail_manifest": _sha256(args.tail_manifest),
            "m0_protocol": _sha256(args.m0_protocol),
        },
        "dev_accessed": True,
        "final_accessed": False,
    }
    (args.output_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tier_rank = {None: -1, "L0": 0, "L1": 1, "L2": 2, "L3": 3}
    tail_worst = sorted(by_split["tail"], key=lambda row: (tier_rank[row["tier"]], row["pdms_scaled"], row["token"]))[:5]
    tail_clear = sorted(
        (row for row in by_split["tail"] if row["strict_clear"]),
        key=lambda row: (-row["pdms_scaled"], row["token"]),
    )[:5]
    example_tokens = [row["token"] for row in tail_worst + tail_clear]
    examples = []
    for token in dict.fromkeys(example_tokens):
        metric_row = next(row for row in analyzed if row["token"] == token)
        examples.append({**metric_row, "raw_response": rollout_by_token[token]["raw_response"], "poses": rollout_by_token[token]["poses"]})
    (args.output_dir / "representative_examples.json").write_text(
        json.dumps(examples, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--natural-manifest", type=Path, required=True)
    parser.add_argument("--tail-manifest", type=Path, required=True)
    parser.add_argument("--m0-protocol", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--evidence-phase", default="dev_eval")
    parser.add_argument("--evaluation-seed", type=int, default=20260827)
    parser.add_argument("--output-dir", type=Path, required=True)
    summarize_eval(parser.parse_args())


if __name__ == "__main__":
    main()
