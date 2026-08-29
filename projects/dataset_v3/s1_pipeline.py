from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Manifest contains duplicate tokens")
    return tokens


def group_rows(rows: list[dict[str, Any]], tokens: list[str], expected_group_size: int = 4) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["token"])].append(row)
    if set(groups) != set(tokens):
        raise ValueError("Rollout tokens do not match the manifest")
    invalid = {token: len(groups[token]) for token in tokens if len(groups[token]) != expected_group_size}
    if invalid:
        raise ValueError(f"Unexpected group sizes: {invalid}")
    return groups


def _post_json(url: str, payload: dict[str, Any], timeout: float, retries: int) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception:
            if attempt == retries:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def replay_group(
    token: str,
    rows: list[dict[str, Any]],
    post_group: Callable[[str, list[list[list[float]]]], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    valid_indices = [index for index, row in enumerate(rows) if bool(row["parsed_ok"])]
    metrics = post_group(token, [rows[index]["poses"] for index in valid_indices]) if valid_indices else []
    if len(metrics) != len(valid_indices):
        raise ValueError(f"Metric response length mismatch for {token}")

    enriched = [dict(row) for row in rows]
    for index, metric in zip(valid_indices, metrics):
        for field in METRIC_FIELDS:
            value = float(metric[field])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {field} for {token}")
            enriched[index][field] = value
        if abs(float(rows[index]["pdms"]) - float(metric["pdms"])) > 1e-8:
            raise ValueError(f"PDMS replay mismatch for {token}")
        if abs(float(rows[index]["pdms_scaled"]) - float(metric["pdms_scaled"])) > 1e-8:
            raise ValueError(f"Scaled PDMS replay mismatch for {token}")
        enriched[index]["metric_replayed"] = True

    for index, row in enumerate(enriched):
        if index not in valid_indices:
            for field in METRIC_FIELDS:
                row[field] = float(row.get(field, 0.0)) if field in {"pdms", "pdms_scaled"} else 0.0
            row["metric_replayed"] = False
    return enriched


def replay_metrics(args: argparse.Namespace) -> None:
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Replay output already exists")
    tokens = read_manifest(args.manifest)
    groups = group_rows(read_jsonl(args.input), tokens)

    def post_group(token: str, poses: list[list[list[float]]]) -> list[dict[str, Any]]:
        result = _post_json(
            args.url,
            {"token": token, "poses": poses, "verbose": False},
            args.timeout,
            args.retries,
        )
        if not isinstance(result, list):
            raise ValueError(f"Metric server returned a non-list for {token}")
        return result

    results: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(replay_group, token, groups[token], post_group): token for token in tokens}
        for completed, future in enumerate(as_completed(futures), start=1):
            token = futures[future]
            results[token] = future.result()
            if completed % 100 == 0 or completed == len(tokens):
                print(f"metric_replay_groups={completed}/{len(tokens)}", flush=True)

    output_rows = [row for token in tokens for row in results[token]]
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(args.output)

    report = {
        "groups": len(tokens),
        "rollouts": len(output_rows),
        "metric_replayed": sum(bool(row["metric_replayed"]) for row in output_rows),
        "parse_failures": sum(not bool(row["parsed_ok"]) for row in output_rows),
        "pdms_replay_tolerance": 1e-8,
        "input_sha256": sha256_file(args.input),
        "output_sha256": sha256_file(args.output),
        "manifest_sha256": sha256_file(args.manifest),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical(value: Any, allowed: tuple[float, ...], name: str) -> float:
    numeric = float(value)
    matches = [candidate for candidate in allowed if abs(numeric - candidate) <= 1e-6 + 1e-12]
    if len(matches) != 1:
        raise ValueError(f"{name}={numeric} is outside the audited evaluator values")
    return matches[0]


def candidate_tier(row: dict[str, Any]) -> str | None:
    if not bool(row["parsed_ok"]):
        return None
    collision = _canonical(row["no_at_fault_collisions"], (0.0, 0.5, 1.0), "collision")
    drivable = _canonical(row["drivable_area_compliance"], (0.0, 1.0), "drivable")
    ttc = _canonical(row["time_to_collision_within_bound"], (0.0, 1.0), "ttc")
    if collision == 0.0:
        return "L0"
    if collision < 1.0 or drivable < 1.0:
        return "L1"
    if ttc < 1.0:
        return "L2"
    return "L3"


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    result = {}
    for label, fraction in (("q00", 0.0), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q90", 0.9), ("q95", 0.95), ("q100", 1.0)):
        index = round((len(ordered) - 1) * fraction)
        result[label] = ordered[index]
    return result


def summarize_screen(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    tokens = read_manifest(args.manifest)
    groups = group_rows(read_jsonl(args.input), tokens)
    args.output_dir.mkdir(parents=True)

    table = []
    tier_counts: Counter[str] = Counter()
    for token in tokens:
        rows = groups[token]
        tiers = [candidate_tier(row) for row in rows]
        tier_counts.update(tier for tier in tiers if tier is not None)
        values = [float(row["pdms_scaled"]) for row in rows]
        valid_tiers = [tier for tier in tiers if tier is not None]
        table.append(
            {
                "token": token,
                "tiers": "|".join(tier or "invalid" for tier in tiers),
                "valid_rollouts": len(valid_tiers),
                "severe_count": sum(tier in {"L0", "L1"} for tier in valid_tiers),
                "near_risk_count": sum(tier == "L2" for tier in valid_tiers),
                "strict_clear_count": sum(tier == "L3" for tier in valid_tiers),
                "mixed_tier": int(len(set(valid_tiers)) >= 2),
                "pdms_scaled_mean": statistics.fmean(values),
                "pdms_scaled_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "headroom": max(values) - statistics.fmean(values),
            }
        )

    with (args.output_dir / "screen_group_geometry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    stds = [float(row["pdms_scaled_std"]) for row in table]
    headrooms = [float(row["headroom"]) for row in table]
    report = {
        "groups": len(table),
        "rollouts": len(table) * 4,
        "tier_definition_status": "V2_CANDIDATE_REAUDITED_ON_V3_EVALUATOR_VALUES_NOT_YET_R0_FROZEN",
        "tier_counts": {tier: tier_counts[tier] for tier in TIERS},
        "invalid_rollouts": 4 * len(table) - sum(tier_counts.values()),
        "groups_with_severe": sum(int(row["severe_count"]) > 0 for row in table),
        "groups_with_near_risk": sum(int(row["near_risk_count"]) > 0 for row in table),
        "mixed_tier_groups": sum(bool(row["mixed_tier"]) for row in table),
        "exact_zero_std_groups": sum(value <= 1e-12 for value in stds),
        "low_nonzero_std_groups": sum(1e-12 < value < 0.05 for value in stds),
        "std_quantiles": _quantiles(stds),
        "headroom_quantiles": _quantiles(headrooms),
        "input_sha256": sha256_file(args.input),
        "manifest_sha256": sha256_file(args.manifest),
    }
    (args.output_dir / "screen_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def stable_key(seed: int, namespace: str, token: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{token}".encode()).hexdigest()


def select_candidate_tokens(
    rows: list[dict[str, str]],
    seed: int,
    high_headroom_fraction: float = 0.10,
    batch_size: int = 4,
) -> tuple[list[str], dict[str, Any]]:
    risk = {
        row["token"]
        for row in rows
        if int(row["severe_count"]) + int(row["near_risk_count"]) > 0
    }
    high_headroom_count = math.ceil(len(rows) * high_headroom_fraction)
    ranked = sorted(
        rows,
        key=lambda row: (-float(row["headroom"]), stable_key(seed, "candidate-headroom", row["token"])),
    )
    high_headroom = {row["token"] for row in ranked[:high_headroom_count]}
    selected = risk | high_headroom
    target_size = math.ceil(len(selected) / batch_size) * batch_size
    if len(selected) < target_size:
        for row in ranked:
            selected.add(row["token"])
            if len(selected) == target_size:
                break
    ordered = [row["token"] for row in rows if row["token"] in selected]
    report = {
        "screen_groups": len(rows),
        "risk_groups": len(risk),
        "high_headroom_fraction": high_headroom_fraction,
        "high_headroom_groups": len(high_headroom),
        "risk_high_headroom_overlap": len(risk & high_headroom),
        "batch_size": batch_size,
        "candidate_groups": len(ordered),
        "batch_closure_additions": len(selected - (risk | high_headroom)),
        "minimum_selected_headroom": min(float(row["headroom"]) for row in rows if row["token"] in selected),
    }
    return ordered, report


def build_candidate(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    with args.geometry.open(encoding="utf-8", newline="") as handle:
        geometry = list(csv.DictReader(handle))
    manifest_tokens = read_manifest(args.manifest)
    if [row["token"] for row in geometry] != manifest_tokens:
        raise ValueError("Geometry order does not match the screen manifest")

    candidates, report = select_candidate_tokens(
        geometry,
        seed=args.seed,
        high_headroom_fraction=args.high_headroom_fraction,
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True)
    candidate_manifest = args.output_dir / "candidate_908.txt"
    candidate_manifest.write_text("".join(f"{token}\n" for token in candidates), encoding="utf-8")

    table = pq.read_table(args.parquet)
    parquet_tokens = [str(value["token"]) for value in table.column("answer").to_pylist()]
    index_by_token = {token: index for index, token in enumerate(parquet_tokens)}
    if set(index_by_token) != set(manifest_tokens):
        raise ValueError("Screen parquet tokens do not match the manifest")
    candidate_table = table.take(pa.array([index_by_token[token] for token in candidates]))
    candidate_parquet = args.output_dir / "candidate_908.parquet"
    pq.write_table(candidate_table, candidate_parquet)

    with args.master_index.open(encoding="utf-8-sig", newline="") as handle:
        master = {row["token"]: row for row in csv.DictReader(handle)}
    logs = Counter(master[token]["log_name"] for token in candidates)
    report.update(
        {
            "candidate_intent_counts": dict(sorted(Counter(master[token]["intent"] for token in candidates).items())),
            "candidate_unique_logs": len(logs),
            "candidate_max_per_log": max(logs.values()),
            "candidate_rule": "all groups with any L0-L2 candidate tier UNION screen top-10% headroom; add next headroom rank only to close batch-of-4",
            "confirm_protocol": {
                "blocks": 2,
                "screen_seed": 20260827,
                "confirm_seed": 20260828,
                "rollouts_per_block": 4,
                "total_rollouts_per_candidate": 8,
                "classification": "requires occurrence in both independent blocks; exact category ratios freeze after confirm capacity audit and before Select",
            },
            "geometry_sha256": sha256_file(args.geometry),
            "screen_manifest_sha256": sha256_file(args.manifest),
            "candidate_manifest_sha256": sha256_file(candidate_manifest),
            "candidate_parquet_sha256": sha256_file(candidate_parquet),
        }
    )
    (args.output_dir / "candidate_freeze_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "exit_code").write_text("0\n", encoding="utf-8")
    (args.output_dir / "COMPLETE").touch()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    replay = commands.add_parser("replay-metrics")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--report", type=Path, required=True)
    replay.add_argument("--url", default="http://127.0.0.1:8901/score_group")
    replay.add_argument("--workers", type=int, default=4)
    replay.add_argument("--timeout", type=float, default=300.0)
    replay.add_argument("--retries", type=int, default=3)
    replay.set_defaults(function=replay_metrics)

    summarize = commands.add_parser("summarize-screen")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.set_defaults(function=summarize_screen)

    candidate = commands.add_parser("build-candidate")
    candidate.add_argument("--geometry", type=Path, required=True)
    candidate.add_argument("--manifest", type=Path, required=True)
    candidate.add_argument("--parquet", type=Path, required=True)
    candidate.add_argument("--master-index", type=Path, required=True)
    candidate.add_argument("--output-dir", type=Path, required=True)
    candidate.add_argument("--seed", type=int, default=20260827)
    candidate.add_argument("--high-headroom-fraction", type=float, default=0.10)
    candidate.add_argument("--batch-size", type=int, default=4)
    candidate.set_defaults(function=build_candidate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
