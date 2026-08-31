from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


REWARD_FIELDS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
    "time_to_at_fault_collision",
    "time_to_ttc_infraction",
    "min_distance_to_actors",
    "pdms",
    "pdms_scaled",
)
REWARD_TYPES = ("raw_pdms", "cdt_task", "safety_continuous")
FAMILIES = ("proximity", "construction", "signal")
INTENTS = ("straight", "left", "right")

_REWARD_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "EasyR1"
    / "verl"
    / "utils"
    / "reward_score"
    / "navsim"
    / "cdt_scalar_reward.py"
)


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


def load_reward_module() -> Any:
    spec = importlib.util.spec_from_file_location("v4_cdt_scalar_reward", _REWARD_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reward module from {_REWARD_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def group_rows(
    rows: list[dict[str, Any]], tokens: list[str], expected_group_size: int = 4
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["token"])].append(row)
    if set(groups) != set(tokens):
        raise ValueError("Rollout tokens do not match the manifest")
    invalid = {token: len(groups[token]) for token in tokens if len(groups[token]) != expected_group_size}
    if invalid:
        raise ValueError(f"Unexpected group sizes: {invalid}")
    return groups


def _validate_metric(metric: dict[str, Any]) -> None:
    distance = metric["min_distance_to_actors"]
    if distance is not None:
        distance = float(distance)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(f"min_distance_to_actors is invalid: {distance}")
    for field in ("time_to_at_fault_collision", "time_to_ttc_infraction"):
        value = metric[field]
        if value is not None:
            value = float(value)
            if value < 0.0 or (not math.isfinite(value) and value != math.inf):
                raise ValueError(f"{field} is invalid: {value}")


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
        _validate_metric(metric)
        for field in REWARD_FIELDS:
            value = metric[field]
            if value is not None:
                value = float(value)
                if field not in {"time_to_at_fault_collision", "time_to_ttc_infraction"} and not math.isfinite(value):
                    raise ValueError(f"Non-finite {field} for {token}")
            enriched[index][field] = value
        if abs(float(rows[index]["pdms"]) - float(metric["pdms"])) > 1e-8:
            raise ValueError(f"PDMS replay mismatch for {token}")
        if abs(float(rows[index]["pdms_scaled"]) - float(metric["pdms_scaled"])) > 1e-8:
            raise ValueError(f"Scaled PDMS replay mismatch for {token}")
        enriched[index]["metric_replayed"] = True

    for index, row in enumerate(enriched):
        if index not in valid_indices:
            for field in REWARD_FIELDS:
                if field in {"pdms", "pdms_scaled"}:
                    row[field] = float(row.get(field, 0.0))
                else:
                    row[field] = 0.0
            row["metric_replayed"] = False
    return enriched


def replay(args: argparse.Namespace) -> None:
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
                print(f"reward_replay_groups={completed}/{len(tokens)}", flush=True)

    output_rows = [row for token in tokens for row in results[token]]
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
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


def trainer_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Metrics exactly as the trainer reward callable sees them."""
    metrics = {}
    for field in REWARD_FIELDS:
        value = row[field]
        metrics[field] = float("inf") if value is None else float(value)
    if not bool(row["parsed_ok"]):
        for field in REWARD_FIELDS:
            metrics[field] = 0.0
    return metrics


def candidate_rewards(row: dict[str, Any], reward: Any) -> dict[str, float]:
    metrics = trainer_metrics(row)
    return {
        "raw_pdms": reward.raw_pdms_reward(metrics),
        "cdt_task": reward.cdt_task_reward(bool(row["parsed_ok"]), metrics)[0],
        "safety_continuous": reward.safety_continuous_reward(metrics),
        "hard_safe": reward.safety_hard_gate(metrics),
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        label: ordered[round((len(ordered) - 1) * fraction)]
        for label, fraction in (("q00", 0.0), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q95", 0.95), ("q100", 1.0))
    }


def group_spread(values: list[float]) -> dict[str, float]:
    distinct = len(set(values))
    return {
        "distinct": distinct,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "range": max(values) - min(values),
        "mean": statistics.fmean(values),
    }


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denom_left = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    denom_right = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    if denom_left == 0.0 or denom_right == 0.0:
        return float("nan")
    return numerator / (denom_left * denom_right)


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[str(row["token"])] = row
    return labels


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of an empty list")
    return statistics.median(values)


def audit_report(
    groups: dict[str, list[dict[str, Any]]],
    tokens: list[str],
    labels: dict[str, dict[str, str]],
    reward: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_counts = Counter(labels[token]["exclusive_family"] for token in tokens)
    table: list[dict[str, Any]] = []
    by_family: dict[str, list[float]] = {family: [] for family in FAMILIES}
    family_group_ranges: dict[str, list[float]] = {family: [] for family in FAMILIES}
    family_distinct2: Counter[str] = Counter()
    family_all_unsafe: Counter[str] = Counter()
    spread_by_type: dict[str, list[dict[str, float]]] = {reward_type: [] for reward_type in REWARD_TYPES}
    raw_equal_groups = 0
    raw_equal_safety_spread_groups = 0
    inversion_checked = 0
    inversion_violations = 0
    safe_values: list[float] = []
    unsafe_values: list[float] = []

    for token in tokens:
        token_rows = groups[token]
        family = labels[token]["exclusive_family"]
        rewards = [candidate_rewards(row, reward) for row in token_rows]
        for row, entry in zip(token_rows, rewards):
            table.append(
                {
                    "token": token,
                    "log_name": labels[token]["log_name"],
                    "intent": labels[token]["intent"],
                    "exclusive_family": family,
                    "parsed_ok": int(bool(row["parsed_ok"])),
                    "hard_safe": entry["hard_safe"],
                    "ego_progress": float(row["ego_progress"]),
                    "raw_pdms_reward": entry["raw_pdms"],
                    "cdt_task_reward": entry["cdt_task"],
                    "safety_continuous_reward": entry["safety_continuous"],
                    "time_to_at_fault_collision": (
                        float("inf") if row["time_to_at_fault_collision"] is None else float(row["time_to_at_fault_collision"])
                    ),
                    "time_to_ttc_infraction": (
                        float("inf") if row["time_to_ttc_infraction"] is None else float(row["time_to_ttc_infraction"])
                    ),
                    "min_distance_to_actors": (
                        float("inf") if row["min_distance_to_actors"] is None else float(row["min_distance_to_actors"])
                    ),
                }
            )

        for reward_type in REWARD_TYPES:
            spread_by_type[reward_type].append(group_spread([entry[reward_type] for entry in rewards]))
        raw_values = [entry["raw_pdms"] for entry in rewards]
        safety_values = [entry["safety_continuous"] for entry in rewards]
        if len(set(raw_values)) == 1:
            raw_equal_groups += 1
            if len(set(safety_values)) >= 2:
                raw_equal_safety_spread_groups += 1

        by_family[family].extend(safety_values)
        family_group_ranges[family].append(max(safety_values) - min(safety_values))
        if len(set(safety_values)) >= 2:
            family_distinct2[family] += 1
        if all(entry["hard_safe"] == 0.0 for entry in rewards):
            family_all_unsafe[family] += 1

        unsafe = [entry["safety_continuous"] for entry in rewards if entry["hard_safe"] == 0.0]
        safe = [entry["safety_continuous"] for entry in rewards if entry["hard_safe"] == 1.0]
        if unsafe and safe:
            inversion_checked += 1
            unsafe_values.extend(unsafe)
            safe_values.extend(safe)
            if min(safe) <= max(unsafe):
                inversion_violations += 1

    def type_summary(reward_type: str) -> dict[str, Any]:
        spreads = spread_by_type[reward_type]
        ranges = [entry["range"] for entry in spreads]
        stds = [entry["std"] for entry in spreads]
        means = [entry["mean"] for entry in spreads]
        return {
            "groups_with_distinct2": sum(entry["distinct"] >= 2 for entry in spreads),
            "groups_with_distinct2_fraction": sum(entry["distinct"] >= 2 for entry in spreads) / len(spreads),
            "zero_spread_groups": sum(entry["distinct"] == 1 for entry in spreads),
            "mean_group_std": statistics.fmean(stds),
            "median_group_range": _median(ranges),
            "mean_group_mean": statistics.fmean(means),
        }

    family_rows = {}
    for family in FAMILIES:
        family_rows[family] = {
            "groups": len(family_group_ranges[family]),
            "mean_reward": statistics.fmean(by_family[family]),
            "median_group_range": _median(family_group_ranges[family]),
            "groups_with_distinct2": family_distinct2[family],
            "groups_with_distinct2_fraction": family_distinct2[family] / len(family_group_ranges[family]),
            "all_unsafe_groups": family_all_unsafe[family],
        }

    all_valid = [row for row in table if row["parsed_ok"]]
    correlations = {
        "safety_vs_ego_progress": _pearson(
            [float(row["safety_continuous_reward"]) for row in all_valid],
            [float(row["ego_progress"]) for row in all_valid],
        ),
        "safety_vs_min_distance": _pearson(
            [float(row["safety_continuous_reward"]) for row in all_valid],
            [float(row["min_distance_to_actors"]) for row in all_valid],
        ),
        "safety_vs_pdms": _pearson(
            [float(row["safety_continuous_reward"]) for row in all_valid],
            [float(row["raw_pdms_reward"]) for row in all_valid],
        ),
    }

    report = {
        "manifest_tokens": len(tokens),
        "rollouts": sum(len(groups[token]) for token in tokens),
        "family_counts": dict(sorted(family_counts.items())),
        "reward_distribution": {
            reward_type: _quantiles([entry[f"{reward_type}_reward"] for entry in table])
            for reward_type in REWARD_TYPES
        },
        "reward_type_summary": {reward_type: type_summary(reward_type) for reward_type in REWARD_TYPES},
        "effective_gain": {
            "raw_pdms_equal_groups": raw_equal_groups,
            "raw_pdms_equal_safety_spread_groups": raw_equal_safety_spread_groups,
        },
        "hard_safety_inversion": {
            "checked_groups": inversion_checked,
            "violations": inversion_violations,
            "safe_min": min(safe_values) if safe_values else None,
            "unsafe_max": max(unsafe_values) if unsafe_values else None,
        },
        "family_differentiation": family_rows,
        "correlations": correlations,
        "not_gt_imitation": {
            "by_construction": True,
            "reward_inputs": [
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "time_to_at_fault_collision",
                "time_to_ttc_infraction",
                "min_distance_to_actors",
                "ego_progress",
                "history_comfort",
            ],
            "gt_trajectory_input": False,
        },
        "gates": {
            "differentiation": {
                "effective_gain_groups_gt_zero": raw_equal_safety_spread_groups > 0,
                "zero_spread_safety_lte_raw": type_summary("safety_continuous")["zero_spread_groups"]
                <= type_summary("raw_pdms")["zero_spread_groups"],
            },
            "inversion": {"violations_zero": inversion_violations == 0},
            "family": {
                "proximity_ge_construction": family_rows["proximity"]["median_group_range"]
                >= family_rows["construction"]["median_group_range"],
                "proximity_ge_signal": family_rows["proximity"]["median_group_range"]
                >= family_rows["signal"]["median_group_range"],
            },
            "not_gt_imitation": {"by_construction": True},
        },
        "input_sha256": {
            "rollouts": None,
            "manifest": None,
            "labels": None,
        },
        "dev_accessed": False,
        "final_accessed": False,
    }
    return table, report


def audit(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    tokens = read_manifest(args.manifest)
    rows = read_jsonl(args.input)
    groups = group_rows(rows, tokens)
    labels = read_labels(args.labels)
    if set(labels) != set(tokens):
        raise ValueError("Label tokens do not match the manifest")

    family_counts = Counter(labels[token]["exclusive_family"] for token in tokens)
    if any(labels[token]["exclusive_family"] not in FAMILIES for token in tokens):
        raise ValueError("Manifest contains a token outside the frozen risk families")
    if dict(family_counts) != {"proximity": 1000, "construction": 500, "signal": 500}:
        raise ValueError(f"Frozen Risk50 family composition drifted: {dict(family_counts)}")

    table, report = audit_report(groups, tokens, labels, load_reward_module())
    report["input_sha256"] = {
        "rollouts": sha256_file(args.input),
        "manifest": sha256_file(args.manifest),
        "labels": sha256_file(args.labels),
    }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "candidate_reward_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (args.output_dir / "reward_audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--input", type=Path, required=True)
    replay_parser.add_argument("--manifest", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument("--report", type=Path, required=True)
    replay_parser.add_argument("--url", default="http://127.0.0.1:8901/score_group")
    replay_parser.add_argument("--workers", type=int, default=2)
    replay_parser.add_argument("--timeout", type=float, default=300.0)
    replay_parser.add_argument("--retries", type=int, default=3)
    replay_parser.set_defaults(function=replay)

    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--input", type=Path, required=True)
    audit_parser.add_argument("--manifest", type=Path, required=True)
    audit_parser.add_argument("--labels", type=Path, required=True)
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser.set_defaults(function=audit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
