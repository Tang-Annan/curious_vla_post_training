from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from projects.dataset_v3.data_prep import scene_risk
from projects.dataset_v3.inventory import load_navsim_log, sha256_file


SAFETY_FIELDS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
)
METRIC_FIELDS = (*SAFETY_FIELDS, "ego_progress", "history_comfort", "pdms", "pdms_scaled")
STABLE_CATEGORIES = {"stable_severe", "stable_mixed_recoverable", "stable_near_risk"}
PROXIMITY_BINS = ("vehicle_and_vru", "vehicle", "vru", "near_noninteraction", "far", "no_actor")


def read_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Manifest contains duplicate tokens: {path}")
    return tokens


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite distance: {value}")
    return numeric


def proximity_bin(vehicle: float | None, vru: float | None) -> str:
    vehicle_risk = vehicle is not None and vehicle <= 5.0
    vru_risk = vru is not None and vru <= 10.0
    if vehicle_risk and vru_risk:
        return "vehicle_and_vru"
    if vehicle_risk:
        return "vehicle"
    if vru_risk:
        return "vru"
    distances = [value for value in (vehicle, vru) if value is not None]
    if not distances:
        return "no_actor"
    return "near_noninteraction" if min(distances) <= 20.0 else "far"


def feature_row(
    *,
    token: str,
    log_name: str,
    intent: str,
    map_location: str,
    vehicle: float | None,
    vru: float | None,
) -> dict[str, Any]:
    interaction = (vehicle is not None and vehicle <= 5.0) or (vru is not None and vru <= 10.0)
    nearest = min((value for value in (vehicle, vru) if value is not None), default=None)
    return {
        "token": token,
        "log_name": log_name,
        "intent": intent,
        "map_location": map_location,
        "month": log_name[:7],
        "min_vehicle_distance_m": "" if vehicle is None else vehicle,
        "min_vru_distance_m": "" if vru is None else vru,
        "nearest_actor_distance_m": "" if nearest is None else nearest,
        "interaction_tail_flag": int(interaction),
        "proximity_bin": proximity_bin(vehicle, vru),
    }


def extract_log_features(path: Path, token_list: list[str]) -> list[dict[str, Any]]:
    targets = set(token_list)
    with path.open("rb") as handle:
        frames = load_navsim_log(handle)
    rows = []
    for frame in frames:
        token = str(frame.get("token", ""))
        if token not in targets:
            continue
        vehicle, vru, _ = scene_risk(frame)
        rows.append(
            feature_row(
                token=token,
                log_name=path.stem,
                intent="",
                map_location=str(frame.get("map_location", "")),
                vehicle=vehicle,
                vru=vru,
            )
        )
    if {row["token"] for row in rows} != targets:
        missing = sorted(targets - {row["token"] for row in rows})
        raise ValueError(f"Raw log {path.stem} is missing target tokens: {missing[:5]}")
    return rows


def extract_train_features(
    raw_log_root: Path,
    stability_rows: list[dict[str, str]],
    master: dict[str, dict[str, str]],
    workers: int,
) -> list[dict[str, Any]]:
    targets_by_log: dict[str, list[str]] = defaultdict(list)
    stability_by_token = {row["token"]: row for row in stability_rows}
    for token, row in stability_by_token.items():
        master_row = master[token]
        if master_row["source_universe"] != "sft_seen" or master_row["split"] != "grpo_screen":
            raise ValueError(f"Screen token escaped the frozen train universe: {token}")
        targets_by_log[master_row["log_name"]].append(token)

    raw_paths = {path.stem: path for path in raw_log_root.rglob("*.pkl")}
    if len(raw_paths) != 1310:
        raise ValueError(f"Expected 1,310 unique NAVSIM logs, found {len(raw_paths)}")
    missing_logs = sorted(set(targets_by_log) - set(raw_paths))
    if missing_logs:
        raise ValueError(f"Missing train logs: {missing_logs[:5]}")

    extracted: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_log_features, raw_paths[log_name], tokens): log_name
            for log_name, tokens in targets_by_log.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            for row in future.result():
                if row["token"] in extracted:
                    raise ValueError(f"Duplicate extracted train token: {row['token']}")
                extracted[row["token"]] = row
            if completed % 100 == 0 or completed == len(futures):
                print(f"raw_logs={completed}/{len(futures)}", flush=True)

    if set(extracted) != set(stability_by_token):
        raise ValueError("Extracted train features do not cover the frozen Screen")
    rows = []
    for token in stability_by_token:
        row = extracted[token]
        master_row = master[token]
        stability = stability_by_token[token]
        row.update(
            {
                "intent": master_row["intent"],
                "category": stability["category"],
                "screen_severe_count": int(stability["screen_severe_count"]),
                "screen_near_risk_count": int(stability["screen_near_risk_count"]),
                "screen_mixed_recoverable": int(stability["screen_mixed_recoverable"]),
            }
        )
        rows.append(row)
    return rows


def counter_js(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        raise ValueError("Cannot compare empty distributions")
    result = 0.0
    for key in keys:
        p = left[key] / left_total
        q = right[key] / right_total
        midpoint = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {key: None for key in ("q00", "q25", "q50", "q75", "q90", "q100")}
    return {
        key: float(value)
        for key, value in zip(
            ("q00", "q25", "q50", "q75", "q90", "q100"),
            np.quantile(array, (0, 0.25, 0.5, 0.75, 0.9, 1)),
        )
    }


def feature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenes": len(rows),
        "logs": len({row["log_name"] for row in rows}),
        "interaction_scenes": sum(int(row["interaction_tail_flag"]) for row in rows),
        "interaction_rate": sum(int(row["interaction_tail_flag"]) for row in rows) / len(rows),
        "proximity_bins": dict(sorted(Counter(str(row["proximity_bin"]) for row in rows).items())),
        "intent_counts": dict(sorted(Counter(str(row["intent"]) for row in rows).items())),
        "month_counts": dict(sorted(Counter(str(row["month"]) for row in rows).items())),
        "map_location_counts": dict(sorted(Counter(str(row["map_location"]) for row in rows).items())),
        "nearest_actor_distance_m": quantiles(
            float(row["nearest_actor_distance_m"])
            for row in rows
            if row["nearest_actor_distance_m"] != ""
        ),
    }


def cluster_rate_delta(
    rows: list[dict[str, Any]],
    left_tokens: set[str],
    right_tokens: set[str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    logs = sorted({str(row["log_name"]) for row in rows})
    by_log = {log: [row for row in rows if row["log_name"] == log] for log in logs}
    left_total = np.asarray([sum(row["token"] in left_tokens for row in by_log[log]) for log in logs])
    right_total = np.asarray([sum(row["token"] in right_tokens for row in by_log[log]) for log in logs])
    left_risk = np.asarray(
        [sum(row["token"] in left_tokens and int(row["interaction_tail_flag"]) for row in by_log[log]) for log in logs]
    )
    right_risk = np.asarray(
        [sum(row["token"] in right_tokens and int(row["interaction_tail_flag"]) for row in by_log[log]) for log in logs]
    )
    point = right_risk.sum() / right_total.sum() - left_risk.sum() / left_total.sum()
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, resamples, 512):
        count = min(512, resamples - start)
        indices = rng.integers(0, len(logs), size=(count, len(logs)))
        sampled_left_total = left_total[indices].sum(axis=1)
        sampled_right_total = right_total[indices].sum(axis=1)
        valid = (sampled_left_total > 0) & (sampled_right_total > 0)
        sampled = (
            right_risk[indices].sum(axis=1)[valid] / sampled_right_total[valid]
            - left_risk[indices].sum(axis=1)[valid] / sampled_left_total[valid]
        )
        deltas.extend(sampled.tolist())
    lower, upper = np.quantile(np.asarray(deltas), (0.025, 0.975))
    return {"point_delta": float(point), "ci_lower": float(lower), "ci_upper": float(upper)}


def outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"scenes": 0}
    return {
        "scenes": len(rows),
        "strict_clear_rate": sum(bool(row["strict_clear"]) for row in rows) / len(rows),
        "metric_means": {
            field: sum(float(row[field]) for row in rows) / len(rows) for field in METRIC_FIELDS
        },
    }


def model_outcome_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    natural = [row for row in rows if row["split"] == "dev_natural"]
    tail = [row for row in rows if row["split"] == "dev_tail"]
    interaction = [row for row in rows if int(row["interaction_tail_flag"])]
    noninteraction = [row for row in rows if not int(row["interaction_tail_flag"])]
    tail_interaction = [row for row in tail if int(row["interaction_tail_flag"])]
    tail_noninteraction = [row for row in tail if not int(row["interaction_tail_flag"])]
    interaction_summary = outcome_summary(interaction)
    noninteraction_summary = outcome_summary(noninteraction)
    tail_interaction_summary = outcome_summary(tail_interaction)
    tail_noninteraction_summary = outcome_summary(tail_noninteraction)
    return {
        "natural": outcome_summary(natural),
        "tail": outcome_summary(tail),
        "all_dev_by_gt_interaction": {
            "interaction": interaction_summary,
            "noninteraction": noninteraction_summary,
            "strict_clear_delta": interaction_summary["strict_clear_rate"]
            - noninteraction_summary["strict_clear_rate"],
        },
        "tail_by_scene_interaction": {
            "interaction": tail_interaction_summary,
            "noninteraction": tail_noninteraction_summary,
            "strict_clear_delta": tail_interaction_summary["strict_clear_rate"]
            - tail_noninteraction_summary["strict_clear_rate"],
        },
    }


def build_report(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    model_rows: dict[str, list[dict[str, Any]]],
    random_tokens: set[str],
    tailmix_tokens: set[str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    random_rows = [row for row in train_rows if row["token"] in random_tokens]
    tailmix_rows = [row for row in train_rows if row["token"] in tailmix_tokens]
    stable_tokens = {row["token"] for row in train_rows if row["category"] in STABLE_CATEGORIES}
    anchor_tokens = {row["token"] for row in train_rows if row["category"] == "random_anchor"}
    stable_rows = [row for row in train_rows if row["token"] in stable_tokens]
    anchor_rows = [row for row in train_rows if row["token"] in anchor_tokens]
    dev_natural = [row for row in dev_rows if row["split"] == "dev_natural"]
    dev_tail = [row for row in dev_rows if row["split"] == "dev_tail"]

    random_summary = feature_summary(random_rows)
    tailmix_summary = feature_summary(tailmix_rows)
    dev_tail_summary = feature_summary(dev_tail)
    proximity_js_random = counter_js(
        Counter(row["proximity_bin"] for row in random_rows), Counter(row["proximity_bin"] for row in dev_tail)
    )
    proximity_js_tailmix = counter_js(
        Counter(row["proximity_bin"] for row in tailmix_rows), Counter(row["proximity_bin"] for row in dev_tail)
    )
    random_gap = abs(random_summary["interaction_rate"] - dev_tail_summary["interaction_rate"])
    tailmix_gap = abs(tailmix_summary["interaction_rate"] - dev_tail_summary["interaction_rate"])
    selector_delta = cluster_rate_delta(
        train_rows, random_tokens, tailmix_tokens, resamples=resamples, seed=seed
    )
    category_delta = cluster_rate_delta(
        train_rows, anchor_tokens, stable_tokens, resamples=resamples, seed=seed + 1
    )
    checks = {
        "tailmix_gt_interaction_point_enrichment": selector_delta["point_delta"] > 0,
        "tailmix_gt_interaction_ci_lower_positive": selector_delta["ci_lower"] > 0,
        "stable_policy_risk_gt_interaction_point_enrichment": category_delta["point_delta"] > 0,
        "stable_policy_risk_gt_interaction_ci_lower_positive": category_delta["ci_lower"] > 0,
        "tailmix_proximity_distribution_closer_to_dev_tail_than_random": proximity_js_tailmix < proximity_js_random,
        "tailmix_interaction_rate_closer_to_dev_tail_than_random": tailmix_gap < random_gap,
    }
    directional = [
        checks["tailmix_gt_interaction_point_enrichment"],
        checks["stable_policy_risk_gt_interaction_point_enrichment"],
        checks["tailmix_proximity_distribution_closer_to_dev_tail_than_random"],
        checks["tailmix_interaction_rate_closer_to_dev_tail_than_random"],
    ]
    if all(directional):
        support_status = "ALIGNED_ON_ALL_DIRECTIONAL_CHECKS"
    elif not any(directional):
        support_status = "MISALIGNED_ON_ALL_DIRECTIONAL_CHECKS"
    else:
        support_status = "MIXED_DIRECTIONAL_EVIDENCE"

    return {
        "status": "SEMANTIC_ALIGNMENT_AUDIT_COMPLETE",
        "semantic_definitions": {
            "training_tailmix": "policy-conditioned dual-rollout stability categories plus matched random anchors",
            "evaluation_tail": "policy-independent log-level GT actor proximity enrichment; vehicle<=5m or VRU<=10m",
            "definition_identity": False,
        },
        "coverage": {
            "train_screen": len(train_rows),
            "random": len(random_rows),
            "tailmix": len(tailmix_rows),
            "stable_policy_risk": len(stable_rows),
            "random_anchor": len(anchor_rows),
            "dev_natural": len(dev_natural),
            "dev_tail": len(dev_tail),
            "models": sorted(model_rows),
        },
        "train_selector_comparison": {
            "random": random_summary,
            "tailmix": tailmix_summary,
            "tailmix_minus_random_interaction_rate": selector_delta,
        },
        "stable_category_alignment": {
            "stable_policy_risk": feature_summary(stable_rows),
            "random_anchor": feature_summary(anchor_rows),
            "stable_minus_anchor_interaction_rate": category_delta,
            "by_category": {
                category: feature_summary([row for row in train_rows if row["category"] == category])
                for category in sorted({row["category"] for row in train_rows})
            },
        },
        "evaluation_geometry": {
            "dev_natural": feature_summary(dev_natural),
            "dev_tail": dev_tail_summary,
        },
        "train_to_dev_support": {
            "proximity_bin_js_divergence": {
                "random_to_dev_tail": proximity_js_random,
                "tailmix_to_dev_tail": proximity_js_tailmix,
            },
            "interaction_rate_absolute_gap": {
                "random_to_dev_tail": random_gap,
                "tailmix_to_dev_tail": tailmix_gap,
            },
            "categorical_js_divergence": {
                field: {
                    "random_to_dev_tail": counter_js(
                        Counter(str(row[field]) for row in random_rows),
                        Counter(str(row[field]) for row in dev_tail),
                    ),
                    "tailmix_to_dev_tail": counter_js(
                        Counter(str(row[field]) for row in tailmix_rows),
                        Counter(str(row[field]) for row in dev_tail),
                    ),
                }
                for field in ("intent", "month", "map_location")
            },
            "directional_checks": checks,
            "support_status": support_status,
        },
        "dev_outcome_alignment": {
            model: model_outcome_report(rows) for model, rows in sorted(model_rows.items())
        },
        "bootstrap": {"resamples": resamples, "cluster": "log_name", "seed": seed},
        "dev_accessed": True,
        "final_accessed": False,
    }


def parse_models(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Dev model must use NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate model name: {name}")
        result[name] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train-logs", type=Path, required=True)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--stability-capacity", type=Path, required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--tailmix-manifest", type=Path, required=True)
    parser.add_argument("--dev-natural", type=Path, required=True)
    parser.add_argument("--dev-tail", type=Path, required=True)
    parser.add_argument("--dev-model", action="append", default=[], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")
    if args.workers < 1 or args.bootstrap_resamples < 1:
        raise ValueError("Workers and bootstrap resamples must be positive")

    stability_rows = read_csv(args.stability_capacity)
    if len(stability_rows) != 8000 or len({row["token"] for row in stability_rows}) != 8000:
        raise ValueError("Stability capacity is not the frozen 8,000-token Screen")
    master_rows = read_csv(args.master_index)
    master = {row["token"]: row for row in master_rows}
    if len(master) != len(master_rows):
        raise ValueError("Master Index contains duplicate tokens")
    random_tokens = set(read_manifest(args.random_manifest))
    tailmix_tokens = set(read_manifest(args.tailmix_manifest))
    if len(random_tokens) != 2000 or len(tailmix_tokens) != 2000:
        raise ValueError("Selector manifests must contain exactly 2,000 tokens")

    natural_tokens = read_manifest(args.dev_natural)
    tail_tokens = read_manifest(args.dev_tail)
    if len(natural_tokens) != 210 or len(tail_tokens) != 206 or set(natural_tokens) & set(tail_tokens):
        raise ValueError("Dev manifests do not match the frozen 210/206 split")
    dev_tokens = natural_tokens + tail_tokens
    dev_split = {**{token: "dev_natural" for token in natural_tokens}, **{token: "dev_tail" for token in tail_tokens}}
    dev_rows = []
    for token in dev_tokens:
        row = master[token]
        if row["source_universe"] != "sft_unseen" or row["split"] != dev_split[token]:
            raise ValueError(f"Dev token escaped the frozen Dev universe: {token}")
        dev_rows.append(
            {
                **feature_row(
                    token=token,
                    log_name=row["log_name"],
                    intent=row["intent"],
                    map_location=row["map_location"],
                    vehicle=optional_float(row["min_vehicle_distance_m"]),
                    vru=optional_float(row["min_vru_distance_m"]),
                ),
                "split": dev_split[token],
            }
        )
    dev_by_token = {row["token"]: row for row in dev_rows}

    train_rows = extract_train_features(args.raw_train_logs, stability_rows, master, args.workers)
    for row in train_rows:
        row["random"] = int(row["token"] in random_tokens)
        row["tailmix"] = int(row["token"] in tailmix_tokens)

    model_paths = parse_models(args.dev_model)
    model_rows = {}
    long_rows = []
    for model, path in model_paths.items():
        rows = read_csv(path)
        by_token = {row["token"]: row for row in rows}
        if set(by_token) != set(dev_tokens):
            raise ValueError(f"Dev coverage mismatch for model {model}")
        combined = []
        for token in dev_tokens:
            raw = by_token[token]
            row = {
                **dev_by_token[token],
                "model": model,
                "tier": raw["tier"],
                "strict_clear": raw["strict_clear"].lower() == "true",
                **{field: float(raw[field]) for field in METRIC_FIELDS},
            }
            combined.append(row)
            long_rows.append(row)
        model_rows[model] = combined

    report = build_report(
        train_rows,
        dev_rows,
        model_rows,
        random_tokens,
        tailmix_tokens,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    report["input_sha256"] = {
        "master_index": sha256_file(args.master_index),
        "stability_capacity": sha256_file(args.stability_capacity),
        "random_manifest": sha256_file(args.random_manifest),
        "tailmix_manifest": sha256_file(args.tailmix_manifest),
        "dev_natural": sha256_file(args.dev_natural),
        "dev_tail": sha256_file(args.dev_tail),
        "dev_models": {model: sha256_file(path) for model, path in sorted(model_paths.items())},
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "train_scene_features.csv", train_rows)
    write_csv(args.output_dir / "dev_model_outcomes.csv", long_rows)
    (args.output_dir / "tail_semantic_alignment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
