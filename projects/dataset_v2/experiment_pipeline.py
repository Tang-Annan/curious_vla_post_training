#!/usr/bin/env python3
"""Prepare and analyze the preregistered Dataset V2 experiment stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.dataset_v2.build_dataset_v2 import Row, largest_remainder, select_rows, stable_key


INTENTS = ("straight", "left", "right")
METRICS = (
    "pdms",
    "pdms_scaled",
    "safe",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
)
REQUIRED_TRAIN_TAGS = (
    "actor/pg_loss",
    "actor/entropy_loss",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/pg_clipfrac_higher",
    "actor/pg_clipfrac_lower",
    "actor/grad_norm",
    "actor/lr",
    "critic/advantages/mean",
    "reward/overall",
    "reward/pdms",
    "reward/pdms_scaled",
    "reward/safe",
    "reward/no_at_fault_collisions",
    "reward/drivable_area_compliance",
    "reward/ego_progress",
    "reward/time_to_collision_within_bound",
    "reward/history_comfort",
    "response_length/clip_ratio",
    "timing_s/step",
)


def load_tokens(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate tokens in {path}")
    return tokens


def write_tokens(path: Path, tokens: list[str]) -> None:
    path.write_text("".join(f"{token}\n" for token in tokens), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_master(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["token"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError("Master index is empty")
    return rows


def master_rows(master: dict[str, dict[str, str]], tokens: list[str]) -> list[Row]:
    missing = set(tokens) - set(master)
    if missing:
        raise ValueError(f"Master index is missing {len(missing)} tokens")
    return [
        Row(
            index=int(master[token]["source_row"]),
            token=token,
            log_name=master[token]["log_name"],
            intent=master[token]["intent"],
            source_image=master[token]["source_image"],
            v2_image=master[token]["v2_image"],
        )
        for token in tokens
    ]


def fixed_subset(
    master: dict[str, dict[str, str]], tokens: list[str], total: int, seed: int, namespace: str, max_per_log: int
) -> list[str]:
    rows = master_rows(master, tokens)
    quotas = largest_remainder(Counter(row.intent for row in rows), total)
    selected = select_rows(
        rows,
        total=total,
        quotas=quotas,
        seed=seed,
        namespace=namespace,
        max_per_log=max_per_log,
        min_logs=total // max_per_log,
    )
    return [row.token for row in selected]


def write_shuffled_parquet(source: Path, output: Path, selected: list[str], master: dict[str, dict[str, str]], seed: int) -> None:
    ranked = sorted(selected, key=lambda token: stable_key(seed, "image-shuffle", token))
    replacement = {token: master[ranked[(index + 1) % len(ranked)]]["v2_image"] for index, token in enumerate(ranked)}
    if any(master[token]["log_name"] == master[ranked[(index + 1) % len(ranked)]]["log_name"] for index, token in enumerate(ranked)):
        raise AssertionError("Image shuffle did not cross logs")
    wanted = set(selected)
    records = []
    for batch in pq.ParquetFile(source).iter_batches():
        for row in batch.to_pylist():
            token = str(row["answer"]["token"])
            if token in wanted:
                records.append({**row, "images": [replacement[token]]})
    if len(records) != len(selected):
        raise ValueError(f"Shuffled parquet coverage mismatch: {len(records)} != {len(selected)}")
    pq.write_table(pa.Table.from_pylist(records), output, compression="zstd")


def prepare(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    master = load_master(args.master_index)
    phase1 = load_tokens(args.phase1_manifest)
    image_tokens = fixed_subset(master, phase1, 256, args.seed, "image-sensitivity", 1)
    stability_tokens = fixed_subset(master, phase1, 500, args.seed, "selector-stability", 2)
    image_manifest = args.output_dir / "image_sensitivity_256.txt"
    stability_manifest = args.output_dir / "selector_stability_500.txt"
    shuffled_parquet = args.output_dir / "image_sensitivity_shuffled.parquet"
    write_tokens(image_manifest, image_tokens)
    write_tokens(stability_manifest, stability_tokens)
    write_shuffled_parquet(args.train_parquet, shuffled_parquet, image_tokens, master, args.seed)
    report = {
        "seed": args.seed,
        "image_sensitivity": {"tokens": 256, "logs": 256, "manifest_sha256": sha256(image_manifest)},
        "selector_stability": {
            "tokens": 500,
            "logs": len({master[token]["log_name"] for token in stability_tokens}),
            "manifest_sha256": sha256(stability_manifest),
        },
        "shuffled_parquet_sha256": sha256(shuffled_parquet),
    }
    (args.output_dir / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_groups(path: Path, allowed: set[str], expected: int) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            token = str(row["token"])
            if token not in allowed:
                raise ValueError(f"Unexpected rollout token {token}")
            groups[token].append(row)
    if set(groups) != allowed:
        raise ValueError(f"Rollout token coverage mismatch: missing={len(allowed - set(groups))}")
    mismatched = [token for token, rows in groups.items() if len(rows) != expected]
    if mismatched:
        raise ValueError(f"Rollout group-size mismatch for {len(mismatched)} tokens")
    return groups


def cluster_bootstrap(
    differences: dict[str, dict[str, float]], master: dict[str, dict[str, str]], samples: int, seed: int
) -> dict[str, dict[str, float | list[float]]]:
    by_log: dict[str, list[dict[str, float]]] = defaultdict(list)
    for token, row in differences.items():
        by_log[master[token]["log_name"]].append(row)
    logs = sorted(by_log)
    sums = np.asarray([[sum(row[metric] for row in by_log[log]) for metric in METRICS] for log in logs])
    counts = np.asarray([len(by_log[log]) for log in logs], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(METRICS)), dtype=float)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        indices = rng.integers(0, len(logs), size=(stop - start, len(logs)))
        draws[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1, keepdims=True)
    result = {}
    for index, metric in enumerate(METRICS):
        values = np.asarray([row[metric] for row in differences.values()])
        low, high = np.quantile(draws[:, index], [0.025, 0.975])
        result[metric] = {"mean_difference": float(values.mean()), "paired_log_cluster_95_ci": [float(low), float(high)]}
    return result


def analyze_i0(args: argparse.Namespace) -> dict:
    tokens = load_tokens(args.manifest)
    allowed = set(tokens)
    master = load_master(args.master_index)
    correct = load_groups(args.correct, allowed, 1)
    shuffled = load_groups(args.shuffled, allowed, 1)
    differences = {}
    response_changes = 0
    trajectory_changes = 0
    for token in tokens:
        left, right = correct[token][0], shuffled[token][0]
        response_changes += left.get("response") != right.get("response")
        trajectory_changes += left.get("poses") != right.get("poses")
        differences[token] = {metric: float(left[metric]) - float(right[metric]) for metric in METRICS}
    bootstrap = cluster_bootstrap(differences, master, args.bootstrap_samples, args.seed)
    correct_parse = float(np.mean([bool(correct[token][0].get("parsed_ok", True)) for token in tokens]))
    shuffled_parse = float(np.mean([bool(shuffled[token][0].get("parsed_ok", True)) for token in tokens]))
    pdms = bootstrap["pdms"]
    gates = {
        "pdms_difference_at_least_0_01": pdms["mean_difference"] >= 0.01,
        "pdms_ci_lower_above_zero": pdms["paired_log_cluster_95_ci"][0] > 0.0,
        "correct_parse_at_least_0_995": correct_parse >= 0.995,
        "shuffled_parse_at_least_0_995": shuffled_parse >= 0.995,
    }
    gates["passed"] = all(gates.values())
    report = {
        "id": "V2-I0",
        "tokens": len(tokens),
        "logs": len({master[token]["log_name"] for token in tokens}),
        "response_change_rate": response_changes / len(tokens),
        "trajectory_change_rate": trajectory_changes / len(tokens),
        "parse_rate": {"correct": correct_parse, "shuffled": shuffled_parse},
        "differences": bootstrap,
        "gates": gates,
        "decision": "proceed" if gates["passed"] else "pause_grpo_image_sensitivity_failed",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def group_stats(rows: list[dict]) -> dict[str, float]:
    raw = np.asarray([float(row["pdms"]) for row in rows])
    scaled = np.asarray([float(row["pdms_scaled"]) for row in rows])
    return {
        "pdms_mean": float(raw.mean()),
        "pdms_std": float(raw.std(ddof=1)),
        "pdms_min": float(raw.min()),
        "pdms_max": float(raw.max()),
        "scaled_mean": float(scaled.mean()),
        "scaled_std": float(scaled.std(ddof=1)),
        "scaled_max": float(scaled.max()),
    }


def adas_eligible(stats: dict[str, float]) -> bool:
    value_range = stats["pdms_max"] - stats["pdms_min"]
    if stats["pdms_std"] <= 0.01 or value_range <= 1e-6:
        return False
    p_est = stats["pdms_mean"] / value_range
    if not 0.0 <= p_est <= 1.0 or p_est**4 + (1.0 - p_est) ** 4 >= 0.20:
        return False
    k_est = round(p_est * 4)
    predicted_std = math.sqrt(k_est * (4 - k_est) / 12) * value_range
    confidence_error = abs(predicted_std - stats["pdms_std"]) / stats["pdms_std"]
    return math.isfinite(confidence_error) and confidence_error < 0.10


def fals_score(stats: dict[str, float]) -> float:
    return (1.0 - stats["scaled_mean"]) * (stats["scaled_max"] - stats["scaled_mean"])


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left | right else 1.0


def rankdata(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda token: (values[token], token))
    ranks = {}
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[index]]:
            stop += 1
        rank = (index + stop - 1) / 2.0
        for token in ordered[index:stop]:
            ranks[token] = rank
        index = stop
    return ranks


def spearman(left: dict[str, float], right: dict[str, float]) -> float:
    left_ranks, right_ranks = rankdata(left), rankdata(right)
    tokens = sorted(left)
    return float(np.corrcoef([left_ranks[token] for token in tokens], [right_ranks[token] for token in tokens])[0, 1])


def analyze_s0(args: argparse.Namespace) -> dict:
    tokens = load_tokens(args.manifest)
    allowed = set(tokens)
    if len(args.block) != 4:
        raise ValueError("V2-S0 requires exactly four independent G4 blocks")
    block_count = len(args.block)
    eligible_sets = []
    scores = []
    for path in args.block:
        groups = load_groups(path, allowed, 4)
        stats = {token: group_stats(groups[token]) for token in tokens}
        eligible_sets.append({token for token in tokens if adas_eligible(stats[token])})
        scores.append({token: fals_score(stats[token]) for token in tokens})
    eligible_ratios = np.asarray([len(tokens_set) / len(tokens) for tokens_set in eligible_sets])
    adas_jaccards = [jaccard(eligible_sets[a], eligible_sets[b]) for a, b in combinations(range(block_count), 2)]
    correlations = [spearman(scores[a], scores[b]) for a, b in combinations(range(block_count), 2)]
    top_size = math.ceil(len(tokens) * 0.25)
    top_sets = [set(sorted(tokens, key=lambda token: (-block[token], token))[:top_size]) for block in scores]
    top_jaccards = [jaccard(top_sets[a], top_sets[b]) for a, b in combinations(range(block_count), 2)]
    ratio_mean = float(eligible_ratios.mean())
    ratio_cv = float(eligible_ratios.std(ddof=1) / ratio_mean) if ratio_mean else math.inf
    gates = {
        "adas_eligible_ratio_cv_at_most_0_20": ratio_cv <= 0.20,
        "adas_membership_jaccard_median_at_least_0_50": float(np.median(adas_jaccards)) >= 0.50,
        "fals_rank_spearman_median_at_least_0_60": float(np.median(correlations)) >= 0.60,
        "fals_top25_jaccard_median_at_least_0_50": float(np.median(top_jaccards)) >= 0.50,
        "all_statistics_finite": bool(np.isfinite([*eligible_ratios, *adas_jaccards, *correlations, *top_jaccards]).all()),
    }
    report = {
        "id": "V2-S0",
        "blocks": block_count,
        "tokens": len(tokens),
        "adas": {
            "eligible_ratios": eligible_ratios.tolist(),
            "eligible_ratio_cv": ratio_cv,
            "membership_jaccard_median": float(np.median(adas_jaccards)),
        },
        "fals": {
            "rank_spearman_median": float(np.median(correlations)),
            "top25_jaccard_median": float(np.median(top_jaccards)),
        },
        "gates": {**gates, "adas_passed": gates["adas_eligible_ratio_cv_at_most_0_20"] and gates["adas_membership_jaccard_median_at_least_0_50"] and gates["all_statistics_finite"], "fals_passed": gates["fals_rank_spearman_median_at_least_0_60"] and gates["fals_top25_jaccard_median_at_least_0_50"] and gates["all_statistics_finite"]},
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def constrained_ranked_select(
    tokens: list[str], master: dict[str, dict[str, str]], scores: dict[str, float] | None, seed: int, namespace: str
) -> tuple[list[str] | None, dict[str, int]]:
    quotas = {"straight": 634, "left": 251, "right": 115}
    selected = []
    per_log = Counter()
    counts = Counter()
    ranked = sorted(
        tokens,
        key=lambda token: (
            -(scores[token] if scores is not None else 0.0),
            stable_key(seed, namespace, token),
        ),
    )
    for token in ranked:
        row = master[token]
        intent, log_name = row["intent"], row["log_name"]
        if counts[intent] >= quotas[intent] or per_log[log_name] >= 5:
            continue
        selected.append(token)
        counts[intent] += 1
        per_log[log_name] += 1
        if len(selected) == 1000:
            break
    return (selected if len(selected) == 1000 else None), dict(counts)


def signal_geometry(tokens: list[str], stats: dict[str, dict[str, float]]) -> dict[str, float]:
    rows = [stats[token] for token in tokens]
    return {
        "mean_pdms_scaled": float(np.mean([row["scaled_mean"] for row in rows])),
        "mean_group_std": float(np.mean([row["scaled_std"] for row in rows])),
        "exact_zero_ratio": float(np.mean([row["scaled_std"] == 0.0 for row in rows])),
        "mean_headroom": float(np.mean([row["scaled_max"] - row["scaled_mean"] for row in rows])),
    }


def build_manifests(args: argparse.Namespace) -> dict:
    master = load_master(args.master_index)
    candidates = load_tokens(args.candidate_manifest)
    groups = load_groups(args.rollouts, set(candidates), 4)
    stats = {token: group_stats(groups[token]) for token in candidates}
    eligible = [token for token in candidates if adas_eligible(stats[token])]
    fals_scores = {token: fals_score(stats[token]) for token in candidates}
    adas, adas_counts = constrained_ranked_select(eligible, master, None, args.seed, "adas-g4-current")
    fals, fals_counts = constrained_ranked_select(candidates, master, fals_scores, args.seed, "fals-g4")
    extension_required = adas is None or fals is None
    report = {
        "id": "V2-M0",
        "candidate_tokens": len(candidates),
        "adas_eligible_tokens": len(eligible),
        "adas_selected_counts": adas_counts,
        "fals_selected_counts": fals_counts,
        "extension_required": extension_required and len(candidates) == 6000,
        "gates": {"adas_feasible": adas is not None, "fals_feasible": fals is not None},
    }
    if extension_required:
        report["decision"] = "enable_common_extension" if len(candidates) == 6000 else "selector_infeasible_close_failed_branches"
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        if len(candidates) == 6000:
            random_tokens = load_tokens(args.random_manifest)
        else:
            random_tokens, _ = constrained_ranked_select(candidates, master, None, args.seed, "random-8k")
            if random_tokens is None:
                raise ValueError("Random-8K manifest is infeasible")
        manifests = {"random": random_tokens, "adas": adas, "fals": fals}
        for name, values in manifests.items():
            write_tokens(args.output_dir / f"{name}_1k.txt", values)
        overlap = {}
        for left, right in combinations(manifests, 2):
            intersection = len(set(manifests[left]) & set(manifests[right]))
            overlap[f"{left}__{right}"] = {
                "intersection": intersection,
                "jaccard": intersection / len(set(manifests[left]) | set(manifests[right])),
            }
        fals_cutoff = min(fals_scores[token] for token in fals)
        report.update(
            {
                "decision": "freeze_manifests",
                "manifests": {
                    name: {
                        "sha256": sha256(args.output_dir / f"{name}_1k.txt"),
                        "logs": len({master[token]["log_name"] for token in values}),
                        "signal_geometry": signal_geometry(values, stats),
                    }
                    for name, values in manifests.items()
                },
                "overlap": overlap,
                "fals_cutoff": fals_cutoff,
                "fals_cutoff_ties": sum(math.isclose(score, fals_cutoff) for score in fals_scores.values()),
            }
        )
        (args.output_dir / "manifest_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def compare_rollouts(args: argparse.Namespace) -> dict:
    tokens = load_tokens(args.manifest)
    allowed = set(tokens)
    master = load_master(args.master_index)
    baseline = load_groups(args.baseline, allowed, 1)
    candidate = load_groups(args.candidate, allowed, 1)
    differences = {
        token: {metric: float(candidate[token][0][metric]) - float(baseline[token][0][metric]) for metric in METRICS}
        for token in tokens
    }
    result = {
        "tokens": len(tokens),
        "logs": len({master[token]["log_name"] for token in tokens}),
        "difference": "candidate_minus_baseline",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "metrics": cluster_bootstrap(differences, master, args.bootstrap_samples, args.seed),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def finite_numbers(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from finite_numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from finite_numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def verify_train(args: argparse.Namespace) -> dict:
    manifest = set(load_tokens(args.manifest))
    train = load_groups(args.train_rollouts, manifest, 4)
    train_rows = [row for rows in train.values() for row in rows]
    dev_rows = []
    if args.dev_rollouts:
        dev_manifest = set(load_tokens(args.dev_manifest))
        dev = load_groups(args.dev_rollouts, dev_manifest, 1)
        dev_rows = [row for rows in dev.values() for row in rows]
    for row in [*train_rows, *dev_rows]:
        values = list(finite_numbers(row))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Rollouts contain non-finite values")
        if int(row.get("response_length", 0)) >= 512:
            raise ValueError("Rollouts contain clipped responses")
    train_parse = float(np.mean([bool(row.get("parsed_ok", True)) for row in train_rows]))
    dev_parse = float(np.mean([bool(row.get("parsed_ok", True)) for row in dev_rows])) if dev_rows else None
    if train_parse < 0.995 or (dev_parse is not None and dev_parse < 0.995):
        raise ValueError(f"Parse gate failed: train={train_parse} dev={dev_parse}")

    log_rows = [json.loads(line) for line in args.training_log.read_text(encoding="utf-8-sig").splitlines() if line]
    if [int(row["step"]) for row in log_rows] != list(range(1, args.expected_steps + 1)):
        raise ValueError("Training steps are incomplete or out of order")
    if any(not math.isfinite(value) for row in log_rows for value in finite_numbers(row)):
        raise ValueError("Training log contains non-finite values")
    if any(float(row["response_length"]["clip_ratio"]) != 0.0 for row in log_rows):
        raise ValueError("Training response clipping is nonzero")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = list(args.tensorboard_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError("TensorBoard event file is missing")
    scalar_tags = set()
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        scalar_tags.update(accumulator.Tags().get("scalars", []))
    missing_tags = set(REQUIRED_TRAIN_TAGS) - scalar_tags
    if missing_tags:
        raise ValueError(f"TensorBoard scalar tags missing: {sorted(missing_tags)}")
    report = {
        "passed": True,
        "steps": args.expected_steps,
        "train_groups": len(train),
        "train_rollouts": len(train_rows),
        "dev_rollouts": len(dev_rows),
        "train_parse_rate": train_parse,
        "dev_parse_rate": dev_parse,
        "clipping": 0,
        "finite": True,
        "tensorboard_event_files": len(event_files),
        "tensorboard_scalar_tags": sorted(scalar_tags),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("prepare")
    command.add_argument("--master-index", type=Path, required=True)
    command.add_argument("--phase1-manifest", type=Path, required=True)
    command.add_argument("--train-parquet", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--seed", type=int, default=20260825)
    command.set_defaults(function=prepare)

    command = subparsers.add_parser("analyze-i0")
    command.add_argument("--correct", type=Path, required=True)
    command.add_argument("--shuffled", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--master-index", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--bootstrap-samples", type=int, default=20000)
    command.add_argument("--seed", type=int, default=20260825)
    command.set_defaults(function=analyze_i0)

    command = subparsers.add_parser("analyze-s0")
    command.add_argument("--block", type=Path, action="append", required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=analyze_s0)

    command = subparsers.add_parser("build-manifests")
    command.add_argument("--rollouts", type=Path, required=True)
    command.add_argument("--candidate-manifest", type=Path, required=True)
    command.add_argument("--random-manifest", type=Path, required=True)
    command.add_argument("--master-index", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--report", type=Path, required=True)
    command.add_argument("--seed", type=int, default=20260825)
    command.set_defaults(function=build_manifests)

    command = subparsers.add_parser("compare")
    command.add_argument("--baseline", type=Path, required=True)
    command.add_argument("--candidate", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--master-index", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--bootstrap-samples", type=int, default=20000)
    command.add_argument("--seed", type=int, default=20260825)
    command.set_defaults(function=compare_rollouts)

    command = subparsers.add_parser("verify-train")
    command.add_argument("--train-rollouts", type=Path, required=True)
    command.add_argument("--dev-rollouts", type=Path)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--dev-manifest", type=Path)
    command.add_argument("--training-log", type=Path, required=True)
    command.add_argument("--tensorboard-dir", type=Path, required=True)
    command.add_argument("--expected-steps", type=int, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=verify_train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = args.function(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
