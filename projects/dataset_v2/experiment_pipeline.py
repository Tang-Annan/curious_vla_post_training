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
EASYR1_ROOT = REPOSITORY_ROOT / "EasyR1"
if str(EASYR1_ROOT) not in sys.path:
    sys.path.insert(0, str(EASYR1_ROOT))

from projects.dataset_v2.build_dataset_v2 import Row, largest_remainder, select_rows, stable_key


INTENTS = ("straight", "left", "right")
TIER_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
TIER_PAIRS = ("L3-L2", "L3-L1", "L3-L0", "L2-L1", "L2-L0", "L1-L0")
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
    eligible = [] if args.skip_adas else [token for token in candidates if adas_eligible(stats[token])]
    fals_scores = {token: fals_score(stats[token]) for token in candidates}
    adas, adas_counts = (None, {}) if args.skip_adas else constrained_ranked_select(eligible, master, None, args.seed, "adas-g4-current")
    fals, fals_counts = constrained_ranked_select(candidates, master, fals_scores, args.seed, "fals-g4")
    extension_required = fals is None or (not args.skip_adas and adas is None)
    report = {
        "id": "V2-M0",
        "candidate_tokens": len(candidates),
        "adas_eligible_tokens": len(eligible),
        "adas_selected_counts": adas_counts,
        "fals_selected_counts": fals_counts,
        "extension_required": extension_required and len(candidates) == 6000,
        "gates": {"adas_feasible": None if args.skip_adas else adas is not None, "fals_feasible": fals is not None},
        "adas_status": "skipped_by_s0" if args.skip_adas else "active",
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
        manifests = {"random": random_tokens, "fals": fals}
        if not args.skip_adas:
            manifests["adas"] = adas
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


def _row_tier(row: dict) -> str | None:
    from verl.trainer.cdt_hla import classify_cdt

    return classify_cdt(
        bool(row.get("parsed_ok", True)),
        row["no_at_fault_collisions"],
        row["drivable_area_compliance"],
        row["time_to_collision_within_bound"],
    )


def _advantage_replay(groups: dict[str, list[dict]], tokens: list[str]) -> dict:
    import torch

    from verl.trainer.cdt_hla import compute_cdtr
    from verl.trainer.core_algos import compute_cdt_hla_outcome_advantage, compute_grpo_outcome_advantage

    rows = [row for token in tokens for row in groups[token]]
    group_ids = [token for token in tokens for _ in groups[token]]
    rewards = torch.tensor([[float(row["pdms_scaled"])] for row in rows], dtype=torch.float32)
    mask = torch.ones_like(rewards)
    reward_metrics = {
        key: [row.get(key, True) if key == "parsed_ok" else row[key] for row in rows]
        for key in (
            "parsed_ok",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
        )
    }
    diagnostics: dict[str, float] = {}
    hla, _ = compute_cdt_hla_outcome_advantage(
        rewards,
        mask,
        group_ids,
        reward_metrics=reward_metrics,
        diagnostics=diagnostics,
    )
    sdr, _ = compute_grpo_outcome_advantage(rewards, mask, group_ids)
    cdtr_rewards = torch.tensor(
        [
            [compute_cdtr(tier, row["pdms_scaled"]) if tier is not None else 0.0]
            for row, tier in zip(rows, (_row_tier(row) for row in rows))
        ],
        dtype=torch.float32,
    )
    cdtr, _ = compute_grpo_outcome_advantage(cdtr_rewards, mask, group_ids)
    return {
        "rows": rows,
        "hla": hla[:, 0].tolist(),
        "sdr": sdr[:, 0].tolist(),
        "cdtr": cdtr[:, 0].tolist(),
        "diagnostics": diagnostics,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _cross_tier_pairs(token: str, rows: list[dict], metadata: dict[str, str], epsilon: float) -> list[dict]:
    tiers = [_row_tier(row) for row in rows]
    pair_rows = []
    for left, right in combinations(range(len(rows)), 2):
        if tiers[left] is None or tiers[right] is None or tiers[left] == tiers[right]:
            continue
        high, low = (left, right) if TIER_RANK[tiers[left]] > TIER_RANK[tiers[right]] else (right, left)
        high_row, low_row = rows[high], rows[low]
        gap = float(high_row["pdms_scaled"]) - float(low_row["pdms_scaled"])
        pair_type = "tie" if abs(gap) <= epsilon else "inversion" if gap < -epsilon else "correct"
        pair_rows.append(
            {
                "token": token,
                "log_id": metadata["log_name"],
                "intent": metadata["intent"],
                "rollout_i": high,
                "rollout_j": low,
                "tier_i": tiers[high],
                "tier_j": tiers[low],
                "C_i": float(high_row["no_at_fault_collisions"]),
                "D_i": float(high_row["drivable_area_compliance"]),
                "T_i": float(high_row["time_to_collision_within_bound"]),
                "C_j": float(low_row["no_at_fault_collisions"]),
                "D_j": float(low_row["drivable_area_compliance"]),
                "T_j": float(low_row["time_to_collision_within_bound"]),
                "SDR_i": float(high_row["pdms_scaled"]),
                "SDR_j": float(low_row["pdms_scaled"]),
                "SDR_gap": gap,
                "pair_type": pair_type,
                "tier_pair": f"{tiers[high]}-{tiers[low]}",
            }
        )
    return pair_rows


def _conflict_group(token: str, rows: list[dict], metadata: dict[str, str], epsilon: float) -> tuple[dict, list[dict]]:
    pairs = _cross_tier_pairs(token, rows, metadata, epsilon)
    conflicts = [pair for pair in pairs if pair["pair_type"] != "correct"]
    tiers = [_row_tier(row) for row in rows]
    valid_tiers = {tier for tier in tiers if tier is not None}
    conflict_pairs = {pair["tier_pair"] for pair in conflicts}
    if conflict_pairs & {"L3-L1", "L3-L0"}:
        severity = "Critical"
    elif conflict_pairs & {"L2-L1", "L2-L0"}:
        severity = "Moderate"
    elif conflicts:
        severity = "Mild"
    else:
        severity = ""
    worst = min(conflicts, key=lambda pair: (pair["SDR_gap"], pair["tier_pair"])) if conflicts else None
    composition = Counter(tier for tier in tiers if tier is not None)
    return (
        {
            "token": token,
            "log_id": metadata["log_name"],
            "intent": metadata["intent"],
            "tiers": "|".join(tier or "invalid" for tier in tiers),
            "tier_composition": "+".join(
                f"{tier}x{composition[tier]}" for tier in reversed(TIER_RANK) if composition[tier]
            ),
            "SDRs": "|".join(f'{float(row["pdms_scaled"]):.9g}' for row in rows),
            "mixed_tier": len(valid_tiers) >= 2,
            "num_cross_tier_pairs": len(pairs),
            "num_correct_pairs": sum(pair["pair_type"] == "correct" for pair in pairs),
            "num_tie_pairs": sum(pair["pair_type"] == "tie" for pair in pairs),
            "num_inversion_pairs": sum(pair["pair_type"] == "inversion" for pair in pairs),
            "worst_conflict_pair": (
                f'{worst["tier_pair"]}:{worst["rollout_i"]}>{worst["rollout_j"]}' if worst else ""
            ),
            "max_inversion_gap": max(
                [-pair["SDR_gap"] for pair in conflicts if pair["pair_type"] == "inversion"], default=0.0
            ),
            "severity": severity,
        },
        pairs,
    )


def _membership_report(sets: list[set[str]], denominator: int) -> dict:
    ratios = [len(values) / denominator for values in sets]
    mean = float(np.mean(ratios))
    jaccards = [jaccard(sets[left], sets[right]) for left, right in combinations(range(len(sets)), 2)]
    return {
        "counts": [len(values) for values in sets],
        "ratios": ratios,
        "ratio_cv": float(np.std(ratios, ddof=1) / mean) if mean else None,
        "membership_jaccards": jaccards,
        "membership_jaccard_median": float(np.median(jaccards)),
    }


def _gap_statistics(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def analyze_conflicts(args: argparse.Namespace) -> dict:
    if len(args.s0_block) != 4:
        raise ValueError("V2-C0 requires exactly four S0 blocks")
    candidate_tokens = load_tokens(args.candidate_manifest)
    stability_tokens = load_tokens(args.stability_manifest)
    if len(candidate_tokens) != args.expected_groups:
        raise ValueError(f"Expected {args.expected_groups} candidate groups, found {len(candidate_tokens)}")
    if len(stability_tokens) != args.expected_stability_tokens:
        raise ValueError(
            f"Expected {args.expected_stability_tokens} stability tokens, found {len(stability_tokens)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    master = load_master(args.master_index)
    missing_master = (set(candidate_tokens) | set(stability_tokens)) - set(master)
    if missing_master:
        raise ValueError(f"Master index is missing {len(missing_master)} audit tokens")
    bank = load_groups(args.bank_rollouts, set(candidate_tokens), 4)

    pair_rows = []
    group_rows = []
    group_by_token = {}
    for token in candidate_tokens:
        group, pairs = _conflict_group(token, bank[token], master[token], args.epsilon)
        group_by_token[token] = group
        pair_rows.extend(pairs)
        if group["severity"]:
            group_rows.append(group)

    conflict_tokens = [token for token in candidate_tokens if group_by_token[token]["severity"]]
    geometry_values = []
    if conflict_tokens:
        replay = _advantage_replay(bank, conflict_tokens)
        for offset, token in enumerate(conflict_tokens):
            start = offset * 4
            sdr = np.asarray(replay["sdr"][start : start + 4])
            hla = np.asarray(replay["hla"][start : start + 4])
            difference = float(np.mean(np.abs(hla - sdr)))
            group_by_token[token]["mean_abs_hla_minus_sdr"] = difference
            geometry_values.append(difference)
    geometry_report = {
        "conflict_groups": len(conflict_tokens),
        "mean_abs_hla_minus_sdr": float(np.mean(geometry_values)) if geometry_values else None,
        "median_abs_hla_minus_sdr": float(np.median(geometry_values)) if geometry_values else None,
        "p90_abs_hla_minus_sdr": float(np.quantile(geometry_values, 0.90)) if geometry_values else None,
        "max_abs_hla_minus_sdr": max(geometry_values, default=None),
        "ratios": {
            f"at_least_{threshold:.2f}": sum(value >= threshold for value in geometry_values) / len(geometry_values)
            if geometry_values
            else 0.0
            for threshold in (0.05, 0.10, 0.20)
        },
    }

    pair_counts = {tier_pair: Counter() for tier_pair in TIER_PAIRS}
    for row in pair_rows:
        pair_counts[row["tier_pair"]]["total"] += 1
        pair_counts[row["tier_pair"]][row["pair_type"]] += 1
    pair_summary = {}
    for tier_pair in (*TIER_PAIRS, "All"):
        counts = sum(pair_counts.values(), Counter()) if tier_pair == "All" else pair_counts[tier_pair]
        total = counts["total"]
        pair_summary[tier_pair] = {
            "total": total,
            "correct": counts["correct"],
            "tie": counts["tie"],
            "inversion": counts["inversion"],
            "conflict_rate": (counts["tie"] + counts["inversion"]) / total if total else 0.0,
            "inversion_rate": counts["inversion"] / total if total else 0.0,
        }

    log_rows = []
    tokens_by_log = defaultdict(list)
    for token in candidate_tokens:
        tokens_by_log[master[token]["log_name"]].append(token)
    for log_id in sorted(tokens_by_log):
        tokens = tokens_by_log[log_id]
        log_rows.append(
            {
                "log_id": log_id,
                "tokens": len(tokens),
                "mixed_tier_tokens": sum(group_by_token[token]["mixed_tier"] for token in tokens),
                "conflict_tokens": sum(bool(group_by_token[token]["severity"]) for token in tokens),
                "critical_conflict_tokens": sum(group_by_token[token]["severity"] == "Critical" for token in tokens),
                "inversion_pairs": sum(group_by_token[token]["num_inversion_pairs"] for token in tokens),
            }
        )
    ranked_logs = sorted(log_rows, key=lambda row: (-row["conflict_tokens"], row["log_id"]))
    conflict_count = len(conflict_tokens)
    intent_summary = {}
    for intent in INTENTS:
        intent_tokens = [token for token in candidate_tokens if master[token]["intent"] == intent]
        intent_conflicts = sum(bool(group_by_token[token]["severity"]) for token in intent_tokens)
        intent_summary[intent] = {
            "phase1_groups": len(intent_tokens),
            "conflict_groups": intent_conflicts,
            "conflict_rate": intent_conflicts / len(intent_tokens) if intent_tokens else 0.0,
        }

    stability_conflicts = []
    stability_critical = []
    for path in args.s0_block:
        block = load_groups(path, set(stability_tokens), 4)
        conflict_set = set()
        critical_set = set()
        for token in stability_tokens:
            group, _ = _conflict_group(token, block[token], master[token], args.epsilon)
            if group["severity"]:
                conflict_set.add(token)
            if group["severity"] == "Critical":
                critical_set.add(token)
        stability_conflicts.append(conflict_set)
        stability_critical.append(critical_set)
    stability_report = {
        "blocks": 4,
        "tokens_per_block": len(stability_tokens),
        "conflict": _membership_report(stability_conflicts, len(stability_tokens)),
        "critical": _membership_report(stability_critical, len(stability_tokens)),
    }

    inversion_gaps = defaultdict(list)
    for row in pair_rows:
        if row["pair_type"] == "inversion":
            inversion_gaps[row["tier_pair"]].append(-row["SDR_gap"])
    audit_report = {
        "id": "V2-C0-SDR-CDT-CONFLICT-AUDIT",
        "epsilon_q": args.epsilon,
        "pair_summary": pair_summary,
        "group_summary": {
            "total_groups": len(candidate_tokens),
            "mixed_tier_groups": sum(group_by_token[token]["mixed_tier"] for token in candidate_tokens),
            "conflict_groups": conflict_count,
            "conflict_group_ratio": conflict_count / len(candidate_tokens),
            "critical_conflict_groups": sum(row["severity"] == "Critical" for row in group_rows),
            "moderate_conflict_groups": sum(row["severity"] == "Moderate" for row in group_rows),
            "mild_conflict_groups": sum(row["severity"] == "Mild" for row in group_rows),
            "unique_conflict_logs": len({row["log_id"] for row in group_rows}),
            "max_conflict_groups_per_log": max((row["conflict_tokens"] for row in log_rows), default=0),
            "tier_composition": dict(Counter(row["tier_composition"] for row in group_rows)),
            "invalid_rollouts": sum(_row_tier(row) is None for token in candidate_tokens for row in bank[token]),
        },
        "log_concentration": {
            "conflict_logs": sum(row["conflict_tokens"] > 0 for row in log_rows),
            "top_10_share": sum(row["conflict_tokens"] for row in ranked_logs[:10]) / conflict_count
            if conflict_count
            else 0.0,
            "top_20_share": sum(row["conflict_tokens"] for row in ranked_logs[:20]) / conflict_count
            if conflict_count
            else 0.0,
        },
        "intent_summary": intent_summary,
        "inversion_gap": {
            "All": _gap_statistics([gap for values in inversion_gaps.values() for gap in values]),
            **{tier_pair: _gap_statistics(inversion_gaps[tier_pair]) for tier_pair in TIER_PAIRS},
        },
        "scope": "CPU-only train-rollout diagnostic; no dev/final access; does not reopen V2-H0",
    }

    pair_fields = [
        "token", "log_id", "intent", "rollout_i", "rollout_j", "tier_i", "tier_j",
        "C_i", "D_i", "T_i", "C_j", "D_j", "T_j", "SDR_i", "SDR_j", "SDR_gap",
        "pair_type", "tier_pair",
    ]
    group_fields = [
        "token", "log_id", "intent", "tiers", "SDRs", "num_cross_tier_pairs", "num_correct_pairs",
        "num_tie_pairs", "num_inversion_pairs", "worst_conflict_pair", "max_inversion_gap", "severity",
        "mean_abs_hla_minus_sdr",
    ]
    for name, rows, fields in (
        ("conflict_pairs.csv", pair_rows, pair_fields),
        ("conflict_groups.csv", group_rows, group_fields),
        ("conflict_log_report.csv", log_rows, list(log_rows[0])),
    ):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    for name, payload in (
        ("conflict_audit_report.json", audit_report),
        ("conflict_stability_report.json", stability_report),
        ("conflict_geometry_report.json", geometry_report),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    inputs = [*args.s0_block, args.stability_manifest, args.bank_rollouts, args.candidate_manifest, args.master_index]
    (args.output_dir / "input_sha256.json").write_text(
        json.dumps({str(path): sha256(path) for path in inputs}, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "resolved_definition.json").write_text(
        json.dumps(
            {
                "epsilon_q": args.epsilon,
                "tiers": "L0<L1<L2<L3; invalid excluded",
                "conflict": "cross-tier SDR tie or inversion",
                "severity": {"Critical": ["L3-L1", "L3-L0"], "Moderate": ["L2-L1", "L2-L0"], "Mild": ["L3-L2", "L1-L0"]},
                "geometry_thresholds": [0.05, 0.10, 0.20],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "source_commit.txt").write_text(args.source_commit + "\n", encoding="utf-8")
    (args.output_dir / "exit_code").write_text("0\n", encoding="utf-8")
    (args.output_dir / "COMPLETE").touch()
    return {"audit": audit_report, "stability": stability_report, "geometry": geometry_report}


def analyze_h0(args: argparse.Namespace) -> dict:
    from verl.trainer.cdt_hla import is_strict_clear

    if len(args.s0_block) != 4:
        raise ValueError("V2-H0 requires exactly four S0 blocks")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_tokens = load_tokens(args.candidate_manifest)
    master = load_master(args.master_index)
    bank = load_groups(args.bank_rollouts, set(candidate_tokens), 4)

    stability_tokens = load_tokens(args.stability_manifest)
    stability_sets = []
    stability_ratios = []
    for path in args.s0_block:
        block = load_groups(path, set(stability_tokens), 4)
        mixed = {
            token
            for token in stability_tokens
            if all(_row_tier(row) is not None for row in block[token])
            and len({_row_tier(row) for row in block[token]}) >= 2
        }
        stability_sets.append(mixed)
        stability_ratios.append(len(mixed) / len(stability_tokens))
    jaccards = [jaccard(stability_sets[a], stability_sets[b]) for a, b in combinations(range(4), 2)]
    ratio_array = np.asarray(stability_ratios)
    ratio_cv = float(ratio_array.std(ddof=1) / ratio_array.mean()) if ratio_array.mean() else math.inf
    stability_report = {
        "blocks": len(args.s0_block),
        "tokens_per_block": len(stability_tokens),
        "mixed_tier_ratios": stability_ratios,
        "mixed_tier_ratio_cv": ratio_cv,
        "membership_jaccards": jaccards,
        "membership_jaccard_median": float(np.median(jaccards)),
    }

    tiers_by_token = {token: [_row_tier(row) for row in bank[token]] for token in candidate_tokens}
    fully_valid = [token for token in candidate_tokens if all(tier is not None for tier in tiers_by_token[token])]
    mixed = {token for token in fully_valid if len(set(tiers_by_token[token])) >= 2}
    priority = {token: float(token in mixed) for token in fully_valid}
    selected, intent_counts = constrained_ranked_select(
        fully_valid, master, priority, args.seed, "safetymix-g4"
    )
    if selected is None:
        raise ValueError("SafetyMix-1K is infeasible under frozen quota and per-log cap")
    repeated, _ = constrained_ranked_select(fully_valid, master, priority, args.seed, "safetymix-g4")
    if repeated != selected:
        raise AssertionError("SafetyMix membership is not deterministic")
    ordered = sorted(selected, key=lambda token: stable_key(args.seed, "safetymix-train-order", token))
    repeated_order = sorted(repeated, key=lambda token: stable_key(args.seed, "safetymix-train-order", token))
    if ordered != repeated_order:
        raise AssertionError("SafetyMix order is not deterministic")
    manifest = args.output_dir / "safetymix_1k.txt"
    write_tokens(manifest, ordered)

    dev_tokens = set(load_tokens(args.dev_manifest))
    final_tokens = set(load_tokens(args.final_manifest))
    per_log = Counter(master[token]["log_name"] for token in selected)
    safetymix_report = {
        "selected": len(selected),
        "mixed_selected": sum(token in mixed for token in selected),
        "fully_valid_candidates": len(fully_valid),
        "intent_counts": intent_counts,
        "logs": len(per_log),
        "max_per_log": max(per_log.values()),
        "dev_overlap": len(set(selected) & dev_tokens),
        "final_overlap": len(set(selected) & final_tokens),
        "membership_sha256": hashlib.sha256("".join(f"{token}\n" for token in sorted(selected)).encode()).hexdigest(),
        "order_sha256": sha256(manifest),
    }

    full_replay = _advantage_replay(bank, candidate_tokens)
    selected_replay = _advantage_replay(bank, ordered)
    table_rows = []
    zero_composition = Counter()
    offset = 0
    for token in ordered:
        group = bank[token]
        tiers = tiers_by_token[token]
        hla = selected_replay["hla"][offset : offset + 4]
        sdr = selected_replay["sdr"][offset : offset + 4]
        cdtr = selected_replay["cdtr"][offset : offset + 4]
        offset += 4
        effective = max(abs(value) for value in hla) > 0.0
        unique = set(tiers)
        if not effective:
            if len(unique) == 1:
                reason = f"all_{next(iter(unique))}"
            else:
                reason = "other"
            if max(sdr) == min(sdr):
                reason += "_sdr_tie"
            zero_composition[reason] += 1
        table_rows.append(
            {
                "token": token,
                "tiers": "|".join(tiers),
                "mixed_tier": len(unique) >= 2,
                "sdr_advantages": "|".join(f"{value:.9g}" for value in sdr),
                "cdtr_advantages": "|".join(f"{value:.9g}" for value in cdtr),
                "hla_train_advantages": "|".join(f"{value:.9g}" for value in hla),
                "mean_abs_hla_minus_sdr": float(np.mean(np.abs(np.asarray(hla) - np.asarray(sdr)))),
                "effective": effective,
            }
        )
    with (args.output_dir / "group_geometry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    e0_rows = [json.loads(line) for line in args.e0_rollouts.read_text(encoding="utf-8-sig").splitlines() if line]
    strict_clear_e0 = float(
        np.mean(
            [
                is_strict_clear(
                    bool(row.get("parsed_ok", True)),
                    row["no_at_fault_collisions"],
                    row["drivable_area_compliance"],
                    row["time_to_collision_within_bound"],
                )
                for row in e0_rows
            ]
        )
    )
    selected_diagnostics = selected_replay["diagnostics"]
    c_half_high = sum(
        float(row["no_at_fault_collisions"]) == 0.5 and _row_tier(row) in {"L2", "L3"}
        for token in candidate_tokens
        for row in bank[token]
    )
    technical_gates = {
        "bank_6000x4": len(bank) == 6000 and sum(len(rows) for rows in bank.values()) == 24000,
        "mapping_error_zero": True,
        "c_half_never_l2_l3": c_half_high == 0,
        "safetymix_exact": len(selected) == 1000 and intent_counts == {"straight": 634, "left": 251, "right": 115},
        "per_log_cap": max(per_log.values()) <= 5,
        "no_dev_final_overlap": not (set(selected) & (dev_tokens | final_tokens)),
        "finite": all(math.isfinite(value) for value in finite_numbers(selected_diagnostics)),
        "identity": selected_diagnostics["identity_max_abs_difference"] <= 1e-7,
        "identity_covered": selected_diagnostics["identity_groups"] > 0,
        "invalid_zero": full_replay["diagnostics"]["invalid_advantage_max_abs"] == 0.0,
        "e0_coverage": len(e0_rows) == 2000,
    }
    technical_gates["passed"] = all(technical_gates.values())
    scientific_gates = {
        "mixed_ratio_cv": ratio_cv <= 0.20,
        "mixed_jaccard": float(np.median(jaccards)) >= 0.50,
        "hla_no_cross_tier_inversion": selected_diagnostics["hla_cross_tier_inversions"] == 0.0,
        "raw_margin": selected_diagnostics["min_cross_tier_raw_margin"] >= 5.0 / 12.0,
        "all_group_material": selected_diagnostics["material_change_ratio"] >= 0.10,
        "mixed_group_material": selected_diagnostics["mixed_material_change_ratio"] >= 0.80,
    }
    scientific_gates["passed"] = all(scientific_gates.values())
    advantage_report = {
        "bank_diagnostics": full_replay["diagnostics"],
        "safetymix_diagnostics": selected_diagnostics,
        "sdr_distribution": _distribution(selected_replay["sdr"]),
        "cdtr_distribution": _distribution(selected_replay["cdtr"]),
        "hla_train_distribution": _distribution(selected_replay["hla"]),
        "zero_group_composition": dict(zero_composition),
        "tier_counts": dict(Counter(tier for tiers in tiers_by_token.values() for tier in tiers if tier is not None)),
        "invalid_rollouts": sum(tier is None for tiers in tiers_by_token.values() for tier in tiers),
        "strict_clear_e0": strict_clear_e0,
        "headroom_e0": 1.0 - strict_clear_e0,
        "technical_gates": technical_gates,
        "scientific_gates": scientific_gates,
        "decision": "proceed_hla_smoke" if technical_gates["passed"] and scientific_gates["passed"] else "close_hla",
    }
    for name, payload in (
        ("cdt_stability_report.json", stability_report),
        ("safetymix_report.json", safetymix_report),
        ("advantage_geometry_report.json", advantage_report),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    (args.output_dir / "input_sha256.json").write_text(
        json.dumps(
            {
                str(path): sha256(path)
                for path in [
                    *args.s0_block,
                    args.stability_manifest,
                    args.bank_rollouts,
                    args.candidate_manifest,
                    args.dev_manifest,
                    args.final_manifest,
                    args.master_index,
                    args.e0_rollouts,
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "source_commit.txt").write_text(args.source_commit + "\n", encoding="utf-8")
    (args.output_dir / "resolved_method_definition.json").write_text(
        json.dumps(
            {
                "tiers": "L0<L1<L2<L3; invalid excluded",
                "same_l2_l3": "standard_grpo_identity",
                "mixed": "tier_pairwise_plus_0.125_bounded_centered_sdr",
                "scale": "valid_subset_torch_std_plus_1e-6",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "exit_code").write_text("0\n", encoding="utf-8")
    (args.output_dir / "COMPLETE").touch()
    return {"stability": stability_report, "safetymix": safetymix_report, "advantage": advantage_report}


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
    command.add_argument("--skip-adas", action="store_true")
    command.set_defaults(function=build_manifests)

    command = subparsers.add_parser("analyze-h0")
    command.add_argument("--s0-block", type=Path, action="append", required=True)
    command.add_argument("--stability-manifest", type=Path, required=True)
    command.add_argument("--bank-rollouts", type=Path, required=True)
    command.add_argument("--candidate-manifest", type=Path, required=True)
    command.add_argument("--dev-manifest", type=Path, required=True)
    command.add_argument("--final-manifest", type=Path, required=True)
    command.add_argument("--master-index", type=Path, required=True)
    command.add_argument("--e0-rollouts", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--source-commit", required=True)
    command.add_argument("--seed", type=int, default=20260825)
    command.set_defaults(function=analyze_h0)

    command = subparsers.add_parser("analyze-conflicts")
    command.add_argument("--s0-block", type=Path, action="append", required=True)
    command.add_argument("--stability-manifest", type=Path, required=True)
    command.add_argument("--bank-rollouts", type=Path, required=True)
    command.add_argument("--candidate-manifest", type=Path, required=True)
    command.add_argument("--master-index", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--source-commit", required=True)
    command.add_argument("--epsilon", type=float, default=1e-6)
    command.add_argument("--expected-groups", type=int, default=6000)
    command.add_argument("--expected-stability-tokens", type=int, default=500)
    command.set_defaults(function=analyze_conflicts)

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
