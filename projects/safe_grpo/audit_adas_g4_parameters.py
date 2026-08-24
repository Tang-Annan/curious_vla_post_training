#!/usr/bin/env python3
"""Select one preregistered G4 ADAS parameter set using frozen train-only scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ADAS_DIR = Path(__file__).resolve().parents[2] / "EasyR1" / "scripts" / "adas"
sys.path.insert(0, str(ADAS_DIR))
from filter_dynamic import filter_groups  # noqa: E402


SEED = 20260812
GROUP_SIZE = 4
STD_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 0.10
MAX_POOL_RATIO = 0.80
CANDIDATES = (("balanced_2_of_4", 0.20), ("any_mixed_1_to_3_of_4", 0.35))


def load_tokens(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Token file contains duplicates: {path}")
    return tokens


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_group_stats(scores_path: Path, train_tokens: list[str]) -> pd.DataFrame:
    scores = pd.read_csv(scores_path, dtype={"token": str})
    required = {"token", "pdms", "pdms_scaled"}
    if not required <= set(scores.columns):
        raise ValueError(f"ADAS scores are missing columns: {sorted(required - set(scores.columns))}")
    scores = scores[["token", "pdms", "pdms_scaled"]].copy()
    scores["pdms"] = pd.to_numeric(scores["pdms"], errors="raise")
    scores["pdms_scaled"] = pd.to_numeric(scores["pdms_scaled"], errors="raise")
    if not np.isfinite(scores[["pdms", "pdms_scaled"]].to_numpy()).all():
        raise ValueError("ADAS scores contain non-finite values.")

    counts = scores.groupby("token").size()
    train_set = set(train_tokens)
    if set(counts.index) != train_set:
        raise ValueError("ADAS score token coverage does not exactly match the frozen train manifest.")
    if not bool((counts == GROUP_SIZE).all()):
        raise ValueError("Every frozen train token must have exactly four G4 scores.")

    grouped = scores.groupby("token", sort=True).agg(
        group_size=("pdms", "size"),
        pdms_mean=("pdms", "mean"),
        pdms_std=("pdms", "std"),
        pdms_min=("pdms", "min"),
        pdms_max=("pdms", "max"),
        pdms_scaled_mean=("pdms_scaled", "mean"),
        pdms_scaled_std=("pdms_scaled", "std"),
        pdms_scaled_min=("pdms_scaled", "min"),
        pdms_scaled_max=("pdms_scaled", "max"),
    ).reset_index()
    grouped["pdms_range"] = grouped["pdms_max"] - grouped["pdms_min"]
    grouped["pdms_scaled_range"] = grouped["pdms_scaled_max"] - grouped["pdms_scaled_min"]
    return grouped


def signal_summary(grouped: pd.DataFrame, tokens: set[str]) -> dict[str, float | int | None]:
    selected = grouped[grouped["token"].isin(tokens)]
    if len(selected) != len(tokens):
        raise ValueError("Signal summary token coverage is incomplete.")
    if selected.empty:
        return {
            "tokens": 0,
            "pdms_zero_std_ratio": None,
            "pdms_scaled_zero_std_ratio": None,
            "pdms_std_mean": None,
            "pdms_scaled_std_mean": None,
            "pdms_headroom_mean": None,
            "pdms_scaled_headroom_mean": None,
        }
    raw_std = selected["pdms_std"].to_numpy(dtype=float)
    scaled_std = selected["pdms_scaled_std"].to_numpy(dtype=float)
    return {
        "tokens": len(selected),
        "pdms_zero_std_ratio": float(np.mean(raw_std == 0.0)),
        "pdms_scaled_zero_std_ratio": float(np.mean(scaled_std == 0.0)),
        "pdms_std_mean": float(np.mean(raw_std)),
        "pdms_scaled_std_mean": float(np.mean(scaled_std)),
        "pdms_headroom_mean": float(np.mean(selected["pdms_max"] - selected["pdms_mean"])),
        "pdms_scaled_headroom_mean": float(
            np.mean(selected["pdms_scaled_max"] - selected["pdms_scaled_mean"])
        ),
    }


def write_tokens(path: Path, tokens: list[str]) -> None:
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    train = load_tokens(args.train_manifest)
    dev = load_tokens(args.dev_manifest)
    heldout = load_tokens(args.heldout_manifest)
    random_tokens = load_tokens(args.random_manifest)
    train_set, dev_set, heldout_set = set(train), set(dev), set(heldout)
    if train_set & dev_set or train_set & heldout_set or dev_set & heldout_set:
        raise ValueError("Frozen train/dev/held-out manifests overlap.")
    if len(random_tokens) != 1000 or not set(random_tokens) <= train_set:
        raise ValueError("Random reference must contain 1,000 frozen train tokens.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        args.output_dir / "p0_group_stats.csv",
        args.output_dir / "p0_report.json",
        args.output_dir / "eligible_pool.txt",
        args.output_dir / "adas_g4_train_seed20260812_1000.txt",
    ]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("P0 output files already exist.")

    grouped = build_group_stats(args.adas_scores, train)
    grouped.to_csv(output_paths[0], index=False)
    random_summary = signal_summary(grouped, set(random_tokens))
    candidate_reports = []
    chosen: dict[str, Any] | None = None

    for label, epsilon in CANDIDATES:
        eligible = filter_groups(
            grouped,
            p=epsilon,
            n_rollout=GROUP_SIZE,
            group_size=GROUP_SIZE,
            std_threshold=STD_THRESHOLD,
            conf=CONFIDENCE_THRESHOLD,
        )
        eligible_tokens = sorted(str(token) for token in eligible["token"])
        pool_ratio = len(eligible_tokens) / len(train)
        finite = bool(
            np.isfinite(
                eligible[["p_est", "diversity_metric", "predicted_std", "confidence_error"]].to_numpy(dtype=float)
            ).all()
        )
        p_est_in_range = bool(eligible["p_est"].between(0.0, 1.0, inclusive="both").all())
        sampled_tokens: list[str] = []
        sampled_summary: dict[str, float | int | None] | None = None
        if len(eligible_tokens) >= 1000:
            sampled_tokens = sorted(random.Random(args.seed).sample(eligible_tokens, 1000))
            sampled_summary = signal_summary(grouped, set(sampled_tokens))

        gates = {
            "eligible_pool_at_least_1000": len(eligible_tokens) >= 1000,
            "eligible_pool_ratio_at_most_0_80": pool_ratio <= MAX_POOL_RATIO,
            "eligible_metrics_finite": finite,
            "eligible_p_est_is_probability": p_est_in_range,
            "selected_scaled_zero_std_below_random": bool(
                sampled_summary
                and sampled_summary["pdms_scaled_zero_std_ratio"]
                < random_summary["pdms_scaled_zero_std_ratio"]
            ),
            "selected_scaled_std_mean_above_random": bool(
                sampled_summary
                and sampled_summary["pdms_scaled_std_mean"] > random_summary["pdms_scaled_std_mean"]
            ),
        }
        passed = all(gates.values())
        report = {
            "label": label,
            "epsilon_diversity": epsilon,
            "eligible_tokens": len(eligible_tokens),
            "eligible_ratio": pool_ratio,
            "eligible_signal": signal_summary(grouped, set(eligible_tokens)),
            "selected_signal": sampled_summary,
            "gates": {**gates, "passed": passed},
        }
        candidate_reports.append(report)
        if chosen is None and passed:
            chosen = {**report, "eligible_token_values": eligible_tokens, "selected_token_values": sampled_tokens}

    manifest_written = chosen is not None
    chosen_parameters = None
    if chosen is not None:
        write_tokens(output_paths[2], chosen.pop("eligible_token_values"))
        write_tokens(output_paths[3], chosen.pop("selected_token_values"))
        chosen_parameters = {
            "label": chosen["label"],
            "epsilon_diversity": chosen["epsilon_diversity"],
            "n_rollout": GROUP_SIZE,
            "group_size": GROUP_SIZE,
            "std_threshold": STD_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "eligible_pool_sha256": sha256(output_paths[2]),
            "adas_manifest_sha256": sha256(output_paths[3]),
        }

    report = {
        "inputs": {
            "adas_scores": str(args.adas_scores),
            "adas_scores_sha256": sha256(args.adas_scores),
            "train_manifest_sha256": sha256(args.train_manifest),
            "dev_manifest_sha256": sha256(args.dev_manifest),
            "heldout_manifest_sha256": sha256(args.heldout_manifest),
            "random_manifest_sha256": sha256(args.random_manifest),
        },
        "protocol": {
            "seed": args.seed,
            "candidate_order": [dict(label=label, epsilon_diversity=epsilon) for label, epsilon in CANDIDATES],
            "n_rollout": GROUP_SIZE,
            "group_size": GROUP_SIZE,
            "std_threshold": STD_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "minimum_pool_tokens": 1000,
            "maximum_pool_ratio": MAX_POOL_RATIO,
            "selection_rule": "first candidate in preregistered strict-to-broad order passing every gate",
        },
        "coverage": {"train_tokens": len(train), "score_rows": len(train) * GROUP_SIZE},
        "random_reference_signal": random_summary,
        "candidates": candidate_reports,
        "chosen_parameters": chosen_parameters,
        "manifest_written": manifest_written,
        "decision": "freeze_adas_g4_manifest_for_a4_sdr" if manifest_written else "close_adas_route_no_valid_g4_parameters",
        "dev_or_heldout_model_inference": False,
    }
    output_paths[1].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adas-scores", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = audit(args)
    print(json.dumps({"decision": report["decision"], "chosen_parameters": report["chosen_parameters"]}))


if __name__ == "__main__":
    main()
