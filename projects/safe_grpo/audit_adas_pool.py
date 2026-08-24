#!/usr/bin/env python3
"""Audit whether the released ADAS token list defines a selective train-only pool."""

import argparse
import hashlib
import json
from pathlib import Path


def load_tokens(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Token file contains duplicates: {path}")
    return tokens


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(args: argparse.Namespace) -> dict:
    released = load_tokens(args.released_filter)
    train = load_tokens(args.train_manifest)
    dev = load_tokens(args.dev_manifest)
    heldout = load_tokens(args.heldout_manifest)
    random = load_tokens(args.random_manifest)

    train_set, dev_set, heldout_set = set(train), set(dev), set(heldout)
    if train_set & dev_set or train_set & heldout_set or dev_set & heldout_set:
        raise ValueError("Frozen train/dev/held-out manifests overlap.")
    released_set = set(released)
    known = train_set | dev_set | heldout_set
    eligible_train = released_set & train_set
    if not set(random) <= train_set or len(random) != 1000:
        raise ValueError("Random reference manifest is not a 1,000-token subset of train.")

    excluded_train = train_set - eligible_train
    enough_tokens = len(eligible_train) >= 1000
    selective = bool(excluded_train)
    gate_passed = enough_tokens and selective
    min_g4_diversity = 2 * (0.5**4)
    report = {
        "inputs": {
            "released_filter": str(args.released_filter),
            "released_filter_sha256": sha256(args.released_filter),
            "train_manifest_sha256": sha256(args.train_manifest),
            "dev_manifest_sha256": sha256(args.dev_manifest),
            "heldout_manifest_sha256": sha256(args.heldout_manifest),
            "random_manifest_sha256": sha256(args.random_manifest),
        },
        "coverage": {
            "released_tokens": len(released),
            "train_tokens": len(train),
            "dev_tokens": len(dev),
            "heldout_tokens": len(heldout),
            "released_in_train": len(released_set & train_set),
            "released_in_dev": len(released_set & dev_set),
            "released_in_heldout": len(released_set & heldout_set),
            "released_outside_frozen_splits": len(released_set - known),
            "train_tokens_excluded_by_released_gate": len(excluded_train),
            "eligible_train_ratio": len(eligible_train) / len(train),
        },
        "gates": {
            "eligible_pool_at_least_1000": enough_tokens,
            "released_gate_is_selective_within_train": selective,
            "passed": gate_passed,
        },
        "g4_bernoulli_boundary": {
            "n_rollout": 4,
            "minimum_p_to_n_plus_one_minus_p_to_n": min_g4_diversity,
            "historical_epsilon": 0.1,
            "any_scene_can_pass_strict_less_than_epsilon": min_g4_diversity < 0.1,
        },
        "interpretation": (
            "The released ADAS list admits every frozen train token, so sampling 1,000 eligible tokens would be "
            "Random sampling rather than an ADAS selector. Recomputing the historical Bernoulli gate with G=4 and "
            "epsilon=0.1 is mathematically empty."
        ),
        "manifest_written": False,
        "decision": "freeze_adas_and_hybrid_routes_as_undefined" if not gate_passed else "eligible_for_adas_sampling",
        "dev_or_heldout_model_inference": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--released-filter", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args)
    print(json.dumps({"gates": report["gates"], "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
