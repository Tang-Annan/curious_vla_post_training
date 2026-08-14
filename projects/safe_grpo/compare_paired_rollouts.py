import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "pdms_scaled",
    "pdms",
    "safe",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
)


def load_manifest(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Manifest contains duplicate tokens.")
    return tokens


def load_rows(path: Path, tokens: list[str]) -> dict[str, dict]:
    allowed = set(tokens)
    rows = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        token = str(row["token"])
        if token not in allowed:
            raise ValueError(f"Rollout contains token outside the manifest: {token}")
        if token in rows:
            raise ValueError(f"Rollout contains duplicate token: {token}")
        missing_metrics = [metric for metric in METRICS if metric not in row]
        if missing_metrics:
            raise ValueError(f"Rollout token {token} is missing metrics: {missing_metrics}")
        rows[token] = row
    missing_tokens = allowed - rows.keys()
    if missing_tokens:
        raise ValueError(f"Rollout is missing {len(missing_tokens)} manifest tokens.")
    return rows


def compare(
    baseline_path: Path,
    candidate_path: Path,
    manifest_path: Path,
    bootstrap_samples: int = 20000,
    seed: int = 20260814,
) -> dict:
    tokens = load_manifest(manifest_path)
    baseline = load_rows(baseline_path, tokens)
    candidate = load_rows(candidate_path, tokens)
    differences = np.asarray(
        [[float(candidate[token][metric]) - float(baseline[token][metric]) for token in tokens] for metric in METRICS]
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((len(METRICS), bootstrap_samples), dtype=float)
    chunk_size = 1000
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        indices = rng.integers(0, len(tokens), size=(stop - start, len(tokens)))
        bootstrap_means[:, start:stop] = differences[:, indices].mean(axis=2)

    metrics = {}
    for index, metric in enumerate(METRICS):
        baseline_values = np.asarray([float(baseline[token][metric]) for token in tokens])
        candidate_values = np.asarray([float(candidate[token][metric]) for token in tokens])
        low, high = np.quantile(bootstrap_means[index], [0.025, 0.975])
        metrics[metric] = {
            "baseline_mean": float(baseline_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "mean_difference": float(differences[index].mean()),
            "paired_bootstrap_95_ci": [float(low), float(high)],
        }
    return {
        "tokens": len(tokens),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "difference": "candidate_minus_baseline",
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.manifest, args.bootstrap_samples, args.seed)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
