#!/usr/bin/env python3
"""Build NAVSIM metric caches for the frozen project manifest."""

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry, save_cache_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, list[str]]:
    tokens_by_log: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tokens_by_log[row["log_name"]].append(row["token"])
    return dict(tokens_by_log)


def existing_caches(output: Path) -> dict[str, Path]:
    return {path.parent.name: path for path in output.rglob("metric_cache.pkl")}


def write_metadata(output: Path, paths: dict[str, Path]) -> None:
    entries = [CacheMetadataEntry(paths[token]) for token in sorted(paths)]
    save_cache_metadata(entries, output, 0)


def cache_log(log_name: str, tokens: list[str], logs: Path, maps: Path, output: Path) -> dict[str, Path]:
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        log_names=[log_name],
        tokens=tokens,
    )
    loader = SceneLoader(
        data_path=logs,
        original_sensor_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    found = set(loader.tokens)
    if found != set(tokens):
        raise RuntimeError(f"{log_name}: missing manifest tokens {sorted(set(tokens) - found)}")

    processor = MetricCacheProcessor(
        cache_path=str(output),
        force_feature_computation=True,
        proposal_sampling=TrajectorySampling(num_poses=40, interval_length=0.1),
    )
    paths = {}
    for token in tokens:
        scene = loader.get_scene_from_token(token)
        scenario = NavSimScenario(scene, map_root=str(maps), map_version="nuplan-maps-v1.0")
        entry = processor.compute_and_save_metric_cache(scenario)
        if entry is None:
            raise RuntimeError(f"Failed to cache token {token}")
        paths[token] = Path(entry.file_name)
    return paths


def main() -> None:
    args = parse_args()
    tokens_by_log = load_manifest(args.manifest)
    expected = {token for tokens in tokens_by_log.values() for token in tokens}
    cached = existing_caches(args.output)
    unexpected = set(cached) - expected
    if unexpected:
        raise RuntimeError(f"Output contains {len(unexpected)} unexpected tokens")

    pending = [
        (log_name, [token for token in tokens if token not in cached])
        for log_name, tokens in tokens_by_log.items()
        if any(token not in cached for token in tokens)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(cache_log, log_name, tokens, args.logs, args.maps, args.output): log_name
            for log_name, tokens in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            cached.update(future.result())
            write_metadata(args.output, cached)
            print(f"logs={index}/{len(pending)} tokens={len(cached)}/{len(expected)}", flush=True)

    write_metadata(args.output, cached)
    if set(cached) != expected:
        raise RuntimeError(f"Coverage mismatch: cached={len(cached)} expected={len(expected)}")


if __name__ == "__main__":
    main()
