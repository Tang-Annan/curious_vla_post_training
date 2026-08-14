import argparse
import json
from pathlib import Path


def load_steps(path: Path, expected_steps: int, required_keys: tuple[str, ...]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [
        row
        for row in rows
        if 1 <= int(row["step"]) <= expected_steps and all(key in row for key in required_keys)
    ]
    by_step = {int(row["step"]): row for row in selected}
    if len(by_step) != len(selected):
        raise ValueError(f"Duplicate training steps in {path} after filtering for {required_keys}.")
    expected = set(range(1, expected_steps + 1))
    if set(by_step) != expected:
        missing = sorted(expected - set(by_step))
        raise ValueError(f"Missing expected training steps in {path}: {missing}")
    return [by_step[step] for step in sorted(by_step)]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def analyze(
    pilot_log: Path,
    parent_log: Path,
    expected_steps: int = 20,
    target_groups: int = 4,
    max_generation_batches: int = 5,
    max_mean_raw_overhead: float = 2.30,
    max_wall_time_ratio: float = 2.0,
) -> dict:
    pilot_rows = load_steps(pilot_log, expected_steps, ("sampling", "timing_s"))
    parent_rows = load_steps(parent_log, expected_steps, ("timing_s",))
    sampling = [row.get("sampling") for row in pilot_rows]
    if any(item is None for item in sampling):
        raise ValueError("Pilot log is missing structured sampling metrics.")

    used_groups = [int(item["used_groups"]) for item in sampling]
    generated_groups = [int(item["generated_groups"]) for item in sampling]
    kept_groups = [int(item["kept_groups"]) for item in sampling]
    dropped_groups = [int(item["dropped_groups"]) for item in sampling]
    generation_batches = [int(item["generation_batches"]) for item in sampling]
    raw_overhead = [float(item["raw_rollout_overhead"]) for item in sampling]
    pilot_step_time = [float(row["timing_s"]["step"]) for row in pilot_rows]
    parent_step_time = [float(row["timing_s"]["step"]) for row in parent_rows]

    mean_raw_overhead = sum(raw_overhead) / expected_steps
    pilot_wall_time = sum(pilot_step_time)
    parent_wall_time = sum(parent_step_time)
    wall_time_ratio = pilot_wall_time / parent_wall_time
    gates = {
        "all_steps_use_target_groups": all(value == target_groups for value in used_groups),
        "generation_batches_at_most_cap": max(generation_batches) <= max_generation_batches,
        "mean_raw_rollout_overhead_at_most_limit": mean_raw_overhead <= max_mean_raw_overhead,
        "wall_time_ratio_at_most_limit": wall_time_ratio <= max_wall_time_ratio,
    }
    gates["passed"] = all(gates.values())
    return {
        "expected_steps": expected_steps,
        "target_groups": target_groups,
        "max_generation_batches": max_generation_batches,
        "limits": {
            "mean_raw_rollout_overhead": max_mean_raw_overhead,
            "wall_time_ratio": max_wall_time_ratio,
        },
        "sampling": {
            "generated_groups": sum(generated_groups),
            "kept_groups": sum(kept_groups),
            "dropped_groups": sum(dropped_groups),
            "mean_raw_rollout_overhead": mean_raw_overhead,
            "max_generation_batches_observed": max(generation_batches),
            "mean_generation_batches": sum(generation_batches) / expected_steps,
        },
        "timing": {
            "pilot_total_step_seconds": pilot_wall_time,
            "parent_total_step_seconds": parent_wall_time,
            "wall_time_ratio": wall_time_ratio,
            "pilot_step_seconds_p50": quantile(pilot_step_time, 0.50),
            "pilot_step_seconds_p90": quantile(pilot_step_time, 0.90),
            "parent_step_seconds_p50": quantile(parent_step_time, 0.50),
            "parent_step_seconds_p90": quantile(parent_step_time, 0.90),
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-log", type=Path, required=True)
    parser.add_argument("--parent-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=20)
    parser.add_argument("--target-groups", type=int, default=4)
    parser.add_argument("--max-generation-batches", type=int, default=5)
    parser.add_argument("--max-mean-raw-overhead", type=float, default=2.30)
    parser.add_argument("--max-wall-time-ratio", type=float, default=2.0)
    args = parser.parse_args()
    report = analyze(
        args.pilot_log,
        args.parent_log,
        args.expected_steps,
        args.target_groups,
        args.max_generation_batches,
        args.max_mean_raw_overhead,
        args.max_wall_time_ratio,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
