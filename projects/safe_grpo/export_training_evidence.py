"""Export interview-ready training curves and representative rollout samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = {
    "reward_training": ("reward", "overall"),
    "reward_pdms": ("reward", "pdms"),
    "reward_pdms_scaled": ("reward", "pdms_scaled"),
    "reward_safe": ("reward", "safe"),
    "reward_collision": ("reward", "no_at_fault_collisions"),
    "reward_dac": ("reward", "drivable_area_compliance"),
    "reward_progress": ("reward", "ego_progress"),
    "reward_ttc": ("reward", "time_to_collision_within_bound"),
    "reward_comfort": ("reward", "history_comfort"),
    "reward_parse": ("reward", "parsed_ok"),
    "reward_latency_ms": ("reward", "reward_latency_ms"),
    "actor_pg_loss": ("actor", "pg_loss"),
    "actor_entropy_loss": ("actor", "entropy_loss"),
    "actor_kl_loss": ("actor", "kl_loss"),
    "actor_ppo_kl": ("actor", "ppo_kl"),
    "actor_clipfrac_higher": ("actor", "pg_clipfrac_higher"),
    "actor_clipfrac_lower": ("actor", "pg_clipfrac_lower"),
    "actor_grad_norm": ("actor", "grad_norm"),
    "actor_lr": ("actor", "lr"),
    "advantage_mean": ("critic", "advantages", "mean"),
    "advantage_min": ("critic", "advantages", "min"),
    "advantage_max": ("critic", "advantages", "max"),
    "response_length_mean": ("response_length", "mean"),
    "response_length_max": ("response_length", "max"),
    "response_clip_ratio": ("response_length", "clip_ratio"),
    "step_time_s": ("timing_s", "step"),
    "generation_time_s": ("timing_s", "gen"),
    "reward_time_s": ("timing_s", "reward"),
    "reference_time_s": ("timing_s", "ref"),
    "update_actor_time_s": ("timing_s", "update_actor"),
    "throughput_tokens_s": ("perf", "throughput"),
}

PANELS = (
    ("Train reward / safety", ("reward_training", "reward_pdms_scaled", "reward_safe")),
    ("Policy loss", ("actor_pg_loss",)),
    ("Entropy loss", ("actor_entropy_loss",)),
    ("KL / clip", ("actor_kl_loss", "actor_ppo_kl", "actor_clipfrac_higher", "actor_clipfrac_lower")),
    ("Gradient norm", ("actor_grad_norm",)),
    ("Response length", ("response_length_mean", "response_length_max")),
    ("Step timing", ("step_time_s", "generation_time_s", "update_actor_time_s")),
    ("GPU memory MiB", ("gpu_memory_used_mib",)),
)

COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed")


def _nested(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_history(path: Path) -> list[dict[str, float | int | None]]:
    history_by_step: dict[int, dict[str, float | int | None]] = {}
    previous_step = -1
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            step = int(row["step"])
            if step < previous_step:
                raise ValueError("Training steps must be increasing.")
            previous_step = step
            flattened = history_by_step.setdefault(step, {"step": step, **dict.fromkeys(METRICS)})
            for name, metric_path in METRICS.items():
                value = _nested(row, metric_path)
                if value is not None:
                    flattened[name] = value
    history = list(history_by_step.values())
    if not history:
        raise ValueError("Training history is empty.")
    return history


def load_gpu_history(path: Path) -> tuple[list[dict[str, float]], dict[str, float | int | None]]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                parsed = {key: float(value.strip()) for key, value in row.items() if key and value is not None}
            except ValueError:
                continue
            if all(math.isfinite(value) for value in parsed.values()):
                rows.append(parsed)
    if not rows:
        return rows, {"samples": 0, "peak_memory_used_mib": None, "minimum_memory_free_mib": None,
                      "peak_utilization_percent": None, "sampled_wall_seconds": None}
    return rows, {
        "samples": len(rows),
        "peak_memory_used_mib": max(row["memory_used_mib"] for row in rows),
        "minimum_memory_free_mib": min(row["memory_free_mib"] for row in rows),
        "peak_utilization_percent": max(row["utilization_percent"] for row in rows),
        "sampled_wall_seconds": max(row["timestamp"] for row in rows) - min(row["timestamp"] for row in rows),
    }


def metric_summary(history: list[dict[str, float | int | None]]) -> dict[str, dict[str, float | int | None]]:
    summary = {}
    for name in METRICS:
        values = [float(row[name]) for row in history if row[name] is not None]
        summary[name] = {
            "count": len(values),
            "missing": len(history) - len(values),
            "first": values[0] if values else None,
            "last": values[-1] if values else None,
            "mean": statistics.fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def _polyline(values: list[float], left: float, top: float, width: float, height: float,
              lower: float, upper: float) -> str:
    span = upper - lower or 1.0
    divisor = max(len(values) - 1, 1)
    points = []
    for index, value in enumerate(values):
        x = left + width * index / divisor
        y = top + height - height * (value - lower) / span
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def render_svg(history: list[dict[str, float | int | None]], gpu_rows: list[dict[str, float]], output: Path) -> None:
    width, height = 1200, 1280
    panel_width, panel_height = 540, 260
    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:18px;font-weight:700}.axis{font-size:12px;fill:#4b5563}.legend{font-size:11px}</style>',
    ]
    for panel_index, (title, names) in enumerate(PANELS):
        column, row = panel_index % 2, panel_index // 2
        x0, y0 = 45 + column * 590, 35 + row * 305
        left, top = x0 + 55, y0 + 35
        plot_width, plot_height = panel_width - 75, panel_height - 70
        series = []
        for name in names:
            if name == "gpu_memory_used_mib":
                values = [row["memory_used_mib"] for row in gpu_rows]
            else:
                values = [float(row[name]) for row in history if row[name] is not None]
            if values:
                series.append((name, values))
        fragments.extend([
            f'<text x="{x0}" y="{y0 + 18}" class="title">{html.escape(title)}</text>',
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#d1d5db"/>',
        ])
        if not series:
            fragments.append(f'<text x="{left + 10}" y="{top + 25}" class="axis">no data</text>')
            continue
        all_values = [value for _, values in series for value in values]
        lower, upper = min(all_values), max(all_values)
        padding = (upper - lower) * 0.05 or max(abs(upper), 1.0) * 0.05
        lower, upper = lower - padding, upper + padding
        fragments.extend([
            f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" class="axis">{upper:.4g}</text>',
            f'<text x="{left - 8}" y="{top + plot_height}" text-anchor="end" class="axis">{lower:.4g}</text>',
        ])
        for series_index, (name, values) in enumerate(series):
            color = COLORS[series_index % len(COLORS)]
            fragments.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{_polyline(values, left, top, plot_width, plot_height, lower, upper)}"/>'
            )
            legend_x = left + (series_index % 2) * 235
            legend_y = top + plot_height + 18 + (series_index // 2) * 14
            fragments.append(f'<text x="{legend_x}" y="{legend_y}" class="legend" fill="{color}">{html.escape(name)}</text>')
    fragments.append("</svg>")
    output.write_text("\n".join(fragments), encoding="utf-8")


def training_reward(row: dict[str, Any]) -> float:
    return float(row.get("training_reward", row.get("pdms_scaled", row.get("overall", 0.0))))


def export_samples(rollouts_path: Path, output_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines() if line]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["token"])].append(row)
    selected: list[tuple[str, dict[str, Any]]] = []
    selected.extend(("highest_reward", row) for row in sorted(rows, key=training_reward, reverse=True)[:5])
    selected.extend(("lowest_reward", row) for row in sorted(rows, key=training_reward)[:5])
    selected.extend(("longest_response", row) for row in sorted(rows, key=lambda item: int(item.get("response_length", 0)), reverse=True)[:5])
    selected.extend(("parse_failure", row) for row in [item for item in rows if not item.get("parsed_ok", False)][:5])
    diverse_groups = sorted(
        groups.values(), key=lambda group: max(map(training_reward, group)) - min(map(training_reward, group)), reverse=True
    )[:3]
    for group in diverse_groups:
        selected.extend(("largest_group_reward_gap", row) for row in group)
    with output_path.open("w", encoding="utf-8") as handle:
        for category, row in selected:
            handle.write(json.dumps({"evidence_category": category, **row}, ensure_ascii=False) + "\n")
    return {
        "source_rollouts": len(rows),
        "exported_rows": len(selected),
        "raw_response_available": any("response" in row for row in rows),
        "categories": sorted({category for category, _ in selected}),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(log_path: Path, gpu_path: Path, rollouts_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(log_path)
    gpu_rows, gpu_summary = load_gpu_history(gpu_path)
    history_csv = output_dir / "training_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *METRICS])
        writer.writeheader()
        writer.writerows(history)
    curves_svg = output_dir / "training_curves.svg"
    render_svg(history, gpu_rows, curves_svg)
    samples_path = output_dir / "representative_train_samples.jsonl"
    sample_summary = export_samples(rollouts_path, samples_path)
    summary = {
        "steps": len(history),
        "first_step": history[0]["step"],
        "last_step": history[-1]["step"],
        "metrics": metric_summary(history),
        "gpu": gpu_summary,
        "representative_samples": sample_summary,
    }
    summary_path = output_dir / "training_curve_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "inputs": {str(path): sha256(path) for path in (log_path, gpu_path, rollouts_path)},
        "outputs": {str(path): sha256(path) for path in (history_csv, curves_svg, samples_path, summary_path)},
    }
    (output_dir / "training_evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-log", type=Path, required=True)
    parser.add_argument("--gpu-memory", type=Path, required=True)
    parser.add_argument("--train-rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.experiment_log, args.gpu_memory, args.train_rollouts, args.output_dir)
    print(json.dumps({"steps": report["steps"], "last_step": report["last_step"]}))


if __name__ == "__main__":
    main()
