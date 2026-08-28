from argparse import Namespace
import io
from pathlib import Path
import struct
import zipfile
import zlib

import pytest

from projects.dataset_v3.data_prep import (
    SftRow,
    assign_eval_logs,
    build_cache,
    build_problem,
    choose_training_rows,
    decode_zip_member,
)


def test_training_rows_are_deterministic_and_log_disjoint() -> None:
    rows = [
        SftRow(
            source_row=log_index * 9 + row_index,
            token=f"token-{log_index}-{row_index}",
            log_name=f"log-{log_index}",
            intent=("straight", "left", "right")[row_index % 3],
            source_image=f"navsim/trainval_sensor_blobs/trainval/log-{log_index}/CAM_F0/{row_index}.jpg",
        )
        for log_index in range(8)
        for row_index in range(9)
    ]

    first = choose_training_rows(rows, screen_size=24, monitor_size=4, random_size=9)
    second = choose_training_rows(rows, screen_size=24, monitor_size=4, random_size=9)
    screen, monitor, random_rows = first

    assert first == second
    assert len(screen) == 24
    assert len(monitor) == 4
    assert len(random_rows) == 9
    assert {row.log_name for row in screen}.isdisjoint(row.log_name for row in monitor)
    assert {row.token for row in random_rows} <= {row.token for row in screen}


def test_eval_assignment_reserves_every_log_and_balances_each_family() -> None:
    summary = {
        f"log-{index}": {
            "eligible_scenes": 20 - index if index < 6 else 0,
            "interaction_scenes": 12 - index if index < 6 else 0,
            "interaction_rate": (12 - index) / (20 - index) if index < 6 else 0.0,
            "minimum_actor_distance_m": 2.0 + index if index < 6 else None,
        }
        for index in range(10)
    }

    assignment = assign_eval_logs(summary)

    assert set(assignment) == set(summary)
    assert set(assignment.values()) == {"dev_natural", "dev_tail", "final_natural", "final_tail"}
    assert sum(split.endswith("tail") for split in assignment.values()) == 3
    for family in ("natural", "tail"):
        scene_counts = {
            destination: sum(
                summary[log_name]["eligible_scenes"]
                for log_name, split in assignment.items()
                if split == f"{destination}_{family}"
            )
            for destination in ("dev", "final")
        }
        assert abs(scene_counts["dev"] - scene_counts["final"]) <= 20


def test_problem_uses_relative_history_and_four_second_prompt() -> None:
    window = []
    for index in range(14):
        window.append(
            {
                "ego2global_translation": [float(index), 0.0, 0.0],
                "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
                "driving_command": [0, 1, 0, 0],
            }
        )
    template = "prefix\nEach trajectory point format: (x:float, y:float, heading:float)\noptimal future 5-second trajectory"

    problem = build_problem(window, template)

    assert "Current high-level intent (string): go straight" in problem
    assert "t-3: (-3.00, 0.0, 0.0)" in problem
    assert "0.0)    - t-2:" in problem
    assert "t-0: (0.0, 0.0, 0.0)" in problem
    assert "optimal future 4-second trajectory" in problem
    assert "optimal future 5-second trajectory" not in problem

    window[3]["driving_command"] = [0, 0, 0, 1]
    assert "Current high-level intent (string): unknown" in build_problem(window, template)


def test_cache_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="workers must be at least 1"):
        build_cache(Namespace(workers=0))


def test_decode_zip_member_checks_deflate_and_crc() -> None:
    payload = b"verified selective ZIP payload"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("trainval/log/CAM_F0/image.jpg", payload)
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        info = archive.getinfo("trainval/log/CAM_F0/image.jpg")

    member_range = buffer.getvalue()[info.header_offset :]
    assert decode_zip_member(info, member_range) == payload

    damaged = bytearray(member_range)
    name_length, extra_length = struct.unpack_from("<HH", damaged, 26)
    damaged[30 + name_length + extra_length] ^= 1
    with pytest.raises((ValueError, zlib.error)):
        decode_zip_member(info, bytes(damaged))
