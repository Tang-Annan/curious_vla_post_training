from projects.dataset_v3.data_prep import SftRow, assign_eval_logs, build_problem, choose_training_rows


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
    assert "t-0: (0.0, 0.0, 0.0)" in problem
    assert "optimal future 4-second trajectory" in problem
    assert "optimal future 5-second trajectory" not in problem
