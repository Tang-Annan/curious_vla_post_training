import pytest

from projects.safe_preference.build_final_manifest import log_name_from_image, select_rows


def row(token: str, log_name: str = "log-a") -> dict:
    return {
        "answer": {"token": token},
        "images": [f"navsim/trainval_sensor_blobs/trainval/{log_name}/CAM_F0/{token}.jpg"],
    }


def test_select_rows_is_deterministic_and_excludes_old_splits() -> None:
    rows = [row(f"token-{index}") for index in range(10)]

    first = select_rows(rows, {"token-2", "token-7"}, count=5, salt="frozen")
    second = select_rows(reversed(rows), {"token-2", "token-7"}, count=5, salt="frozen")

    assert first == second
    assert len({item["token"] for item in first}) == 5
    assert not ({item["token"] for item in first} & {"token-2", "token-7"})


def test_select_rows_rejects_duplicate_source_tokens() -> None:
    with pytest.raises(ValueError, match="Duplicate token"):
        select_rows([row("same"), row("same")], set(), count=1, salt="frozen")


def test_select_rows_requires_requested_budget() -> None:
    with pytest.raises(ValueError, match="eligible tokens"):
        select_rows([row("one")], {"one"}, count=1, salt="frozen")


def test_log_name_requires_single_view_layout() -> None:
    assert log_name_from_image("root/log-a/CAM_F0/image.jpg") == "log-a"
    with pytest.raises(ValueError, match="Unexpected single-view"):
        log_name_from_image("root/log-a/CAM_L0/image.jpg")
