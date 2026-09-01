from projects.dataset_v3.v4_post_training import (
    CANDIDATE_SAFETY_FIELDS,
    classify_contrast,
    slice_name_membership,
)


def delta(point: float, lower: float = -0.01, upper: float = 0.01) -> dict[str, float]:
    return {"point_delta": point, "ci_lower": lower, "ci_upper": upper}


def contrast_fixture() -> dict[str, dict[str, dict[str, float]]]:
    metrics = {
        "pdms_scaled": delta(0.02),
        "strict_clear": delta(0.01),
        **{field: delta(0.001) for field in CANDIDATE_SAFETY_FIELDS},
    }
    return {
        "risk": metrics,
        "control": {
            "pdms_scaled": delta(-0.005),
            **{field: delta(0.0) for field in CANDIDATE_SAFETY_FIELDS},
        },
        "all_dev": {"pdms_scaled": delta(0.001)},
    }


def test_slice_name_membership_uses_frozen_current_visible_labels() -> None:
    row = {"eval_tier1": "1", "response_complexity": "0", "split": "dev_natural"}
    assert slice_name_membership(row) == {
        "all_dev": True,
        "risk": True,
        "control": False,
        "response_complexity": False,
        "natural": True,
        "tail": False,
    }


def test_contrast_gate_distinguishes_direction_from_statistical_support() -> None:
    directional = classify_contrast(contrast_fixture(), require_risk_safety_gain=True)
    assert directional["status"] == "DIRECTIONAL_EXPLORATORY_PASS"
    assert directional["gates"]["risk_any_safety_positive"] is True

    supported_fixture = contrast_fixture()
    supported_fixture["risk"]["pdms_scaled"] = delta(0.02, 0.001, 0.03)
    supported_fixture["risk"]["strict_clear"] = delta(0.01, 0.001, 0.02)
    supported = classify_contrast(supported_fixture, require_risk_safety_gain=False)
    assert supported["status"] == "STATISTICALLY_SUPPORTED_IMPROVEMENT"


def test_contrast_gate_rejects_control_harm() -> None:
    harmful = contrast_fixture()
    harmful["control"]["pdms_scaled"] = delta(-0.02)
    result = classify_contrast(harmful, require_risk_safety_gain=False)
    assert result["status"] == "NO_IMPROVEMENT_GATE"
    assert result["gates"]["control_pdms_scaled_noninferior"] is False
