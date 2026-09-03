import pytest

from gear_sonic.utils.export_contract import (
    combined_policy_onnx_name,
    normalize_export_tag,
)


@pytest.mark.parametrize(
    ("robot_type", "expected"),
    [
        ("kengo", "kengo"),
        ("galaxea_kengo", "kengo"),
        ("g1_model_12_dex", "g1"),
        ("unitree_h2", "h2"),
    ],
)
def test_normalize_export_tag(robot_type, expected):
    assert normalize_export_tag(robot_type) == expected


def test_kengo_combined_policy_uses_external_kengo_name():
    assert (
        combined_policy_onnx_name("model_step_007400.onnx", "galaxea_kengo")
        == "model_step_007400_kengo.onnx"
    )


@pytest.mark.parametrize("robot_type", ["", "../../kengo", "kengo policy"])
def test_export_tag_rejects_empty_or_unsafe_names(robot_type):
    with pytest.raises(ValueError):
        normalize_export_tag(robot_type)
