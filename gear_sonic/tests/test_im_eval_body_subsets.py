"""Unit tests for embodiment-neutral imitation-evaluation body selection."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest


def _load_resolver_without_training_dependencies():
    """Load the callback with tiny stubs instead of Torch/Transformers/W&B."""

    def no_grad_stub():
        return lambda function: function

    def tqdm_stub_function(*args, **kwargs):
        del args, kwargs
        return None

    torch_stub = ModuleType("torch")
    torch_stub.no_grad = no_grad_stub

    transformers_stub = ModuleType("transformers")
    transformers_stub.TrainerCallback = object

    tqdm_stub = ModuleType("tqdm")
    tqdm_stub.tqdm = tqdm_stub_function

    callback_path = (
        Path(__file__).resolve().parents[1]
        / "trl"
        / "callbacks"
        / "im_eval_callback.py"
    )
    spec = spec_from_file_location("_im_eval_callback_body_subset_test", callback_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "torch": torch_stub,
            "transformers": transformers_stub,
            "tqdm": tqdm_stub,
            "wandb": ModuleType("wandb"),
        },
    ):
        spec.loader.exec_module(module)
    return module._resolve_imitation_metric_body_subsets


RESOLVE_SUBSETS = _load_resolver_without_training_dependencies()


G1_H2_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

KENGO_BODY_NAMES = [
    "torso_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "pelvis_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]


@pytest.mark.parametrize("embodiment", ["g1", "h2"])
def test_g1_and_h2_keep_historical_metric_body_order(embodiment):
    del embodiment
    subsets = RESOLVE_SUBSETS(
        G1_H2_BODY_NAMES,
        ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"],
    )

    assert subsets["vr_3points"] == (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    assert subsets["other_upper_bodies"][0] == "pelvis"
    assert sum(len(names) for key, names in subsets.items() if key != "feet") == 14


def test_kengo_uses_configured_roll_wrists_and_pelvis_link():
    subsets = RESOLVE_SUBSETS(
        KENGO_BODY_NAMES,
        ["left_wrist_roll_link", "right_wrist_roll_link", "torso_link"],
    )

    assert subsets["vr_3points"] == (
        "torso_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
    )
    assert subsets["other_upper_bodies"][0] == "pelvis_link"
    assert sum(len(names) for key, names in subsets.items() if key != "feet") == 14


def test_missing_semantic_body_has_actionable_error():
    with pytest.raises(ValueError, match="right_wrist"):
        RESOLVE_SUBSETS(
            [name for name in KENGO_BODY_NAMES if name != "right_wrist_roll_link"],
            ["left_wrist_roll_link", "right_wrist_roll_link", "torso_link"],
        )
