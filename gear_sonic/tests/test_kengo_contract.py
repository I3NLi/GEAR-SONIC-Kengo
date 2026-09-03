from __future__ import annotations

import torch

from gear_sonic.trl.utils.order_converter import KengoConverter, get_converter
from gear_sonic.utils.kengo_contract import (
    KENGO_ISAACLAB_BODY_NAMES,
    KENGO_ISAACLAB_JOINT_NAMES,
    KENGO_ISAACLAB_TO_MUJOCO_BODY,
    KENGO_ISAACLAB_TO_MUJOCO_DOF,
    KENGO_LOWER_JOINT_INDICES_MUJOCO,
    KENGO_MUJOCO_BODY_NAMES,
    KENGO_MUJOCO_JOINT_NAMES,
    KENGO_MUJOCO_TO_ISAACLAB_BODY,
    KENGO_MUJOCO_TO_ISAACLAB_DOF,
    KENGO_WRIST_JOINT_INDICES_MUJOCO,
)


def test_kengo_contract_has_exact_root_pelvis_and_limb_subsets() -> None:
    assert len(KENGO_ISAACLAB_JOINT_NAMES) == len(KENGO_MUJOCO_JOINT_NAMES) == 23
    assert len(KENGO_ISAACLAB_BODY_NAMES) == len(KENGO_MUJOCO_BODY_NAMES) == 24
    assert KENGO_ISAACLAB_BODY_NAMES[0] == KENGO_MUJOCO_BODY_NAMES[0] == "torso_link"
    assert KENGO_ISAACLAB_BODY_NAMES[3] == "pelvis_link"
    assert KENGO_MUJOCO_BODY_NAMES[11] == "pelvis_link"
    assert "waist_yaw_link" not in KENGO_ISAACLAB_BODY_NAMES
    assert KENGO_LOWER_JOINT_INDICES_MUJOCO == list(range(11, 23))
    assert KENGO_WRIST_JOINT_INDICES_MUJOCO == [4, 9]


def test_kengo_joint_and_body_mappings_are_exact_inverses() -> None:
    isaac_dof = torch.arange(23)
    mujoco_dof = isaac_dof[KENGO_ISAACLAB_TO_MUJOCO_DOF]
    torch.testing.assert_close(mujoco_dof[KENGO_MUJOCO_TO_ISAACLAB_DOF], isaac_dof)

    isaac_body = torch.arange(24)
    mujoco_body = isaac_body[KENGO_ISAACLAB_TO_MUJOCO_BODY]
    torch.testing.assert_close(mujoco_body[KENGO_MUJOCO_TO_ISAACLAB_BODY], isaac_body)


def test_kengo_order_converter_round_trips_dof_and_qpos() -> None:
    converter = KengoConverter()
    assert isinstance(get_converter("galaxea_kengo"), KengoConverter)

    dof = torch.arange(46, dtype=torch.float32).reshape(2, 23)
    torch.testing.assert_close(converter.to_isaaclab(converter.to_mujoco(dof)), dof)

    qpos = torch.arange(60, dtype=torch.float32).reshape(2, 30)
    round_trip = converter.to_isaaclab(converter.to_mujoco(qpos))
    torch.testing.assert_close(round_trip, qpos)
    torch.testing.assert_close(converter.to_mujoco(qpos)[..., :7], qpos[..., :7])

    mapping = converter.get_isaaclab_to_mujoco_mapping()
    assert mapping["lower_joint_indices_mujoco"] == list(range(11, 23))
    assert mapping["wrist_mujoco_dof_indices"] == [4, 9]
