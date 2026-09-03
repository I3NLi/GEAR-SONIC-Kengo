"""Pure-Python kinematic contract for the 23-DoF Galaxea Kengo.

The Kengo training asset is intentionally not distributed with this repository.
This module contains only names and reorder indices so data conversion and unit
tests do not need to import Isaac Lab.
"""

from __future__ import annotations

# Isaac Lab exposes the merged-fixed-joint URDF in this breadth-first order.
KENGO_ISAACLAB_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

# The clean Kengo MJCF is depth-first: left arm, right arm, waist, then legs.
# The uploaded retargeted NPZ files use this exact order.
KENGO_MUJOCO_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def _joint_to_body(name: str) -> str:
    # Kengo's waist motor connects the root torso to a child named
    # ``pelvis_link``; it is the sole joint/body naming exception.
    if name == "waist_yaw_joint":
        return "pelvis_link"
    return name.removesuffix("_joint") + "_link"


KENGO_ISAACLAB_BODY_NAMES = [
    "torso_link",
    *[_joint_to_body(name) for name in KENGO_ISAACLAB_JOINT_NAMES],
]
KENGO_MUJOCO_BODY_NAMES = [
    "torso_link",
    *[_joint_to_body(name) for name in KENGO_MUJOCO_JOINT_NAMES],
]


def _indices_for_output(output_names: list[str], input_names: list[str]) -> list[int]:
    """Return indices such that ``input_values[result]`` has output order."""

    if len(input_names) != len(set(input_names)):
        raise ValueError("input names contain duplicates")
    if set(output_names) != set(input_names):
        raise ValueError("input/output names do not describe the same elements")
    return [input_names.index(name) for name in output_names]


# Naming follows the upstream SONIC convention: the list is used to index the
# source tensor and produce the destination order.
KENGO_ISAACLAB_TO_MUJOCO_DOF = _indices_for_output(
    KENGO_MUJOCO_JOINT_NAMES, KENGO_ISAACLAB_JOINT_NAMES
)
KENGO_MUJOCO_TO_ISAACLAB_DOF = _indices_for_output(
    KENGO_ISAACLAB_JOINT_NAMES, KENGO_MUJOCO_JOINT_NAMES
)
KENGO_ISAACLAB_TO_MUJOCO_BODY = _indices_for_output(
    KENGO_MUJOCO_BODY_NAMES, KENGO_ISAACLAB_BODY_NAMES
)
KENGO_MUJOCO_TO_ISAACLAB_BODY = _indices_for_output(
    KENGO_ISAACLAB_BODY_NAMES, KENGO_MUJOCO_BODY_NAMES
)

KENGO_LOWER_JOINT_NAMES = [
    name
    for name in KENGO_MUJOCO_JOINT_NAMES
    if any(part in name for part in ("_hip_", "_knee_", "_ankle_"))
]
KENGO_LOWER_JOINT_INDICES_MUJOCO = [
    KENGO_MUJOCO_JOINT_NAMES.index(name) for name in KENGO_LOWER_JOINT_NAMES
]
KENGO_WRIST_JOINT_INDICES_MUJOCO = [
    KENGO_MUJOCO_JOINT_NAMES.index("left_wrist_roll_joint"),
    KENGO_MUJOCO_JOINT_NAMES.index("right_wrist_roll_joint"),
]

KENGO_ISAACLAB_TO_MUJOCO_MAPPING = {
    # Upstream calls this field ``isaaclab_joints`` although it contains body
    # names (root plus one body per actuated joint).
    "isaaclab_joints": KENGO_ISAACLAB_BODY_NAMES,
    "isaaclab_to_mujoco_dof": KENGO_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": KENGO_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": KENGO_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": KENGO_MUJOCO_TO_ISAACLAB_BODY,
    "lower_joint_indices_mujoco": KENGO_LOWER_JOINT_INDICES_MUJOCO,
    "wrist_mujoco_dof_indices": KENGO_WRIST_JOINT_INDICES_MUJOCO,
}


assert len(KENGO_ISAACLAB_JOINT_NAMES) == 23
assert len(KENGO_MUJOCO_BODY_NAMES) == 24
assert len(KENGO_LOWER_JOINT_INDICES_MUJOCO) == 12
assert sorted(KENGO_ISAACLAB_TO_MUJOCO_DOF) == list(range(23))
assert sorted(KENGO_MUJOCO_TO_ISAACLAB_DOF) == list(range(23))
assert sorted(KENGO_ISAACLAB_TO_MUJOCO_BODY) == list(range(24))
assert sorted(KENGO_MUJOCO_TO_ISAACLAB_BODY) == list(range(24))
