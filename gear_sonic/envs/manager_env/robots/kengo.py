"""Isaac Lab articulation and ordering contract for Galaxea Kengo."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from gear_sonic.utils.kengo_contract import (
    KENGO_ISAACLAB_BODY_NAMES,
    KENGO_ISAACLAB_JOINT_NAMES,
    KENGO_ISAACLAB_TO_MUJOCO_BODY,
    KENGO_ISAACLAB_TO_MUJOCO_DOF,
    KENGO_ISAACLAB_TO_MUJOCO_MAPPING,
    KENGO_MUJOCO_TO_ISAACLAB_BODY,
    KENGO_MUJOCO_TO_ISAACLAB_DOF,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_URDF = (
    REPO_ROOT
    / "external_dependencies"
    / "kengo_robot_description"
    / "urdf"
    / "kengo_with_fist.urdf"
)
KENGO_URDF = (
    Path(os.environ.get("KENGO_SONIC_URDF", DEFAULT_URDF)).expanduser().resolve()
)

# Alias retained for the upstream order-converter naming convention.  This is
# a body list (root + links), despite the historical ``JOINTS`` name.
KENGO_ISAACLAB_JOINTS = KENGO_ISAACLAB_BODY_NAMES

KENGO_DEFAULT_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "waist_yaw_joint": 0.0,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_roll_joint": -0.16,
    "left_hip_pitch_joint": -0.26,
    "right_hip_pitch_joint": -0.26,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_elbow_joint": 0.87,
    "right_elbow_joint": 0.87,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_knee_joint": 0.54,
    "right_knee_joint": 0.54,
    "left_ankle_pitch_joint": -0.27,
    "right_ankle_pitch_joint": -0.27,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
}


KENGO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        asset_path=str(KENGO_URDF),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=KENGO_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_and_waist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_.*_joint",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=12.0,
            stiffness=105.33,
            damping=6.71,
            armature=0.02668,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=72.0,
            velocity_limit_sim=36.0,
            stiffness=44.42,
            damping=2.82,
            armature=0.01125,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit_sim=36.0,
            velocity_limit_sim=18.0,
            stiffness=22.21,
            damping=1.41,
            armature=0.005625,
        ),
    },
)

# One normalized action unit corresponds to one quarter of the maximum
# position error that can be sustained at the configured torque limit.
KENGO_ACTION_SCALE = {
    ".*_hip_yaw_joint": 0.25 * 120.0 / 105.33,
    ".*_hip_roll_joint": 0.25 * 120.0 / 105.33,
    ".*_hip_pitch_joint": 0.25 * 120.0 / 105.33,
    ".*_knee_joint": 0.25 * 120.0 / 105.33,
    "waist_.*_joint": 0.25 * 120.0 / 105.33,
    ".*_ankle_pitch_joint": 0.25 * 72.0 / 44.42,
    ".*_ankle_roll_joint": 0.25 * 72.0 / 44.42,
    ".*_shoulder_pitch_joint": 0.25 * 36.0 / 22.21,
    ".*_shoulder_roll_joint": 0.25 * 36.0 / 22.21,
    ".*_shoulder_yaw_joint": 0.25 * 36.0 / 22.21,
    ".*_elbow_joint": 0.25 * 36.0 / 22.21,
    ".*_wrist_.*_joint": 0.25 * 36.0 / 22.21,
}


assert set(KENGO_DEFAULT_JOINT_POS) == set(KENGO_ISAACLAB_JOINT_NAMES)
