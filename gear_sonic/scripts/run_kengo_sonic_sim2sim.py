#!/usr/bin/env python3
"""Run a combined Kengo SONIC ONNX policy in MuJoCo.

This is a deliberately small sim-to-sim path for the 23-DoF Kengo policy.  It
does not use the legacy Unitree deployment bridge, DDS, TensorRT, Redis, or any of
the unrelated legacy Kengo policies.  The combined ONNX contract is::

    [future q (10 x 23), future qd (10 x 23), relative root rotation (10 x 6),
     gyro history (10 x 3), q history (10 x 23), qd history (10 x 23),
     previous normalized-action history (10 x 23), gravity history (10 x 3)]
        -> normalized joint action (23)

Run this script from the repository root.  ``mujoco`` and ``onnxruntime`` are
imported lazily, so the motion/observation contract can be unit-tested with
NumPy alone.  ``--policy`` runs ONNX Runtime locally.  Alternatively,
``--remote-policy-ssh`` plus ``--remote-policy-path`` keeps one non-interactive
SSH process connected to ``kengo_sonic_remote_onnx_server.py`` and therefore
does not require the ONNX file (or ONNX Runtime) on the MuJoCo machine.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.kengo_contract import (  # noqa: E402
    KENGO_ISAACLAB_JOINT_NAMES,
)


NUM_JOINTS = 23
POLICY_FPS = 50.0
SIM_DT = 0.005
POLICY_DECIMATION = 4
FUTURE_FRAMES = 10
FUTURE_STRIDE = 5
FUTURE_OFFSETS = np.arange(FUTURE_FRAMES, dtype=np.int64) * FUTURE_STRIDE
HISTORY_LENGTH = 10
FUTURE_JOINT_DIM = FUTURE_FRAMES * NUM_JOINTS
FUTURE_ORIENTATION_DIM = FUTURE_FRAMES * 6
TOKENIZER_DIM = 2 * FUTURE_JOINT_DIM + FUTURE_ORIENTATION_DIM
PROPRIOCEPTION_DIM = HISTORY_LENGTH * (3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS + 3)
POLICY_INPUT_DIM = TOKENIZER_DIM + PROPRIOCEPTION_DIM
POLICY_OUTPUT_DIM = NUM_JOINTS
POLICY_FLOAT_DTYPE = np.dtype("<f4")
POLICY_REQUEST_BYTES = POLICY_INPUT_DIM * POLICY_FLOAT_DTYPE.itemsize
POLICY_RESPONSE_BYTES = POLICY_OUTPUT_DIM * POLICY_FLOAT_DTYPE.itemsize
GRAVITY_W = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
IMU_ORIENTATION_SENSOR = "imu_quat"
IMU_ANGULAR_VELOCITY_SENSOR = "imu-torso-angular-velocity"
DEFAULT_REMOTE_PYTHON = "python3"
DEFAULT_REMOTE_SERVER = "gear_sonic/scripts/kengo_sonic_remote_onnx_server.py"

DEFAULT_XML = Path(
    os.environ.get(
        "KENGO_SONIC_MJCF",
        REPO_ROOT
        / "external_dependencies"
        / "kengo_robot_description"
        / "xml"
        / "kengo_with_fist.xml",
    )
)
DEFAULT_MOTION = (
    REPO_ROOT.parent
    / "assets"
    / "bones_lafan1_kengo_filtered_retargeted_20260626_manual-filtered_20260728"
    / "motions"
    / "230905"
    / "walk_arc_cw_loop_R_normal_pace_001__A444"
    / "walk_arc_cw_loop_R_normal_pace_001__A444_retargeted.npz"
)

_DEFAULT_JOINT_POS_BY_NAME = {
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
DEFAULT_JOINT_POS = np.asarray(
    [_DEFAULT_JOINT_POS_BY_NAME[name] for name in KENGO_ISAACLAB_JOINT_NAMES],
    dtype=np.float32,
)


def _gains_for_joint(name: str) -> tuple[float, float, float, float]:
    """Return training Kp, Kd, effort limit, and armature for one joint."""

    if "ankle" in name:
        return 44.42, 2.82, 72.0, 0.01125
    if any(part in name for part in ("hip_", "knee_", "waist_")):
        return 105.33, 6.71, 120.0, 0.02668
    return 22.21, 1.41, 36.0, 0.005625


_GAINS = np.asarray([_gains_for_joint(name) for name in KENGO_ISAACLAB_JOINT_NAMES])
KP = _GAINS[:, 0].astype(np.float32)
KD = _GAINS[:, 1].astype(np.float32)
EFFORT_LIMIT = _GAINS[:, 2].astype(np.float32)
ARMATURE = _GAINS[:, 3].astype(np.float32)
ACTION_SCALE = (0.25 * EFFORT_LIMIT / KP).astype(np.float32)


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.shape[-1] != 4 or not np.isfinite(values).all():
        raise ValueError(f"expected finite WXYZ quaternions, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-10):
        raise ValueError("quaternion array contains a zero-length quaternion")
    return values / norms


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = _normalize_quaternions(quaternions).copy()
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = _normalize_quaternions(np.asarray(quaternion))
    if quaternion.shape != (4,):
        raise ValueError(
            "inverse quaternion rotation expects one WXYZ quaternion, "
            f"got {quaternion.shape}"
        )
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"expected one finite 3-vector, got {vector.shape}")
    inverse = _quat_conjugate(quaternion)
    q_vector = inverse[1:]
    uv = np.cross(q_vector, vector)
    uuv = np.cross(q_vector, uv)
    return vector + 2.0 * (inverse[0] * uv + uuv)


def _quat_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    values = _normalize_quaternions(quaternions)
    w, x, y, z = np.moveaxis(values, -1, 0)
    entries = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    )
    return np.stack(entries, axis=-1).reshape(values.shape[:-1] + (3, 3))


def relative_rotation_6d(
    robot_quaternion_wxyz: np.ndarray, reference_quaternions_wxyz: np.ndarray
) -> np.ndarray:
    """Match SONIC's ``matrix_from_quat(inv(robot) * reference)[..., :2]``."""

    robot = _normalize_quaternions(np.asarray(robot_quaternion_wxyz, dtype=np.float64))
    references = _normalize_quaternions(reference_quaternions_wxyz)
    relative = _quat_multiply(_quat_conjugate(robot), references)
    matrices = _quat_to_matrix(relative)
    return matrices[..., :2].reshape(-1, 6).astype(np.float32)


def _slerp_sample(
    quaternions: np.ndarray, sample_times: np.ndarray, source_fps: float
) -> np.ndarray:
    source = _continuous_quaternions(quaternions)
    source_indices = np.minimum(sample_times * source_fps, len(source) - 1)
    lower = np.floor(source_indices).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    alpha = source_indices - lower
    q0 = source[lower]
    q1 = source[upper].copy()
    dot = np.sum(q0 * q1, axis=-1)
    negative = dot < 0.0
    q1[negative] *= -1.0
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    result = np.empty_like(q0)
    close = dot > 0.9995
    result[close] = q0[close] + alpha[close, None] * (q1[close] - q0[close])
    far = ~close
    if np.any(far):
        theta = np.arccos(dot[far])
        sin_theta = np.sin(theta)
        weight0 = np.sin((1.0 - alpha[far]) * theta) / sin_theta
        weight1 = np.sin(alpha[far] * theta) / sin_theta
        result[far] = weight0[:, None] * q0[far] + weight1[:, None] * q1[far]
    return _continuous_quaternions(result)


def _linear_sample(
    values: np.ndarray, sample_times: np.ndarray, source_fps: float
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    source_indices = np.minimum(sample_times * source_fps, len(source) - 1)
    lower = np.floor(source_indices).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    alpha = (source_indices - lower).reshape((-1,) + (1,) * (source.ndim - 1))
    return source[lower] * (1.0 - alpha) + source[upper] * alpha


def _joint_angle_sample(
    values: np.ndarray, sample_times: np.ndarray, source_fps: float
) -> np.ndarray:
    """Interpolate 1-DoF angles along the shortest path, as quaternion SLERP does."""

    source = np.asarray(values, dtype=np.float64)
    source_indices = np.minimum(sample_times * source_fps, len(source) - 1)
    lower = np.floor(source_indices).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    alpha = (source_indices - lower)[:, None]
    delta = source[upper] - source[lower]
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return source[lower] + alpha * delta


def training_forward_difference(
    values: np.ndarray, fps: float = POLICY_FPS
) -> np.ndarray:
    """Reproduce SONIC FK's forward-difference DoF velocity convention.

    The upstream implementation appends ``dof_vel[-2:-1]``.  Consequently the
    final sample repeats the second-to-last finite difference when at least
    three frames are present; this function intentionally preserves that
    trained contract.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or len(values) < 2:
        raise ValueError(f"at least two frames are required, got {values.shape}")
    differences = np.diff(values, axis=0) * float(fps)
    tail_index = max(len(differences) - 2, 0)
    return np.concatenate(
        (differences, differences[tail_index : tail_index + 1]), axis=0
    )


def _angular_velocity_world(quaternions: np.ndarray, fps: float) -> np.ndarray:
    quaternions = _continuous_quaternions(quaternions)
    delta = _normalize_quaternions(
        _quat_multiply(quaternions[1:], _quat_conjugate(quaternions[:-1]))
    )
    delta[delta[:, 0] < 0.0] *= -1.0
    vector_norm = np.linalg.norm(delta[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[:, 0], 0.0, 1.0))
    axis = np.zeros_like(delta[:, 1:])
    valid = vector_norm > 1.0e-10
    axis[valid] = delta[valid, 1:] / vector_norm[valid, None]
    differences = axis * angle[:, None] * fps
    tail_index = max(len(differences) - 2, 0)
    return np.concatenate(
        (differences, differences[tail_index : tail_index + 1]), axis=0
    )


@dataclass(frozen=True)
class MotionClip:
    source_path: Path
    source_fps: float
    fps: float
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.num_frames - 1) / self.fps


def _decode_npz_names(
    raw_names: np.ndarray, *, field: str, source_path: Path
) -> list[str]:
    values = np.asarray(raw_names)
    if values.ndim != 1:
        raise ValueError(f"{source_path}: {field} must be one-dimensional")
    names: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            try:
                names.append(value.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError(f"{source_path}: {field} is not valid UTF-8") from exc
        elif isinstance(value, str):
            names.append(value)
        else:
            raise ValueError(f"{source_path}: {field} must contain strings")
    return names


def load_motion_npz(path: str | Path, target_fps: float = POLICY_FPS) -> MotionClip:
    """Load a retargeted Kengo NPZ and reproduce the training-time 50 Hz view."""

    source_path = Path(path).expanduser().resolve()
    target_fps = float(target_fps)
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"{source_path}: invalid target fps {target_fps}")
    with np.load(source_path, allow_pickle=False) as payload:
        fps_key = "framerate" if "framerate" in payload else "fps"
        required = {fps_key, "joint_names", "joint_pos"}
        missing = sorted(key for key in required if key not in payload)
        if missing:
            raise ValueError(f"{source_path}: missing fields {missing}")
        fps_values = np.asarray(payload[fps_key])
        if fps_values.size != 1:
            raise ValueError(
                f"{source_path}: {fps_key} must contain exactly one value, "
                f"got shape {fps_values.shape}"
            )
        source_fps = float(fps_values.reshape(-1)[0])
        joint_names = _decode_npz_names(
            payload["joint_names"], field="joint_names", source_path=source_path
        )
        raw_joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)

        root_pair = next(
            (
                (position_key, quaternion_key)
                for position_key, quaternion_key in (
                    ("base_pos_w", "base_quat_w"),
                    ("root_pos_w", "root_quat_w"),
                )
                if position_key in payload and quaternion_key in payload
            ),
            None,
        )
        if root_pair is not None:
            raw_root_pos = np.asarray(payload[root_pair[0]], dtype=np.float64)
            raw_root_quat = np.asarray(payload[root_pair[1]], dtype=np.float64)
        elif all(
            key in payload
            for key in ("body_names", "body_pos_w", "body_quat_w")
        ):
            body_names = _decode_npz_names(
                payload["body_names"], field="body_names", source_path=source_path
            )
            if len(body_names) != len(set(body_names)):
                raise ValueError(f"{source_path}: body_names contains duplicates")
            try:
                root_body_index = body_names.index("torso_link")
            except ValueError as exc:
                raise ValueError(
                    f"{source_path}: body_names does not contain Kengo root 'torso_link'"
                ) from exc
            raw_body_pos = np.asarray(payload["body_pos_w"], dtype=np.float64)
            raw_body_quat = np.asarray(payload["body_quat_w"], dtype=np.float64)
            if (
                raw_body_pos.ndim != 3
                or raw_body_quat.ndim != 3
                or raw_body_pos.shape[1:] != (len(body_names), 3)
                or raw_body_quat.shape[1:] != (len(body_names), 4)
            ):
                raise ValueError(
                    f"{source_path}: inconsistent body arrays: names={len(body_names)}, "
                    f"body_pos={raw_body_pos.shape}, body_quat={raw_body_quat.shape}"
                )
            raw_root_pos = raw_body_pos[:, root_body_index, :]
            raw_root_quat = raw_body_quat[:, root_body_index, :]
        else:
            raise ValueError(
                f"{source_path}: missing root pose fields; expected base_pos_w/base_quat_w, "
                "root_pos_w/root_quat_w, or body_names/body_pos_w/body_quat_w"
            )

    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"{source_path}: invalid source fps {source_fps}")
    if len(joint_names) != len(set(joint_names)):
        raise ValueError(f"{source_path}: joint_names contains duplicates")
    missing_joints = sorted(set(KENGO_ISAACLAB_JOINT_NAMES) - set(joint_names))
    extra_joints = sorted(set(joint_names) - set(KENGO_ISAACLAB_JOINT_NAMES))
    if missing_joints or extra_joints:
        raise ValueError(
            f"{source_path}: joint contract mismatch; missing={missing_joints}, extra={extra_joints}"
        )
    if raw_joint_pos.ndim != 2 or raw_joint_pos.shape[1] != NUM_JOINTS:
        raise ValueError(
            f"{source_path}: joint_pos must be (T, 23), got {raw_joint_pos.shape}"
        )
    frames = len(raw_joint_pos)
    if (
        frames < 3
        or raw_root_pos.shape != (frames, 3)
        or raw_root_quat.shape != (frames, 4)
    ):
        raise ValueError(
            f"{source_path}: inconsistent motion arrays: joints={raw_joint_pos.shape}, "
            f"root_pos={raw_root_pos.shape}, root_quat={raw_root_quat.shape}"
        )
    if not all(
        np.isfinite(values).all()
        for values in (raw_joint_pos, raw_root_pos, raw_root_quat)
    ):
        raise ValueError(f"{source_path}: motion contains non-finite values")

    isaac_indices = [joint_names.index(name) for name in KENGO_ISAACLAB_JOINT_NAMES]
    raw_joint_pos = raw_joint_pos[:, isaac_indices]
    raw_root_quat = _continuous_quaternions(raw_root_quat)

    if math.isclose(source_fps, target_fps, rel_tol=0.0, abs_tol=1.0e-6):
        joint_pos = raw_joint_pos.copy()
        root_pos = raw_root_pos.copy()
        root_quat = raw_root_quat.copy()
    else:
        duration = (frames - 1) / source_fps
        # Upstream uses torch.arange(0, duration, 1 / target_fps, float32).
        sample_times = np.arange(
            np.float32(0.0),
            np.float32(duration),
            np.float32(1.0 / target_fps),
            dtype=np.float32,
        ).astype(np.float64)
        if len(sample_times) < 3:
            raise ValueError(
                f"{source_path}: resampled clip has only {len(sample_times)} frames"
            )
        joint_pos = _joint_angle_sample(raw_joint_pos, sample_times, source_fps)
        root_pos = _linear_sample(raw_root_pos, sample_times, source_fps)
        root_quat = _slerp_sample(raw_root_quat, sample_times, source_fps)

    joint_vel = training_forward_difference(joint_pos, target_fps)
    root_lin_vel = training_forward_difference(root_pos, target_fps)
    root_ang_vel = _angular_velocity_world(root_quat, target_fps)
    arrays = (joint_pos, joint_vel, root_pos, root_quat, root_lin_vel, root_ang_vel)
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError(f"{source_path}: derived motion contains non-finite values")
    return MotionClip(
        source_path=source_path,
        source_fps=source_fps,
        fps=float(target_fps),
        root_pos_w=root_pos.astype(np.float32),
        root_quat_w=root_quat.astype(np.float32),
        root_lin_vel_w=root_lin_vel.astype(np.float32),
        root_ang_vel_w=root_ang_vel.astype(np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
    )


class MotionReference:
    def __init__(self, clip: MotionClip, *, start_frame: int = 0, loop: bool = False):
        if not 0 <= start_frame < clip.num_frames:
            raise ValueError(
                f"start frame {start_frame} is outside [0, {clip.num_frames})"
            )
        self.clip = clip
        self.start_frame = int(start_frame)
        self.frame = int(start_frame)
        self.loop = bool(loop)
        self.anchor_xy = clip.root_pos_w[start_frame, :2].copy()

    def _index(self, index: np.ndarray | int) -> np.ndarray:
        values = np.asarray(index, dtype=np.int64)
        if self.loop:
            return values % self.clip.num_frames
        return np.minimum(values, self.clip.num_frames - 1)

    def reset(self) -> None:
        self.frame = self.start_frame

    def initial_state(self) -> tuple[np.ndarray, ...]:
        index = self.frame
        root_pos = self.clip.root_pos_w[index].copy()
        root_pos[:2] -= self.anchor_xy
        return (
            root_pos,
            self.clip.root_quat_w[index].copy(),
            self.clip.root_lin_vel_w[index].copy(),
            self.clip.root_ang_vel_w[index].copy(),
            self.clip.joint_pos[index].copy(),
            self.clip.joint_vel[index].copy(),
        )

    def current_target(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = int(self._index(self.frame))
        root_pos = self.clip.root_pos_w[index].copy()
        root_pos[:2] -= self.anchor_xy
        return root_pos, self.clip.root_quat_w[index], self.clip.joint_pos[index]

    def future_observation(self, robot_quaternion_wxyz: np.ndarray) -> np.ndarray:
        indices = self._index(self.frame + FUTURE_OFFSETS)
        joint_pos = self.clip.joint_pos[indices].reshape(-1)
        joint_vel = self.clip.joint_vel[indices].reshape(-1)
        root_orientation = relative_rotation_6d(
            robot_quaternion_wxyz, self.clip.root_quat_w[indices]
        ).reshape(-1)
        result = np.concatenate((joint_pos, joint_vel, root_orientation)).astype(
            np.float32, copy=False
        )
        if result.shape != (TOKENIZER_DIM,):
            raise RuntimeError(f"future observation contract changed: {result.shape}")
        return result

    def advance(self) -> None:
        if self.loop:
            self.frame = (self.frame + 1) % self.clip.num_frames
        else:
            self.frame = min(self.frame + 1, self.clip.num_frames - 1)


def _full_reference_step_budget(
    num_frames: int, start_frame: int
) -> tuple[int, int]:
    """Return exact policy/physics step counts for one non-looping ref pass."""

    policy_steps = int(num_frames) - int(start_frame)
    if policy_steps <= 0:
        raise ValueError(
            f"full-reference frame budget must be positive, got {policy_steps}"
        )
    return policy_steps, policy_steps * POLICY_DECIMATION


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(
            f"{name} must be finite with shape {shape}, got {result.shape}"
        )
    return result


def assemble_sonic_observation(
    future_observation: np.ndarray,
    base_angular_velocity_history: np.ndarray,
    joint_position_history: np.ndarray,
    joint_velocity_history: np.ndarray,
    action_history: np.ndarray,
    gravity_history: np.ndarray,
) -> np.ndarray:
    """Flatten one combined Kengo SONIC input in the exported ONNX order."""

    future = _require_shape("future_observation", future_observation, (TOKENIZER_DIM,))
    gyro = _require_shape(
        "base_angular_velocity_history",
        base_angular_velocity_history,
        (HISTORY_LENGTH, 3),
    )
    joint_pos = _require_shape(
        "joint_position_history", joint_position_history, (HISTORY_LENGTH, NUM_JOINTS)
    )
    joint_vel = _require_shape(
        "joint_velocity_history", joint_velocity_history, (HISTORY_LENGTH, NUM_JOINTS)
    )
    actions = _require_shape(
        "action_history", action_history, (HISTORY_LENGTH, NUM_JOINTS)
    )
    gravity = _require_shape("gravity_history", gravity_history, (HISTORY_LENGTH, 3))
    result = np.concatenate(
        (
            future,
            gyro.reshape(-1),
            joint_pos.reshape(-1),
            joint_vel.reshape(-1),
            actions.reshape(-1),
            gravity.reshape(-1),
        )
    ).astype(np.float32, copy=False)
    if result.shape != (POLICY_INPUT_DIM,):
        raise RuntimeError(f"SONIC observation contract changed: {result.shape}")
    return result


def _append_history(history: np.ndarray, value: np.ndarray) -> None:
    history[:-1] = history[1:]
    history[-1] = value


class SonicHistory:
    """Oldest-to-newest policy history matching Isaac Lab and SONIC deploy."""

    def __init__(self) -> None:
        self.base_angular_velocity = np.zeros((HISTORY_LENGTH, 3), dtype=np.float32)
        self.joint_position = np.zeros((HISTORY_LENGTH, NUM_JOINTS), dtype=np.float32)
        self.joint_velocity = np.zeros((HISTORY_LENGTH, NUM_JOINTS), dtype=np.float32)
        self.action = np.zeros((HISTORY_LENGTH, NUM_JOINTS), dtype=np.float32)
        self.gravity = np.zeros((HISTORY_LENGTH, 3), dtype=np.float32)
        self.initialized = False

    def reset(self) -> None:
        for history in (
            self.base_angular_velocity,
            self.joint_position,
            self.joint_velocity,
            self.action,
            self.gravity,
        ):
            history.fill(0.0)
        self.initialized = False

    def build(
        self,
        future_observation: np.ndarray,
        base_angular_velocity: np.ndarray,
        joint_position_relative: np.ndarray,
        joint_velocity: np.ndarray,
        gravity: np.ndarray,
    ) -> np.ndarray:
        gyro = _require_shape("base_angular_velocity", base_angular_velocity, (3,))
        joint_pos = _require_shape(
            "joint_position_relative", joint_position_relative, (NUM_JOINTS,)
        )
        joint_vel = _require_shape("joint_velocity", joint_velocity, (NUM_JOINTS,))
        gravity = _require_shape("gravity", gravity, (3,))
        if self.initialized:
            _append_history(self.base_angular_velocity, gyro)
            _append_history(self.joint_position, joint_pos)
            _append_history(self.joint_velocity, joint_vel)
            _append_history(self.gravity, gravity)
        else:
            self.base_angular_velocity[:] = gyro
            self.joint_position[:] = joint_pos
            self.joint_velocity[:] = joint_vel
            self.gravity[:] = gravity
            self.initialized = True
        return assemble_sonic_observation(
            future_observation,
            self.base_angular_velocity,
            self.joint_position,
            self.joint_velocity,
            self.action,
            self.gravity,
        )

    def record_action(self, normalized_action: np.ndarray) -> None:
        action = _require_shape("normalized_action", normalized_action, (NUM_JOINTS,))
        _append_history(self.action, action)


def _quaternion_angle(left: np.ndarray, right: np.ndarray) -> float:
    left = _normalize_quaternions(np.asarray(left))
    right = _normalize_quaternions(np.asarray(right))
    dot = float(np.clip(abs(np.sum(left * right)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
    w, x, y, z = _normalize_quaternions(np.asarray(quaternion))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _rms_from_sum(sum_squares: float, count: int) -> float | None:
    return math.sqrt(sum_squares / count) if count else None


@dataclass
class Metrics:
    fall_height: float
    finite: bool = True
    fall_time_s: float | None = None
    min_base_height_m: float = math.inf
    max_base_height_m: float = -math.inf
    max_abs_roll_rad: float = 0.0
    max_abs_pitch_rad: float = 0.0
    max_abs_joint_velocity_rad_s: float = 0.0
    joint_rmse: list[float] = field(default_factory=list)
    root_position_error: list[float] = field(default_factory=list)
    root_orientation_error: list[float] = field(default_factory=list)
    action_sum_squares: float = 0.0
    action_count: int = 0
    action_abs_max: float = 0.0
    action_clipped_components: int = 0
    torque_sum_squares: float = 0.0
    torque_count: int = 0
    torque_abs_max_nm: float = 0.0
    torque_saturated_components: int = 0
    inference_ms: list[float] = field(default_factory=list)

    def record_state(
        self,
        sim_time: float,
        root_position: np.ndarray,
        qvel: np.ndarray,
        torso_quaternion: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> None:
        arrays = (root_position, qvel, torso_quaternion, joint_velocity)
        if not all(np.isfinite(value).all() for value in arrays):
            self.finite = False
            return
        height = float(root_position[2])
        self.min_base_height_m = min(self.min_base_height_m, height)
        self.max_base_height_m = max(self.max_base_height_m, height)
        if self.fall_time_s is None and height < self.fall_height:
            self.fall_time_s = float(sim_time)
        roll, pitch = _roll_pitch(torso_quaternion)
        self.max_abs_roll_rad = max(self.max_abs_roll_rad, abs(roll))
        self.max_abs_pitch_rad = max(self.max_abs_pitch_rad, abs(pitch))
        self.max_abs_joint_velocity_rad_s = max(
            self.max_abs_joint_velocity_rad_s, float(np.max(np.abs(joint_velocity)))
        )

    def record_tracking(
        self,
        root_position: np.ndarray,
        root_quaternion: np.ndarray,
        joint_position: np.ndarray,
        target_root_position: np.ndarray,
        target_root_quaternion: np.ndarray,
        target_joint_position: np.ndarray,
    ) -> None:
        self.root_position_error.append(
            float(np.linalg.norm(root_position - target_root_position))
        )
        self.root_orientation_error.append(
            _quaternion_angle(root_quaternion, target_root_quaternion)
        )
        self.joint_rmse.append(
            float(np.sqrt(np.mean((joint_position - target_joint_position) ** 2)))
        )

    def record_action(
        self, raw_action: np.ndarray, action: np.ndarray, clip: float
    ) -> None:
        self.action_sum_squares += float(np.sum(np.square(action, dtype=np.float64)))
        self.action_count += int(action.size)
        self.action_abs_max = max(self.action_abs_max, float(np.max(np.abs(action))))
        self.action_clipped_components += int(
            np.count_nonzero(np.abs(raw_action) > clip)
        )

    def record_torque(self, raw_torque: np.ndarray, torque: np.ndarray) -> None:
        self.torque_sum_squares += float(np.sum(np.square(torque, dtype=np.float64)))
        self.torque_count += int(torque.size)
        self.torque_abs_max_nm = max(
            self.torque_abs_max_nm, float(np.max(np.abs(torque)))
        )
        self.torque_saturated_components += int(
            np.count_nonzero(np.abs(raw_torque) > EFFORT_LIMIT)
        )


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            received = size - remaining
            raise EOFError(f"received {received} of {size} response bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(stream: BinaryIO, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise BrokenPipeError("policy process accepted no request bytes")
        remaining = remaining[written:]
    stream.flush()


class BinaryFloat32PolicyClient:
    """Persistent fixed-size request/response process with safe replay.

    The protocol is stateless, so if a pipe fails the current observation can
    be sent to a newly-created process without changing policy state.
    """

    def __init__(
        self,
        command_factory: Callable[[], list[str]],
        *,
        label: str,
        process_factory: Callable[..., Any] = subprocess.Popen,
        max_reconnects: int = 3,
    ) -> None:
        if max_reconnects < 0:
            raise ValueError("max_reconnects must be non-negative")
        self._command_factory = command_factory
        self._label = label
        self._process_factory = process_factory
        self.max_reconnects = int(max_reconnects)
        self._process: Any | None = None
        self.starts = 0
        self.reconnects = 0

    def _start(self) -> Any:
        command = self._command_factory()
        if not command or not all(isinstance(part, str) and part for part in command):
            raise RuntimeError("policy process command contains an empty argument")
        process = self._process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Inherit stderr: it must never share the binary stdout protocol,
            # and inheritance also prevents a long-running server from filling
            # an unread diagnostics pipe.
            stderr=None,
            bufsize=0,
        )
        if process.stdin is None or process.stdout is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                process.terminate()
            raise RuntimeError("policy process did not expose binary stdin/stdout")
        self._process = process
        self.starts += 1
        return process

    def _ensure_process(self) -> Any:
        process = self._process
        if process is None:
            return self._start()
        return_code = process.poll()
        if return_code is not None:
            raise EOFError(f"policy process exited with status {return_code}")
        return process

    def _exchange(self, request: bytes) -> np.ndarray:
        process = self._ensure_process()
        _write_all(process.stdin, request)
        response = _read_exact(process.stdout, POLICY_RESPONSE_BYTES)
        action = np.frombuffer(response, dtype=POLICY_FLOAT_DTYPE).astype(
            np.float32, copy=True
        )
        if action.shape != (POLICY_OUTPUT_DIM,) or not np.isfinite(action).all():
            raise RuntimeError(
                f"remote policy returned invalid action shape/value: {action.shape}, "
                f"finite={np.isfinite(action).all()}"
            )
        return action

    def infer(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (POLICY_INPUT_DIM,) or not np.isfinite(values).all():
            raise ValueError(
                f"policy observation must be finite ({POLICY_INPUT_DIM},), got {values.shape}"
            )
        request = values.astype(POLICY_FLOAT_DTYPE, copy=False).tobytes(order="C")
        if len(request) != POLICY_REQUEST_BYTES:
            raise RuntimeError(f"policy request contract changed: {len(request)} bytes")

        failures: list[str] = []
        for attempt in range(self.max_reconnects + 1):
            try:
                return self._exchange(request)
            except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                self._discard_process()
                if attempt < self.max_reconnects:
                    self.reconnects += 1
                    continue
        raise RuntimeError(
            f"{self._label} failed after {self.max_reconnects} stateless "
            "reconnect/replay attempts: " + "; ".join(failures)
        )

    def _discard_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        for stream_name in ("stdin", "stdout"):
            stream = getattr(process, stream_name, None)
            if stream is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        stream.close()
                    except OSError:
                        pass
        if process.poll() is None:
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)

    def close(self) -> None:
        self._discard_process()


@dataclass(frozen=True)
class _PolicyTensorMetadata:
    name: str
    shape: list[int]
    type: str = "tensor(float)"


class RemoteSshOnnxSession:
    """Small ONNX Runtime session facade backed by a persistent SSH process."""

    def __init__(
        self,
        destination: str,
        remote_policy_path: str,
        *,
        remote_python: str = DEFAULT_REMOTE_PYTHON,
        remote_server_path: str = DEFAULT_REMOTE_SERVER,
        ssh_executable: str = "ssh",
        process_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        if (
            not destination
            or destination.startswith("-")
            or any(character.isspace() for character in destination)
            or any(character in destination for character in "\x00\r\n")
        ):
            raise ValueError(f"invalid SSH destination: {destination!r}")
        for label, value in (
            ("remote policy path", remote_policy_path),
            ("remote Python", remote_python),
            ("remote server path", remote_server_path),
        ):
            if not value or any(character in value for character in "\x00\r\n"):
                raise ValueError(f"invalid {label}: {value!r}")
        self.destination = destination
        self.remote_policy_path = remote_policy_path
        self.remote_python = remote_python
        self.remote_server_path = remote_server_path
        self.ssh_executable = ssh_executable
        self._input = _PolicyTensorMetadata("observation", [1, POLICY_INPUT_DIM])
        self._output = _PolicyTensorMetadata("action", [1, POLICY_OUTPUT_DIM])
        self.client = BinaryFloat32PolicyClient(
            self._command,
            label=f"remote ONNX policy at {destination}:{remote_policy_path}",
            process_factory=process_factory,
        )

    def _command(self) -> list[str]:
        remote_arguments = (
            self.remote_python,
            "-u",
            self.remote_server_path,
            f"--policy={self.remote_policy_path}",
        )
        remote_command = "exec " + " ".join(
            shlex.quote(argument) for argument in remote_arguments
        )
        return [
            self.ssh_executable,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "NumberOfPasswordPrompts=0",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=60",
            "-o",
            "ConnectionAttempts=4",
            "-o",
            "TCPKeepAlive=yes",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "IPQoS=throughput",
            "-o",
            "RekeyLimit=1G",
            self.destination,
            remote_command,
        ]

    def get_inputs(self) -> list[_PolicyTensorMetadata]:
        return [self._input]

    def get_outputs(self) -> list[_PolicyTensorMetadata]:
        return [self._output]

    def run(
        self, output_names: list[str], feeds: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        if output_names != [self._output.name] or set(feeds) != {self._input.name}:
            raise ValueError("remote policy session received unexpected ONNX names")
        observation = np.asarray(feeds[self._input.name], dtype=np.float32)
        if observation.shape != (1, POLICY_INPUT_DIM):
            raise ValueError(
                f"remote policy input must be (1, {POLICY_INPUT_DIM}), got {observation.shape}"
            )
        return [self.client.infer(observation[0])[None, :]]

    def close(self) -> None:
        self.client.close()


class KengoSonicSim:
    def __init__(
        self,
        policy_path: Path | None,
        xml_path: Path,
        *,
        action_clip: float,
        actuator_delay_steps: int,
        onnx_threads: int,
        remote_policy_ssh: str | None = None,
        remote_policy_path: str | None = None,
        remote_python: str = DEFAULT_REMOTE_PYTHON,
        remote_server_path: str = DEFAULT_REMOTE_SERVER,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "mujoco is not installed; install mujoco>=3.2,<4"
            ) from exc

        self.mujoco = mujoco
        self.action_clip = float(action_clip)
        self.remote_policy_ssh = remote_policy_ssh
        self.remote_policy_path = remote_policy_path
        if remote_policy_ssh is not None:
            if policy_path is not None or remote_policy_path is None:
                raise ValueError(
                    "remote policy mode requires only SSH destination/path"
                )
            self.session = RemoteSshOnnxSession(
                remote_policy_ssh,
                remote_policy_path,
                remote_python=remote_python,
                remote_server_path=remote_server_path,
            )
            self.policy_mode = "remote_ssh"
            self.policy_provider = "SSH/ONNXRuntime-CPUExecutionProvider"
            self.policy_display = f"{remote_policy_ssh}:{remote_policy_path}"
        else:
            if policy_path is None:
                raise ValueError("local policy mode requires --policy")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is not installed; install onnxruntime>=1.18,<2"
                ) from exc
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = int(onnx_threads)
            self.session = ort.InferenceSession(
                str(policy_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self.policy_mode = "local"
            self.policy_provider = self.session.get_providers()[0]
            self.policy_display = str(policy_path)
        if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) < 1:
            raise RuntimeError(
                "combined SONIC ONNX must expose one input and at least one output"
            )
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        if self.input.type != "tensor(float)" or self.output.type != "tensor(float)":
            raise RuntimeError(
                "ONNX input/output must be float32, "
                f"got {self.input.type} -> {self.output.type}"
            )
        if len(self.input.shape) != 2 or len(self.output.shape) != 2:
            raise RuntimeError(
                "combined SONIC ONNX must be rank two; "
                f"got {self.input.shape} -> {self.output.shape}"
            )
        expected_dimensions = (
            ("input batch", self.input.shape[0], 1),
            ("input feature", self.input.shape[1], POLICY_INPUT_DIM),
            ("output batch", self.output.shape[0], 1),
            ("output feature", self.output.shape[1], POLICY_OUTPUT_DIM),
        )
        mismatches = [
            f"{label}={actual} (expected {expected})"
            for label, actual, expected in expected_dimensions
            if isinstance(actual, int) and actual != expected
        ]
        if mismatches:
            raise RuntimeError(
                f"ONNX shape mismatch {self.input.shape} -> {self.output.shape}: "
                + ", ".join(mismatches)
            )

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        free_joints = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_joints) != 1:
            raise RuntimeError(
                f"Kengo MJCF must contain one free joint, found {len(free_joints)}"
            )
        free_joint = int(free_joints[0])
        self.root_qpos_adr = int(self.model.jnt_qposadr[free_joint])
        self.root_qvel_adr = int(self.model.jnt_dofadr[free_joint])

        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in KENGO_ISAACLAB_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self.actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in KENGO_ISAACLAB_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        if np.any(self.joint_ids < 0) or np.any(self.actuator_ids < 0):
            raise RuntimeError(
                "MJCF does not contain all 23 named Kengo joints and motors"
            )
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids].astype(np.int64)
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids].astype(np.int64)
        self.model.dof_armature[self.qvel_ids] = ARMATURE
        self.model.dof_damping[self.qvel_ids] = 0.0
        self.model.dof_frictionloss[self.qvel_ids] = 0.0

        self.orientation_adr, self.gyro_adr = self._resolve_imu_contract()
        self.history = SonicHistory()
        self.current_target = DEFAULT_JOINT_POS.copy()
        self.target_delay = [
            self.current_target.copy() for _ in range(int(actuator_delay_steps))
        ]
        self.actuator_delay_steps = int(actuator_delay_steps)
        self.policy_steps = 0

    @property
    def policy_process_starts(self) -> int:
        if isinstance(self.session, RemoteSshOnnxSession):
            return self.session.client.starts
        return 0

    @property
    def policy_reconnects(self) -> int:
        if isinstance(self.session, RemoteSshOnnxSession):
            return self.session.client.reconnects
        return 0

    def close(self) -> None:
        close = getattr(self.session, "close", None)
        if close is not None:
            close()

    def _resolve_imu_contract(self) -> tuple[int, int]:
        mujoco = self.mujoco
        orientation = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, IMU_ORIENTATION_SENSOR
        )
        gyro = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, IMU_ANGULAR_VELOCITY_SENSOR
        )
        if orientation < 0 or gyro < 0:
            raise RuntimeError("MJCF is missing the torso framequat/gyro IMU sensors")
        if (
            int(self.model.sensor_dim[orientation]) != 4
            or int(self.model.sensor_dim[gyro]) != 3
        ):
            raise RuntimeError(
                "Kengo IMU sensor dimensions must be framequat=4 and gyro=3"
            )
        if (
            self.model.sensor_objtype[orientation] != self.model.sensor_objtype[gyro]
            or self.model.sensor_objid[orientation] != self.model.sensor_objid[gyro]
        ):
            raise RuntimeError(
                "Kengo framequat and gyro must reference the same IMU site"
            )
        return int(self.model.sensor_adr[orientation]), int(self.model.sensor_adr[gyro])

    def reset(self, initial_state: tuple[np.ndarray, ...]) -> None:
        (
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel_w,
            joint_pos,
            joint_vel,
        ) = initial_state
        self.mujoco.mj_resetData(self.model, self.data)
        qpos = self.root_qpos_adr
        qvel = self.root_qvel_adr
        self.data.qpos[qpos : qpos + 3] = root_pos
        self.data.qpos[qpos + 3 : qpos + 7] = _normalize_quaternions(root_quat)
        self.data.qvel[qvel : qvel + 3] = root_lin_vel
        self.data.qvel[qvel + 3 : qvel + 6] = _quat_rotate_inverse(
            root_quat, root_ang_vel_w
        )
        self.data.qpos[self.qpos_ids] = joint_pos
        self.data.qvel[self.qvel_ids] = joint_vel
        self.current_target = np.asarray(joint_pos, dtype=np.float32).copy()
        self.target_delay = [
            self.current_target.copy() for _ in range(self.actuator_delay_steps)
        ]
        self.history.reset()
        self.policy_steps = 0
        self.mujoco.mj_forward(self.model, self.data)

    def root_state(self) -> tuple[np.ndarray, np.ndarray]:
        address = self.root_qpos_adr
        return (
            self.data.qpos[address : address + 3].astype(np.float32).copy(),
            self.imu_orientation(),
        )

    def imu_orientation(self) -> np.ndarray:
        values = self.data.sensordata[self.orientation_adr : self.orientation_adr + 4]
        return _normalize_quaternions(values).astype(np.float32)

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.qpos[self.qpos_ids].astype(np.float32).copy(),
            self.data.qvel[self.qvel_ids].astype(np.float32).copy(),
        )

    def policy_step(
        self, future_observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        torso_quat = self.imu_orientation()
        gyro = self.data.sensordata[self.gyro_adr : self.gyro_adr + 3].astype(
            np.float32
        )
        gravity = _quat_rotate_inverse(torso_quat, GRAVITY_W).astype(np.float32)
        joint_pos, joint_vel = self.joint_state()
        observation = self.history.build(
            future_observation,
            gyro,
            joint_pos - DEFAULT_JOINT_POS,
            joint_vel,
            gravity,
        )
        started = time.perf_counter()
        output = self.session.run(
            [self.output.name], {self.input.name: observation[None, :]}
        )[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        raw_action = np.asarray(output, dtype=np.float32).reshape(-1)
        if (
            raw_action.shape != (POLICY_OUTPUT_DIM,)
            or not np.isfinite(raw_action).all()
        ):
            raise RuntimeError(
                f"policy returned invalid action shape/value: {raw_action.shape}, "
                f"finite={np.isfinite(raw_action).all()}"
            )
        action = np.clip(raw_action, -self.action_clip, self.action_clip)
        self.history.record_action(action)
        self.current_target = DEFAULT_JOINT_POS + ACTION_SCALE * action
        self.policy_steps += 1
        return raw_action, action, inference_ms

    def physics_step(self) -> tuple[np.ndarray, np.ndarray]:
        target = self.current_target
        if self.target_delay:
            self.target_delay.append(target.copy())
            target = self.target_delay.pop(0)
        joint_pos, joint_vel = self.joint_state()
        raw_torque = KP * (target - joint_pos) - KD * joint_vel
        torque = np.clip(raw_torque, -EFFORT_LIMIT, EFFORT_LIMIT)
        self.data.ctrl[self.actuator_ids] = torque
        self.mujoco.mj_step(self.model, self.data)
        return raw_torque, torque


class VideoRecorder:
    """Stream MuJoCo offscreen RGB frames into an H.264 MP4."""

    def __init__(
        self,
        mujoco: Any,
        model: Any,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        camera_distance: float,
        camera_azimuth: float,
        camera_elevation: float,
    ) -> None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:  # pragma: no cover - depends on runtime extras.
            raise RuntimeError(
                "video recording requires imageio-ffmpeg in the active environment"
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = camera_distance
        self.camera.azimuth = camera_azimuth
        self.camera.elevation = camera_elevation
        self.writer = imageio_ffmpeg.write_frames(
            str(path),
            (width, height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
            ffmpeg_log_level="warning",
            output_params=["-movflags", "+faststart"],
        )
        self.writer.send(None)
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = 0
        self._closed = False

    def capture(self, data: Any, lookat: np.ndarray) -> None:
        self.camera.lookat[:] = np.asarray(lookat, dtype=np.float64)
        self.renderer.update_scene(data, camera=self.camera)
        frame = np.ascontiguousarray(self.renderer.render(), dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise RuntimeError(
                f"unexpected rendered frame shape {frame.shape}; expected "
                f"{(self.height, self.width, 3)}"
            )
        self.writer.send(frame)
        self.frames += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        writer_error: Exception | None = None
        try:
            self.writer.close()
        except Exception as exc:  # Preserve renderer cleanup on encoder failure.
            writer_error = exc
        finally:
            self.renderer.close()
        if writer_error is not None:
            raise writer_error


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, help="Local combined *_kengo.onnx policy")
    parser.add_argument(
        "--remote-policy-ssh",
        help="SSH destination for persistent remote inference, e.g. trainer@host.example",
    )
    parser.add_argument(
        "--remote-policy-path",
        help="POSIX path of the combined ONNX model on the remote host",
    )
    parser.add_argument(
        "--remote-python",
        default=DEFAULT_REMOTE_PYTHON,
        help="Remote Python executable (default: %(default)s)",
    )
    parser.add_argument(
        "--remote-server-path",
        default=DEFAULT_REMOTE_SERVER,
        help="Remote binary-protocol server script (default: %(default)s)",
    )
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--real-time", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--stop-on-fall", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--start-frame", type=int, default=0)
    end_condition = parser.add_mutually_exclusive_group()
    end_condition.add_argument(
        "--full-reference",
        action="store_true",
        help=(
            "Run every reference frame exactly once; conflicts with --loop and "
            "forces fall detection to be non-terminating"
        ),
    )
    end_condition.add_argument("--max-sim-seconds", type=float, default=4.0)
    parser.add_argument("--fall-height", type=float, default=0.5)
    parser.add_argument("--action-clip", type=float, default=20.0)
    parser.add_argument("--actuator-delay-steps", type=int, default=0)
    parser.add_argument("--onnx-threads", type=int, default=1)
    parser.add_argument("--progress-interval", type=float, default=1.0)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--video", type=Path, help="Record an offscreen H.264 MP4")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--camera-distance", type=float, default=2.8)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--min-policy-steps", type=int, default=1)
    parser.add_argument("--max-joint-rmse-rad", type=float)
    parser.add_argument("--max-root-position-rmse-m", type=float)
    parser.add_argument("--max-root-orientation-rmse-rad", type=float)
    args = parser.parse_args(argv)
    if args.full_reference:
        if args.loop:
            parser.error("--full-reference cannot be combined with --loop")
        args.stop_on_fall = False
    if args.remote_policy_ssh is None:
        if args.remote_policy_path is not None:
            parser.error("--remote-policy-path requires --remote-policy-ssh")
        if args.policy is None:
            parser.error("provide --policy or --remote-policy-ssh/--remote-policy-path")
    else:
        if args.policy is not None:
            parser.error("--policy and --remote-policy-ssh are mutually exclusive")
        if not args.remote_policy_path:
            parser.error("--remote-policy-ssh requires --remote-policy-path")
        try:
            # Validate before MuJoCo initialization; the process remains lazy.
            RemoteSshOnnxSession(
                args.remote_policy_ssh,
                args.remote_policy_path,
                remote_python=args.remote_python,
                remote_server_path=args.remote_server_path,
                process_factory=lambda *unused_args, **unused_kwargs: None,
            ).close()
        except ValueError as exc:
            parser.error(str(exc))
    if args.policy is not None:
        args.policy = args.policy.expanduser().resolve()
    args.motion = args.motion.expanduser().resolve()
    args.xml = args.xml.expanduser().resolve()
    if args.metrics_json is not None:
        args.metrics_json = args.metrics_json.expanduser().resolve()
    if args.video is not None:
        args.video = args.video.expanduser().resolve()
    files_to_check = ["motion", "xml"]
    if args.policy is not None:
        files_to_check.append("policy")
    for name in files_to_check:
        if not getattr(args, name).is_file():
            parser.error(f"--{name} file not found: {getattr(args, name)}")
    if args.headless and not args.full_reference and args.max_sim_seconds <= 0.0:
        parser.error("headless mode requires --max-sim-seconds > 0")
    if args.max_sim_seconds < 0.0 or not math.isfinite(args.max_sim_seconds):
        parser.error("--max-sim-seconds must be finite and non-negative")
    if args.action_clip <= 0.0 or not math.isfinite(args.action_clip):
        parser.error("--action-clip must be finite and positive")
    if args.fall_height <= 0.0 or not math.isfinite(args.fall_height):
        parser.error("--fall-height must be finite and positive")
    if args.actuator_delay_steps < 0:
        parser.error("--actuator-delay-steps must be non-negative")
    if args.onnx_threads < 1 or args.min_policy_steps < 1:
        parser.error("--onnx-threads and --min-policy-steps must be positive")
    if args.progress_interval < 0.0 or not math.isfinite(args.progress_interval):
        parser.error("--progress-interval must be finite and non-negative")
    if args.video_width < 16 or args.video_height < 16:
        parser.error("--video-width and --video-height must be at least 16")
    if (
        args.video_fps <= 0.0
        or args.video_fps > 1.0 / SIM_DT
        or not math.isfinite(args.video_fps)
    ):
        parser.error(f"--video-fps must be finite and in (0, {1.0 / SIM_DT:g}]")
    for option in ("camera_distance", "camera_azimuth", "camera_elevation"):
        value = getattr(args, option)
        if not math.isfinite(value):
            parser.error(f"--{option.replace('_', '-')} must be finite")
    if args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive")
    for option in (
        "max_joint_rmse_rad",
        "max_root_position_rmse_m",
        "max_root_orientation_rmse_rad",
    ):
        limit = getattr(args, option)
        if limit is not None and (limit < 0.0 or not math.isfinite(limit)):
            parser.error(
                f"--{option.replace('_', '-')} must be finite and non-negative"
            )
    return args


def _summary_stat(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "rms": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _build_summary(
    args: argparse.Namespace,
    clip: MotionClip,
    reference: MotionReference,
    sim: KengoSonicSim,
    metrics: Metrics,
    physics_steps: int,
    wall_seconds: float,
    runtime_error: str | None,
) -> dict[str, Any]:
    tracking = {
        "joint_rmse_rad": _summary_stat(metrics.joint_rmse),
        "root_position_error_m": _summary_stat(metrics.root_position_error),
        "root_orientation_error_rad": _summary_stat(metrics.root_orientation_error),
    }
    failures: list[str] = []
    if runtime_error is not None:
        failures.append(f"runtime_error: {runtime_error}")
    if not metrics.finite:
        failures.append("MuJoCo state became non-finite")
    if metrics.fall_time_s is not None:
        failures.append(
            f"base height fell below {args.fall_height:.3f} m at {metrics.fall_time_s:.3f} s"
        )
    if sim.policy_steps < args.min_policy_steps:
        failures.append(
            f"only {sim.policy_steps} policy steps ran; required {args.min_policy_steps}"
        )
    threshold_checks = (
        (
            "joint RMSE",
            tracking["joint_rmse_rad"]["rms"],
            args.max_joint_rmse_rad,
            "rad",
        ),
        (
            "root-position RMSE",
            tracking["root_position_error_m"]["rms"],
            args.max_root_position_rmse_m,
            "m",
        ),
        (
            "root-orientation RMSE",
            tracking["root_orientation_error_rad"]["rms"],
            args.max_root_orientation_rmse_rad,
            "rad",
        ),
    )
    for label, measured, limit, unit in threshold_checks:
        if limit is not None and measured is not None and measured > limit:
            failures.append(f"{label} {measured:.6g} {unit} exceeds {limit:.6g} {unit}")

    sim_time = float(sim.data.time)
    base_height_min = (
        None
        if not math.isfinite(metrics.min_base_height_m)
        else metrics.min_base_height_m
    )
    base_height_max = (
        None
        if not math.isfinite(metrics.max_base_height_m)
        else metrics.max_base_height_m
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "success": not failures,
        "failures": failures,
        "policy": {
            "mode": sim.policy_mode,
            "path": str(args.policy) if args.policy is not None else None,
            "remote_ssh": args.remote_policy_ssh,
            "remote_path": args.remote_policy_path,
            "input_name": sim.input.name,
            "input_shape": sim.input.shape,
            "output_name": sim.output.name,
            "output_shape": sim.output.shape,
            "input_dim": POLICY_INPUT_DIM,
            "output_dim": POLICY_OUTPUT_DIM,
            "provider": sim.policy_provider,
            "process_starts": sim.policy_process_starts,
            "reconnects": sim.policy_reconnects,
        },
        "motion": {
            "path": str(clip.source_path),
            "source_fps": clip.source_fps,
            "resampled_fps": clip.fps,
            "frames": clip.num_frames,
            "duration_s": clip.duration_s,
            "start_frame": reference.start_frame,
            "final_frame": reference.frame,
            "loop": reference.loop,
        },
        "runtime": {
            "sim_time_s": sim_time,
            "wall_time_s": wall_seconds,
            "physics_steps": physics_steps,
            "policy_steps": sim.policy_steps,
            "physics_hz_wall": physics_steps / max(wall_seconds, 1.0e-12),
            "policy_hz_wall": sim.policy_steps / max(wall_seconds, 1.0e-12),
            "sim_dt_s": SIM_DT,
            "policy_fps": POLICY_FPS,
            "actuator_delay_steps": args.actuator_delay_steps,
            "runtime_error": runtime_error,
        },
        "stability": {
            "finite": metrics.finite,
            "fall_height_m": args.fall_height,
            "fall_time_s": metrics.fall_time_s,
            "min_base_height_m": base_height_min,
            "max_base_height_m": base_height_max,
            "max_abs_roll_rad": metrics.max_abs_roll_rad,
            "max_abs_pitch_rad": metrics.max_abs_pitch_rad,
            "max_abs_joint_velocity_rad_s": metrics.max_abs_joint_velocity_rad_s,
        },
        "tracking": tracking,
        "control": {
            "action_clip": args.action_clip,
            "action_rms": _rms_from_sum(
                metrics.action_sum_squares, metrics.action_count
            ),
            "action_abs_max": metrics.action_abs_max,
            "action_clip_fraction": (
                metrics.action_clipped_components / metrics.action_count
                if metrics.action_count
                else None
            ),
            "torque_rms_nm": _rms_from_sum(
                metrics.torque_sum_squares, metrics.torque_count
            ),
            "torque_abs_max_nm": metrics.torque_abs_max_nm,
            "torque_saturation_fraction": (
                metrics.torque_saturated_components / metrics.torque_count
                if metrics.torque_count
                else None
            ),
        },
        "performance": {
            "inference_ms": _summary_stat(metrics.inference_ms),
        },
    }
    return summary


def run(args: argparse.Namespace) -> int:
    clip = load_motion_npz(args.motion)
    reference = MotionReference(clip, start_frame=args.start_frame, loop=args.loop)
    full_reference = bool(getattr(args, "full_reference", False))
    full_reference_policy_steps: int | None = None
    full_reference_physics_steps: int | None = None
    if full_reference:
        (
            full_reference_policy_steps,
            full_reference_physics_steps,
        ) = _full_reference_step_budget(clip.num_frames, reference.start_frame)
    print(
        f"[MOTION] {clip.source_path.name}: {clip.num_frames} frames, "
        f"{clip.duration_s:.3f}s at {clip.fps:.1f}Hz (source {clip.source_fps:.6g}Hz)"
    )
    if full_reference:
        print(
            f"[MOTION] full-reference budget={full_reference_policy_steps} policy "
            f"steps/{full_reference_physics_steps} physics steps"
        )
    sim = KengoSonicSim(
        args.policy,
        args.xml,
        action_clip=args.action_clip,
        actuator_delay_steps=args.actuator_delay_steps,
        onnx_threads=args.onnx_threads,
        remote_policy_ssh=args.remote_policy_ssh,
        remote_policy_path=args.remote_policy_path,
        remote_python=args.remote_python,
        remote_server_path=args.remote_server_path,
    )
    sim.reset(reference.initial_state())
    print(
        f"[POLICY] {sim.policy_display}: {sim.input.name}{sim.input.shape} -> "
        f"{sim.output.name}{sim.output.shape}"
    )
    print(
        f"[SIM] dt={SIM_DT:.4f}s ({1.0 / SIM_DT:.0f}Hz), "
        f"policy={POLICY_FPS:.0f}Hz, action_clip={args.action_clip:g}, "
        f"delay={args.actuator_delay_steps} physics steps"
    )

    metrics = Metrics(fall_height=args.fall_height)
    physics_steps = 0
    runtime_error: str | None = None
    viewer_context = None
    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(
            sim.model, sim.data, show_left_ui=False, show_right_ui=False
        )
        viewer = viewer_context.__enter__()
        viewer.cam.distance = 2.5

    recorder = None
    next_video_time = 0.0
    if args.video is not None:
        recorder = VideoRecorder(
            sim.mujoco,
            sim.model,
            args.video,
            width=args.video_width,
            height=args.video_height,
            fps=args.video_fps,
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
        )
        initial_root_pos, _ = sim.root_state()
        recorder.capture(sim.data, initial_root_pos)
        next_video_time = 1.0 / args.video_fps
        print(
            f"[VIDEO] recording {args.video_width}x{args.video_height} "
            f"at {args.video_fps:g}fps -> {args.video}"
        )

    wall_started = time.perf_counter()
    next_tick = wall_started
    next_report = wall_started + args.progress_interval
    try:
        while viewer is None or viewer.is_running():
            if full_reference:
                if physics_steps >= full_reference_physics_steps:
                    break
            elif args.max_sim_seconds > 0.0 and sim.data.time >= args.max_sim_seconds:
                break
            policy_tick = physics_steps % POLICY_DECIMATION == 0
            if policy_tick:
                root_pos, root_quat = sim.root_state()
                joint_pos, joint_vel = sim.joint_state()
                (
                    target_root_pos,
                    target_root_quat,
                    target_joint_pos,
                ) = reference.current_target()
                metrics.record_tracking(
                    root_pos,
                    root_quat,
                    joint_pos,
                    target_root_pos,
                    target_root_quat,
                    target_joint_pos,
                )
                future = reference.future_observation(root_quat)
                raw_action, action, inference_ms = sim.policy_step(future)
                metrics.inference_ms.append(float(inference_ms))
                metrics.record_action(raw_action, action, args.action_clip)
                reference.advance()

            raw_torque, torque = sim.physics_step()
            metrics.record_torque(raw_torque, torque)
            physics_steps += 1
            root_pos, root_quat = sim.root_state()
            _, joint_vel = sim.joint_state()
            metrics.record_state(
                float(sim.data.time), root_pos, sim.data.qvel, root_quat, joint_vel
            )
            if (
                recorder is not None
                and sim.data.time + 0.5 * SIM_DT >= next_video_time
            ):
                recorder.capture(sim.data, root_pos)
                next_video_time += 1.0 / args.video_fps
            if not metrics.finite or (
                not full_reference
                and args.stop_on_fall
                and metrics.fall_time_s is not None
            ):
                break

            if viewer is not None and policy_tick:
                viewer.cam.lookat[:] = root_pos
                viewer.sync()
            now = time.perf_counter()
            if args.progress_interval > 0.0 and now >= next_report:
                joint_error = metrics.joint_rmse[-1] if metrics.joint_rmse else math.nan
                print(
                    f"[SIM] t={sim.data.time:7.3f}s frame={reference.frame:5d} "
                    f"z={root_pos[2]:.3f} joint_rmse={joint_error:.4f}rad "
                    f"infer={metrics.inference_ms[-1]:.3f}ms"
                )
                next_report = now + args.progress_interval
            if args.real_time:
                next_tick += SIM_DT
                delay = next_tick - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
    except Exception as exc:  # Preserve metrics and emit machine-readable failure output.
        runtime_error = f"{type(exc).__name__}: {exc}"
        metrics.finite = metrics.finite and bool(
            np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()
        )
    finally:
        if viewer_context is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    viewer_context.__exit__(None, None, None)
            except Exception as exc:
                if runtime_error is None:
                    runtime_error = f"viewer close {type(exc).__name__}: {exc}"
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                if runtime_error is None:
                    runtime_error = f"video close {type(exc).__name__}: {exc}"
        try:
            sim.close()
        except Exception as exc:
            if runtime_error is None:
                runtime_error = f"policy close {type(exc).__name__}: {exc}"

    wall_seconds = time.perf_counter() - wall_started
    summary = _build_summary(
        args,
        clip,
        reference,
        sim,
        metrics,
        physics_steps,
        wall_seconds,
        runtime_error,
    )
    if recorder is not None:
        video_bytes = args.video.stat().st_size if args.video.is_file() else 0
        summary["video"] = {
            "path": str(args.video),
            "frames": recorder.frames,
            "fps": args.video_fps,
            "width": args.video_width,
            "height": args.video_height,
            "bytes": video_bytes,
        }
        if video_bytes <= 0:
            summary["success"] = False
            summary["failures"].append("video output is missing or empty")
        else:
            print(
                f"[VIDEO] wrote {recorder.frames} frames, "
                f"{video_bytes / (1024 * 1024):.2f} MiB -> {args.video}"
            )
    result_json = json.dumps(
        summary, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    print(f"[RESULT_JSON] {result_json}")
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(
            json.dumps(
                summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[METRICS] {args.metrics_json}")
    return 0 if summary["success"] else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


assert len(KENGO_ISAACLAB_JOINT_NAMES) == NUM_JOINTS
assert POLICY_DECIMATION * SIM_DT == 1.0 / POLICY_FPS
assert TOKENIZER_DIM == 520
assert PROPRIOCEPTION_DIM == 750
assert POLICY_INPUT_DIM == 1270
