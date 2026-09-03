#!/usr/bin/env python3
"""Render one complete Kengo reference motion to a fixed-rate 30 fps MP4.

The renderer intentionally shares the motion loader, offscreen video recorder,
and MuJoCo joint-name contract with ``run_kengo_sonic_sim2sim.py``.  It only
writes reference poses into MuJoCo; it does not run physics or a policy.

Example::

    python gear_sonic/scripts/render_kengo_reference_video.py \
        --motion path/to/kengo_wbt.npz \
        --video reference.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts.run_kengo_sonic_sim2sim import (  # noqa: E402
    DEFAULT_XML,
    NUM_JOINTS,
    VideoRecorder,
    load_motion_npz,
)
from gear_sonic.utils.kengo_contract import (  # noqa: E402
    KENGO_ISAACLAB_JOINT_NAMES,
)


VIDEO_FPS = 30.0


def reference_video_timeline(
    num_reference_frames: int,
    reference_fps: float,
    video_fps: float = VIDEO_FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact video PTS and source-frame indices for a full reference.

    A reference containing ``N`` frames at ``reference_fps`` occupies
    ``N / reference_fps`` seconds: each source pose owns one source-frame
    interval, including the final pose.  The output contains the smallest
    whole number of fixed-rate video frames whose duration covers that whole
    interval.  Each output interval samples the source at its midpoint, and
    the endpoints are pinned so the first and final reference poses are never
    lost to downsampling.
    """

    if isinstance(num_reference_frames, bool) or not isinstance(
        num_reference_frames, (int, np.integer)
    ):
        raise ValueError("num_reference_frames must be an integer")
    if num_reference_frames < 1:
        raise ValueError("num_reference_frames must be positive")
    reference_fps = float(reference_fps)
    video_fps = float(video_fps)
    if not math.isfinite(reference_fps) or reference_fps <= 0.0:
        raise ValueError("reference_fps must be finite and positive")
    if not math.isfinite(video_fps) or video_fps <= 0.0:
        raise ValueError("video_fps must be finite and positive")

    exact_frame_count = num_reference_frames * video_fps / reference_fps
    # Moving one ULP toward -infinity prevents an exactly integral duration
    # represented as 36.00000000000001 from acquiring an unintended frame.
    frame_count = max(1, int(math.ceil(math.nextafter(exact_frame_count, -math.inf))))
    if num_reference_frames > 1:
        frame_count = max(frame_count, 2)

    video_frames = np.arange(frame_count, dtype=np.float64)
    presentation_times = video_frames / video_fps
    midpoint_times = (video_frames + 0.5) / video_fps
    reference_indices = np.floor(midpoint_times * reference_fps).astype(np.int64)
    np.clip(reference_indices, 0, num_reference_frames - 1, out=reference_indices)
    reference_indices[0] = 0
    reference_indices[-1] = num_reference_frames - 1
    return presentation_times, reference_indices


def _resolve_kengo_pose_addresses(mujoco: Any, model: Any) -> tuple[int, np.ndarray]:
    """Resolve the free root and the canonical 23 Kengo joint qpos slots."""

    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise RuntimeError(
            f"Kengo MJCF must contain one free joint, found {len(free_joints)}"
        )
    root_qpos_adr = int(model.jnt_qposadr[int(free_joints[0])])
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in KENGO_ISAACLAB_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    if np.any(joint_ids < 0):
        missing = [
            name
            for name, joint_id in zip(KENGO_ISAACLAB_JOINT_NAMES, joint_ids)
            if joint_id < 0
        ]
        raise RuntimeError(f"MJCF is missing Kengo joints: {missing}")
    joint_qpos_ids = model.jnt_qposadr[joint_ids].astype(np.int64)
    if joint_qpos_ids.shape != (NUM_JOINTS,):
        raise RuntimeError(
            f"expected {NUM_JOINTS} Kengo joint qpos addresses, "
            f"got {joint_qpos_ids.shape}"
        )
    return root_qpos_adr, joint_qpos_ids


def render_reference_video(
    motion_path: Path,
    xml_path: Path,
    video_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    camera_distance: float = 2.8,
    camera_azimuth: float = 135.0,
    camera_elevation: float = -18.0,
) -> dict[str, Any]:
    """Render every 30 fps interval covering the complete reference clip."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - runtime dependency.
        raise RuntimeError("reference rendering requires mujoco>=3.2,<4") from exc

    clip = load_motion_npz(motion_path)
    presentation_times, reference_indices = reference_video_timeline(
        clip.num_frames, clip.fps
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    root_qpos_adr, joint_qpos_ids = _resolve_kengo_pose_addresses(mujoco, model)
    mujoco.mj_resetData(model, data)

    recorder = VideoRecorder(
        mujoco,
        model,
        video_path,
        width=width,
        height=height,
        fps=VIDEO_FPS,
        camera_distance=camera_distance,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
    )
    anchor_xy = clip.root_pos_w[0, :2].astype(np.float64).copy()
    try:
        for reference_index in reference_indices:
            index = int(reference_index)
            root_pos = clip.root_pos_w[index].astype(np.float64).copy()
            root_pos[:2] -= anchor_xy
            data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_pos
            data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = clip.root_quat_w[index]
            data.qpos[joint_qpos_ids] = clip.joint_pos[index]
            data.time = index / clip.fps
            mujoco.mj_forward(model, data)
            recorder.capture(data, root_pos)
    finally:
        recorder.close()

    video_bytes = video_path.stat().st_size if video_path.is_file() else 0
    if recorder.frames != len(reference_indices):
        raise RuntimeError(
            f"recorder wrote {recorder.frames} frames, "
            f"expected {len(reference_indices)}"
        )
    if video_bytes <= 0:
        raise RuntimeError(f"video output is missing or empty: {video_path}")
    return {
        "motion_path": str(clip.source_path),
        "xml_path": str(xml_path),
        "video_path": str(video_path),
        "reference_frames": clip.num_frames,
        "reference_fps": clip.fps,
        "reference_duration_s": clip.num_frames / clip.fps,
        "first_reference_frame": int(reference_indices[0]),
        "last_reference_frame": int(reference_indices[-1]),
        "video_frames": recorder.frames,
        "video_fps": VIDEO_FPS,
        "video_duration_s": len(presentation_times) / VIDEO_FPS,
        "width": width,
        "height": height,
        "bytes": video_bytes,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=2.8)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    args = parser.parse_args(argv)
    args.motion = args.motion.expanduser().resolve()
    args.xml = args.xml.expanduser().resolve()
    args.video = args.video.expanduser().resolve()
    for option in ("motion", "xml"):
        if not getattr(args, option).is_file():
            parser.error(f"--{option} file not found: {getattr(args, option)}")
    if args.video.suffix.lower() != ".mp4":
        parser.error("--video must use the .mp4 extension")
    if args.width < 16 or args.height < 16:
        parser.error("--width and --height must be at least 16")
    for option in ("camera_distance", "camera_azimuth", "camera_elevation"):
        value = getattr(args, option)
        if not math.isfinite(value):
            parser.error(f"--{option.replace('_', '-')} must be finite")
    if args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = render_reference_video(
            args.motion,
            args.xml,
            args.video,
            width=args.width,
            height=args.height,
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
        )
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"[RESULT_JSON] {json.dumps(summary, sort_keys=True, allow_nan=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
