#!/usr/bin/env python3
"""Validate staged Kengo assets and a converted SONIC motion library."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np

try:
    from gear_sonic.data_process.convert_kengo_npz_to_motion_lib import (
        NUM_KENGO_BODIES,
        NUM_KENGO_DOFS,
        parse_mjcf_contract,
    )
except ModuleNotFoundError as exc:
    if exc.name != "gear_sonic":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from gear_sonic.data_process.convert_kengo_npz_to_motion_lib import (
        NUM_KENGO_BODIES,
        NUM_KENGO_DOFS,
        parse_mjcf_contract,
    )

from gear_sonic.utils.kengo_contract import (
    KENGO_ISAACLAB_BODY_NAMES,
    KENGO_ISAACLAB_JOINT_NAMES,
    KENGO_ISAACLAB_TO_MUJOCO_BODY,
    KENGO_ISAACLAB_TO_MUJOCO_DOF,
    KENGO_MUJOCO_BODY_NAMES,
    KENGO_MUJOCO_TO_ISAACLAB_BODY,
    KENGO_MUJOCO_TO_ISAACLAB_DOF,
)

REQUIRED_ENTRY_FIELDS = {
    "root_trans_offset",
    "pose_aa",
    "dof",
    "root_rot",
    "smpl_joints",
    "fps",
}


def _validate_urdf(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    links = {link.get("name") for link in root.findall("link")}
    movable = [joint for joint in root.findall("joint") if joint.get("type") != "fixed"]
    names = [joint.get("name") for joint in movable]
    if set(names) != set(KENGO_ISAACLAB_JOINT_NAMES) or len(names) != NUM_KENGO_DOFS:
        raise ValueError("URDF movable joints do not match the 23-DoF Kengo contract")
    if not set(KENGO_ISAACLAB_BODY_NAMES).issubset(links):
        raise ValueError("URDF does not contain every contracted Kengo body")

    missing_meshes: list[str] = []
    mesh_refs = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh_refs.append(filename)
            if not (path.parent / filename).resolve().is_file():
                missing_meshes.append(filename)
    if missing_meshes:
        raise FileNotFoundError(
            f"URDF mesh references are missing: {missing_meshes[:5]}"
        )
    return {
        "links": len(links),
        "movable_joints": len(movable),
        "mesh_refs": len(mesh_refs),
    }


def _validate_mappings() -> None:
    for size, forward, reverse in (
        (NUM_KENGO_DOFS, KENGO_ISAACLAB_TO_MUJOCO_DOF, KENGO_MUJOCO_TO_ISAACLAB_DOF),
        (
            NUM_KENGO_BODIES,
            KENGO_ISAACLAB_TO_MUJOCO_BODY,
            KENGO_MUJOCO_TO_ISAACLAB_BODY,
        ),
    ):
        values = np.arange(size)
        if not np.array_equal(values[forward][reverse], values):
            raise ValueError(f"Kengo {size}-element mapping does not round-trip")


def validate(
    urdf_path: Path, mjcf_path: Path, motion_path: Path, manifest_path: Path | None
) -> dict[str, object]:
    urdf = urdf_path.expanduser().resolve(strict=True)
    mjcf = mjcf_path.expanduser().resolve(strict=True)
    motion = motion_path.expanduser().resolve(strict=True)
    manifest = (
        manifest_path.expanduser().resolve(strict=True)
        if manifest_path is not None
        else motion.with_suffix(motion.suffix + ".manifest.json").resolve(strict=True)
    )

    _validate_mappings()
    urdf_summary = _validate_urdf(urdf)
    contract = parse_mjcf_contract(mjcf)
    if list(contract.body_names) != KENGO_MUJOCO_BODY_NAMES:
        raise ValueError("MJCF body order does not match the shared Kengo contract")

    data = joblib.load(motion)
    if not isinstance(data, dict) or not data:
        raise ValueError("Motion library must be a non-empty dictionary")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_data["counts"]["written"] != len(data):
        raise ValueError("Manifest written count does not match motion library")
    counts = manifest_data["counts"]
    sonic_filter = manifest_data.get("sonic_keyword_filter")
    if manifest_data.get("schema_version", 0) < 2:
        raise ValueError("Motion manifest predates the auditable SONIC keyword filter")
    if not isinstance(sonic_filter, dict) or not sonic_filter.get("enabled"):
        raise ValueError("Official SONIC filename-keyword filter is not enabled")
    excluded = sonic_filter.get("excluded")
    if not isinstance(excluded, list):
        raise ValueError(
            "SONIC keyword-filter exclusions must be recorded in the manifest"
        )
    if counts.get("filtered_by_sonic_keywords") != len(excluded):
        raise ValueError(
            "SONIC keyword-filter count does not match its exclusion records"
        )
    if counts["selected_before_limit"] != (
        counts["discovered"]
        - counts["filtered_by_quality"]
        - counts["filtered_by_sonic_keywords"]
    ):
        raise ValueError("Manifest filter counts do not reconcile")
    if (
        counts["written"]
        != counts["selected_before_limit"] - counts["skipped_by_limit"]
    ):
        raise ValueError("Manifest written/limit counts do not reconcile")

    frames_total = 0
    seconds_total = 0.0
    for key, entry in data.items():
        if set(entry) != REQUIRED_ENTRY_FIELDS:
            raise ValueError(
                f"{key}: unexpected fields {sorted(set(entry) ^ REQUIRED_ENTRY_FIELDS)}"
            )
        dof = np.asarray(entry["dof"])
        pose_aa = np.asarray(entry["pose_aa"])
        root_pos = np.asarray(entry["root_trans_offset"])
        root_rot = np.asarray(entry["root_rot"])
        smpl_joints = np.asarray(entry["smpl_joints"])
        fps = float(entry["fps"])
        frames = dof.shape[0]
        expected_shapes = {
            "dof": (frames, NUM_KENGO_DOFS),
            "pose_aa": (frames, NUM_KENGO_BODIES, 3),
            "root_trans_offset": (frames, 3),
            "root_rot": (frames, 4),
            "smpl_joints": (frames, 24, 3),
        }
        arrays = {
            "dof": dof,
            "pose_aa": pose_aa,
            "root_trans_offset": root_pos,
            "root_rot": root_rot,
            "smpl_joints": smpl_joints,
        }
        for field, expected_shape in expected_shapes.items():
            if arrays[field].shape != expected_shape:
                raise ValueError(
                    f"{key}: {field} is {arrays[field].shape}, expected {expected_shape}"
                )
            if not np.isfinite(arrays[field]).all():
                raise ValueError(f"{key}: {field} contains non-finite values")
        if frames < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{key}: invalid frames/fps ({frames}, {fps})")
        np.testing.assert_allclose(
            pose_aa[:, 1:, :],
            dof[:, :, None] * contract.joint_axes[None, :, :],
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{key}: joint axis-angle values disagree with DOFs",
        )
        np.testing.assert_allclose(
            np.linalg.norm(root_rot, axis=1),
            1.0,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"{key}: root quaternions are not normalized",
        )
        frames_total += frames
        seconds_total += frames / fps

    if frames_total != manifest_data["counts"]["frames"]:
        raise ValueError("Manifest frame count does not match motion library")
    return {
        "urdf": urdf_summary,
        "mjcf": {
            "bodies": len(contract.body_names),
            "joints": len(contract.joint_names),
        },
        "motions": len(data),
        "frames": frames_total,
        "hours": seconds_total / 3600.0,
        "quality_filter": manifest_data.get("quality_filter"),
        "sonic_keyword_filter": {
            "enabled": sonic_filter["enabled"],
            "keywords": len(sonic_filter.get("keywords", [])),
            "excluded": len(excluded),
        },
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "external_dependencies/kengo_robot_description"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=root / "urdf/kengo_with_fist.urdf")
    parser.add_argument("--mjcf", type=Path, default=root / "xml/kengo_with_fist.xml")
    parser.add_argument(
        "--motion",
        type=Path,
        default=Path("data/kengo_motion_lib/robot_filtered/kengo_sonic_filtered.pkl"),
    )
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            validate(args.urdf, args.mjcf, args.motion, args.manifest),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
