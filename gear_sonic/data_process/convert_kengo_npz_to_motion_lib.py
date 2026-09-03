#!/usr/bin/env python3
"""Convert retargeted Kengo NPZ files into a SONIC motion-lib PKL.

The converter deliberately has no Isaac Lab dependency.  The robot contract is
checked in three places before any output is written:

* the shared Kengo MuJoCo joint-name contract;
* depth-first body/joint order and joint axes in the clean Kengo MJCF; and
* the five fields and array shapes in every input NPZ.

The output is one joblib dictionary containing every selected motion.  Keys are
derived injectively from relative paths, so equal basenames in different source
directories cannot overwrite one another.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import quote

import joblib
import numpy as np

try:
    from gear_sonic.utils.kengo_contract import KENGO_MUJOCO_JOINT_NAMES
except ModuleNotFoundError as exc:
    if exc.name != "gear_sonic":
        raise
    # Keep direct ``python gear_sonic/data_process/...py`` execution usable
    # before the repository is installed as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from gear_sonic.utils.kengo_contract import KENGO_MUJOCO_JOINT_NAMES

NUM_KENGO_DOFS = 23
NUM_KENGO_BODIES = 24
ROOT_BODY_NAME = "torso_link"
REQUIRED_NPZ_FIELDS = frozenset(
    {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
)
DEFAULT_QUALITY_STATUSES = ("PASS", "WARN")
KNOWN_QUALITY_STATUSES = frozenset({"PASS", "WARN", "FAIL"})
# Keep this list in sync with ``filter_and_copy_bones_data.py``.  That script
# filters Bones motions by ``parent/basename``; applying the same predicate
# during Kengo conversion avoids materializing a second tree of private NPZs.
SONIC_DEFAULT_FILTER_KEYWORDS = (
    "bed",
    "bike",
    "chair",
    "climb",
    "com_up_50cm",
    "sitting",
    "step_on",
    "seat",
    "table",
    "_sit_",
    "sit_",
    "ladder",
    "crutch",
    "_bed_",
    "_ride_",
    "scooter",
    "stepdown",
    "acrobatics_",
    "box_hspu",
    "cartwheel",
    "50cm_box_",
    "on_box",
    "fall_from",
    "handstand_ff_",
    "on_1m",
    "form_box",
    "off_1m",
    "230m",
    "jump_over_obstacle_",
    "lift_crate_come_up_",
    "jump_to_shoulder_roll",
    "kozak_dance",
    "stair",
    "handstand",
    "box_jump",
    "monkey_jump",
    "safety_roll",
    "box_dips",
    "walking_on_edge",
    "push_obstacle",
)
QUATERNION_NORM_TOLERANCE = 1e-3
MANIFEST_SCHEMA_VERSION = 2


class KengoConversionError(ValueError):
    """Raised when robot or motion data violates the Kengo conversion contract."""


class BatchConversionError(KengoConversionError):
    """Raised after one or more inputs fail conversion."""


@dataclass(frozen=True)
class MjcfContract:
    """Kinematic information extracted from the clean Kengo MJCF."""

    path: Path
    sha256: str
    root_body_name: str
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_axes: np.ndarray


@dataclass(frozen=True)
class KengoMotion:
    """Validated contents of one retargeted Kengo NPZ."""

    framerate: float
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray
    base_pos_w: np.ndarray
    base_quat_w: np.ndarray


@dataclass(frozen=True)
class QualityRecord:
    """One normalized row from the optional quality report."""

    relative_path: str
    status: str
    reasons: str


@dataclass(frozen=True)
class _InputItem:
    path: Path
    relative_path: str
    key: str
    quality: QualityRecord | None


@dataclass(frozen=True)
class _ConvertedItem:
    item: _InputItem
    entry: dict[str, Any]
    frames: int
    fps: float
    source_sha256: str


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_mismatch(
    label: str, actual: Sequence[str], expected: Sequence[str]
) -> str:
    if len(actual) != len(expected):
        return f"{label} has {len(actual)} entries; expected {len(expected)}"
    for index, (actual_name, expected_name) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if actual_name != expected_name:
            return (
                f"{label}[{index}] is {actual_name!r}; expected {expected_name!r} "
                "(Kengo MuJoCo DFS order)"
            )
    return f"{label} does not match the Kengo MuJoCo joint contract"


def _parse_axis(joint: ET.Element, joint_name: str) -> np.ndarray:
    axis_text = joint.get("axis")
    if axis_text is None:
        raise KengoConversionError(f"MJCF joint {joint_name!r} has no explicit axis")
    try:
        axis = np.asarray(
            [float(value) for value in axis_text.split()], dtype=np.float64
        )
    except ValueError as exc:
        raise KengoConversionError(
            f"MJCF joint {joint_name!r} has a non-numeric axis: {axis_text!r}"
        ) from exc
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise KengoConversionError(
            f"MJCF joint {joint_name!r} axis must contain three finite values; got {axis_text!r}"
        )
    norm = float(np.linalg.norm(axis))
    if norm <= np.finfo(np.float64).eps:
        raise KengoConversionError(f"MJCF joint {joint_name!r} has a zero axis")
    return (axis / norm).astype(np.float32)


def parse_mjcf_contract(mjcf_path: str | Path) -> MjcfContract:
    """Parse and strictly validate the clean 24-body/23-DoF Kengo MJCF.

    Bodies are traversed depth first, matching MuJoCo's kinematic ordering.
    Every non-root body must contain exactly one hinge joint and the actuator
    order must match that DFS order and the shared Kengo contract.
    """

    path = Path(mjcf_path).expanduser().resolve(strict=True)
    try:
        model = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise KengoConversionError(f"Cannot parse MJCF {path}: {exc}") from exc

    if model.tag != "mujoco":
        raise KengoConversionError(f"MJCF root tag must be 'mujoco'; got {model.tag!r}")

    compiler = model.find("compiler")
    angle_unit = "degree" if compiler is None else compiler.get("angle", "degree")
    if angle_unit != "radian":
        raise KengoConversionError(
            f"Kengo NPZ joint positions are radians, but MJCF compiler angle is {angle_unit!r}"
        )

    worldbodies = model.findall("worldbody")
    if len(worldbodies) != 1:
        raise KengoConversionError(
            f"MJCF must contain exactly one worldbody; got {len(worldbodies)}"
        )
    root_bodies = worldbodies[0].findall("body")
    if len(root_bodies) != 1:
        raise KengoConversionError(
            f"MJCF worldbody must contain exactly one root body; got {len(root_bodies)}"
        )

    body_nodes: list[ET.Element] = []

    def visit_body(body: ET.Element) -> None:
        body_nodes.append(body)
        for child in body.findall("body"):
            visit_body(child)

    visit_body(root_bodies[0])
    if len(body_nodes) != NUM_KENGO_BODIES:
        raise KengoConversionError(
            f"MJCF DFS contains {len(body_nodes)} bodies; expected {NUM_KENGO_BODIES} "
            "(torso root plus 23 actuated bodies)"
        )

    body_names = tuple(body.get("name", "") for body in body_nodes)
    if any(not name for name in body_names):
        raise KengoConversionError("Every Kengo MJCF body must have a non-empty name")
    if len(set(body_names)) != len(body_names):
        raise KengoConversionError("Kengo MJCF body names must be unique")
    if body_names[0] != ROOT_BODY_NAME:
        raise KengoConversionError(
            f"MJCF root body is {body_names[0]!r}; expected {ROOT_BODY_NAME!r}"
        )

    root_body = body_nodes[0]
    root_joints = root_body.findall("joint")
    root_freejoints = root_body.findall("freejoint")
    if len(root_joints) + len(root_freejoints) != 1:
        raise KengoConversionError(
            "Kengo MJCF root body must contain exactly one free joint"
        )
    if root_joints and root_joints[0].get("type", "hinge") != "free":
        raise KengoConversionError("Kengo MJCF root joint must have type='free'")

    joint_names: list[str] = []
    joint_axes: list[np.ndarray] = []
    for body_name, body in zip(body_names[1:], body_nodes[1:], strict=True):
        if body.findall("freejoint"):
            raise KengoConversionError(
                f"Non-root body {body_name!r} contains a freejoint"
            )
        joints = body.findall("joint")
        if len(joints) != 1:
            raise KengoConversionError(
                f"Non-root body {body_name!r} must contain exactly one joint; got {len(joints)}"
            )
        joint = joints[0]
        joint_name = joint.get("name", "")
        if not joint_name:
            raise KengoConversionError(f"Joint in body {body_name!r} has no name")
        if joint.get("type", "hinge") != "hinge":
            raise KengoConversionError(
                f"Kengo joint {joint_name!r} must be a hinge; got {joint.get('type')!r}"
            )
        joint_names.append(joint_name)
        joint_axes.append(_parse_axis(joint, joint_name))

    expected_joint_names = tuple(KENGO_MUJOCO_JOINT_NAMES)
    if len(expected_joint_names) != NUM_KENGO_DOFS:
        raise RuntimeError(
            f"Shared Kengo contract contains {len(expected_joint_names)} joints; "
            f"expected {NUM_KENGO_DOFS}"
        )
    if tuple(joint_names) != expected_joint_names:
        raise KengoConversionError(
            _sequence_mismatch(
                "MJCF DFS joint order", joint_names, expected_joint_names
            )
        )
    if len(set(joint_names)) != NUM_KENGO_DOFS:
        raise KengoConversionError("Kengo MJCF joint names must be unique")

    actuators = model.findall("actuator")
    if len(actuators) != 1:
        raise KengoConversionError(
            f"MJCF must contain exactly one actuator block; got {len(actuators)}"
        )
    actuator_children = list(actuators[0])
    if any(child.tag != "motor" for child in actuator_children):
        raise KengoConversionError(
            "Kengo MJCF actuator block must contain only motor elements"
        )
    actuator_joint_names = tuple(child.get("joint", "") for child in actuator_children)
    if actuator_joint_names != expected_joint_names:
        raise KengoConversionError(
            _sequence_mismatch(
                "MJCF actuator joint order", actuator_joint_names, expected_joint_names
            )
        )

    return MjcfContract(
        path=path,
        sha256=_sha256_file(path),
        root_body_name=body_names[0],
        body_names=body_names,
        joint_names=tuple(joint_names),
        joint_axes=np.stack(joint_axes, axis=0),
    )


def _load_joint_names(array: np.ndarray, path: Path) -> tuple[str, ...]:
    if array.shape != (NUM_KENGO_DOFS,):
        raise KengoConversionError(
            f"{path}: joint_names shape is {array.shape}; expected ({NUM_KENGO_DOFS},)"
        )
    if array.dtype.kind == "U":
        names = tuple(str(value) for value in array.tolist())
    elif array.dtype.kind == "S":
        try:
            names = tuple(value.decode("utf-8") for value in array.tolist())
        except UnicodeDecodeError as exc:
            raise KengoConversionError(
                f"{path}: joint_names is not valid UTF-8"
            ) from exc
    else:
        raise KengoConversionError(
            f"{path}: joint_names must use a fixed-width string dtype, got {array.dtype}"
        )
    return names


def _numeric_array(value: np.ndarray, field: str, path: Path) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise KengoConversionError(
            f"{path}: {field} must be numeric, got dtype {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise KengoConversionError(f"{path}: {field} contains NaN or infinity")
    return array


def load_kengo_npz(npz_path: str | Path, contract: MjcfContract) -> KengoMotion:
    """Load one NPZ and validate its exact five-field Kengo contract."""

    path = Path(npz_path).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".npz":
        raise KengoConversionError(f"Expected an .npz input, got {path}")

    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = frozenset(archive.files)
            if fields != REQUIRED_NPZ_FIELDS:
                missing = sorted(REQUIRED_NPZ_FIELDS - fields)
                unexpected = sorted(fields - REQUIRED_NPZ_FIELDS)
                raise KengoConversionError(
                    f"{path}: NPZ fields must be exactly {sorted(REQUIRED_NPZ_FIELDS)}; "
                    f"missing={missing}, unexpected={unexpected}"
                )

            framerate_array = np.asarray(archive["framerate"])
            joint_names_array = np.asarray(archive["joint_names"])
            joint_pos = _numeric_array(archive["joint_pos"], "joint_pos", path).copy()
            base_pos_w = _numeric_array(
                archive["base_pos_w"], "base_pos_w", path
            ).copy()
            base_quat_w = _numeric_array(
                archive["base_quat_w"], "base_quat_w", path
            ).copy()
    except (OSError, ValueError) as exc:
        if isinstance(exc, KengoConversionError):
            raise
        raise KengoConversionError(f"Cannot read NPZ {path}: {exc}") from exc

    if framerate_array.shape != () or framerate_array.dtype.kind not in "fiu":
        raise KengoConversionError(
            f"{path}: framerate must be a numeric scalar; got shape={framerate_array.shape}, "
            f"dtype={framerate_array.dtype}"
        )
    framerate = float(framerate_array)
    if not np.isfinite(framerate) or framerate <= 0:
        raise KengoConversionError(
            f"{path}: framerate must be finite and positive; got {framerate}"
        )

    joint_names = _load_joint_names(joint_names_array, path)
    if joint_names != contract.joint_names:
        raise KengoConversionError(
            f"{path}: {_sequence_mismatch('joint_names', joint_names, contract.joint_names)}"
        )
    if len(set(joint_names)) != NUM_KENGO_DOFS:
        raise KengoConversionError(f"{path}: joint_names contains duplicates")

    if joint_pos.ndim != 2 or joint_pos.shape[1:] != (NUM_KENGO_DOFS,):
        raise KengoConversionError(
            f"{path}: joint_pos shape is {joint_pos.shape}; expected (T, {NUM_KENGO_DOFS})"
        )
    frames = joint_pos.shape[0]
    if frames < 1:
        raise KengoConversionError(f"{path}: motion must contain at least one frame")
    if base_pos_w.shape != (frames, 3):
        raise KengoConversionError(
            f"{path}: base_pos_w shape is {base_pos_w.shape}; expected ({frames}, 3)"
        )
    if base_quat_w.shape != (frames, 4):
        raise KengoConversionError(
            f"{path}: base_quat_w shape is {base_quat_w.shape}; expected ({frames}, 4)"
        )

    quaternion_norms = np.linalg.norm(
        base_quat_w.astype(np.float64, copy=False), axis=1
    )
    if np.any(quaternion_norms <= np.finfo(np.float64).eps):
        bad_frame = int(np.flatnonzero(quaternion_norms <= np.finfo(np.float64).eps)[0])
        raise KengoConversionError(f"{path}: base_quat_w[{bad_frame}] has zero norm")
    norm_errors = np.abs(quaternion_norms - 1.0)
    if np.any(norm_errors > QUATERNION_NORM_TOLERANCE):
        bad_frame = int(np.argmax(norm_errors))
        raise KengoConversionError(
            f"{path}: base_quat_w[{bad_frame}] norm is {quaternion_norms[bad_frame]:.8g}; "
            f"maximum allowed deviation from 1 is {QUATERNION_NORM_TOLERANCE}"
        )
    base_quat_w = base_quat_w / quaternion_norms[:, None]

    return KengoMotion(
        framerate=framerate,
        joint_names=joint_names,
        joint_pos=joint_pos.astype(np.float32),
        base_pos_w=base_pos_w.astype(np.float32),
        base_quat_w=base_quat_w.astype(np.float32),
    )


def quaternion_wxyz_to_rotvec(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized WXYZ quaternions to shortest-path rotation vectors."""

    quaternions64 = np.asarray(quaternions, dtype=np.float64)
    if quaternions64.ndim != 2 or quaternions64.shape[1] != 4:
        raise KengoConversionError(
            f"Quaternion array shape must be (T, 4), got {quaternions64.shape}"
        )
    canonical = quaternions64.copy()
    canonical[canonical[:, 0] < 0.0] *= -1.0
    scalar = np.clip(canonical[:, 0], -1.0, 1.0)
    vector = canonical[:, 1:]
    sin_half_angle = np.linalg.norm(vector, axis=1)
    angle = 2.0 * np.arctan2(sin_half_angle, scalar)
    scale = np.full_like(angle, 2.0)
    nonzero = sin_half_angle > 1e-12
    scale[nonzero] = angle[nonzero] / sin_half_angle[nonzero]
    return (vector * scale[:, None]).astype(np.float32)


def motion_to_motion_lib(motion: KengoMotion, contract: MjcfContract) -> dict[str, Any]:
    """Convert one validated Kengo motion into a SONIC motion-lib entry."""

    frames = motion.joint_pos.shape[0]
    pose_aa = np.zeros((frames, NUM_KENGO_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, :] = quaternion_wxyz_to_rotvec(motion.base_quat_w)
    pose_aa[:, 1:, :] = contract.joint_axes[None, :, :] * motion.joint_pos[:, :, None]

    return {
        "root_trans_offset": motion.base_pos_w.copy(),
        "pose_aa": pose_aa,
        "dof": motion.joint_pos.copy(),
        "root_rot": motion.base_quat_w[:, [1, 2, 3, 0]].copy(),
        "smpl_joints": np.zeros((frames, NUM_KENGO_BODIES, 3), dtype=np.float32),
        "fps": motion.framerate,
    }


def convert_npz(
    npz_path: str | Path, contract_or_mjcf: MjcfContract | str | Path
) -> dict[str, Any]:
    """Convert one Kengo NPZ; accept either a parsed contract or an MJCF path."""

    contract = (
        contract_or_mjcf
        if isinstance(contract_or_mjcf, MjcfContract)
        else parse_mjcf_contract(contract_or_mjcf)
    )
    return motion_to_motion_lib(load_kengo_npz(npz_path, contract), contract)


def _normalize_relative_path(raw_path: str, *, source: str) -> str:
    text = raw_path.strip().replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or re.match(r"^[A-Za-z]:", text) is not None
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise KengoConversionError(f"{source}: unsafe relative_path {raw_path!r}")
    return path.as_posix()


def read_quality_csv(quality_csv: str | Path) -> dict[str, QualityRecord]:
    """Read a per-file report, keyed case-insensitively by normalized relative path."""

    path = Path(quality_csv).expanduser().resolve(strict=True)
    records: dict[str, QualityRecord] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {"relative_path", "status"}
        columns = set(reader.fieldnames or ())
        if not required_columns.issubset(columns):
            raise KengoConversionError(
                f"{path}: quality CSV requires columns {sorted(required_columns)}; "
                f"got {sorted(columns)}"
            )
        for line_number, row in enumerate(reader, start=2):
            relative_path = _normalize_relative_path(
                row.get("relative_path", ""), source=f"{path}:{line_number}"
            )
            status = row.get("status", "").strip().upper()
            if status not in KNOWN_QUALITY_STATUSES:
                raise KengoConversionError(
                    f"{path}:{line_number}: unknown quality status {status!r}; "
                    f"expected one of {sorted(KNOWN_QUALITY_STATUSES)}"
                )
            lookup_key = relative_path.casefold()
            if lookup_key in records:
                raise KengoConversionError(
                    f"{path}:{line_number}: duplicate quality row for {relative_path!r}"
                )
            records[lookup_key] = QualityRecord(
                relative_path=relative_path,
                status=status,
                reasons=row.get("reasons", "").strip(),
            )
    return records


def motion_key_from_relative_path(relative_path: str) -> str:
    """Build a deterministic, reversible, path-safe key from a relative NPZ path."""

    normalized = _normalize_relative_path(relative_path, source="motion key")
    path = PurePosixPath(normalized)
    if path.suffix.lower() != ".npz":
        raise KengoConversionError(f"Motion source must end in .npz: {relative_path!r}")
    without_suffix = path.with_suffix("").as_posix()
    return "kengo__" + quote(without_suffix, safe="-_.~")


def _discover_npz(input_path: Path) -> tuple[Path, list[tuple[Path, str]]]:
    resolved = input_path.expanduser().resolve(strict=True)
    if resolved.is_file():
        if resolved.suffix.lower() != ".npz":
            raise KengoConversionError(f"Input file must end in .npz: {resolved}")
        root = resolved.parent
        candidates = [(resolved, resolved.name)]
    elif resolved.is_dir():
        root = resolved
        candidates = [
            (path, path.relative_to(root).as_posix())
            for path in resolved.rglob("*")
            if path.is_file() and path.suffix.lower() == ".npz"
        ]
    else:
        raise KengoConversionError(
            f"Input is neither a file nor a directory: {resolved}"
        )

    candidates.sort(key=lambda item: (item[1].casefold(), item[1]))
    if not candidates:
        raise KengoConversionError(f"No .npz files found under {resolved}")
    relative_lookup: set[str] = set()
    for _, relative_path in candidates:
        lookup_key = relative_path.casefold()
        if lookup_key in relative_lookup:
            raise KengoConversionError(
                f"Input contains paths that collide case-insensitively: {relative_path!r}"
            )
        relative_lookup.add(lookup_key)
    return root, candidates


def _normalize_statuses(statuses: Sequence[str] | str) -> tuple[str, ...]:
    raw_statuses = [statuses] if isinstance(statuses, str) else list(statuses)
    normalized: list[str] = []
    for value in raw_statuses:
        normalized.extend(
            part.upper() for part in re.split(r"[,;\s]+", value.strip()) if part
        )
    if not normalized:
        raise KengoConversionError("At least one quality status must be selected")
    unknown = sorted(set(normalized) - KNOWN_QUALITY_STATUSES)
    if unknown:
        raise KengoConversionError(
            f"Unknown requested quality statuses {unknown}; expected {sorted(KNOWN_QUALITY_STATUSES)}"
        )
    return tuple(
        status for status in ("PASS", "WARN", "FAIL") if status in set(normalized)
    )


def _normalize_sonic_filter_keywords(keywords: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(keyword.strip().casefold() for keyword in keywords)
    if any(not keyword for keyword in normalized):
        raise KengoConversionError("SONIC filter keywords must be non-empty")
    return normalized


def _matching_sonic_filter_keyword(
    relative_path: str, keywords: Sequence[str]
) -> str | None:
    """Return the first official SONIC filename-filter match, if any."""

    path = PurePosixPath(relative_path)
    name_to_check = f"{path.parent.name}/{path.name}".casefold()
    return next((keyword for keyword in keywords if keyword in name_to_check), None)


def _make_input_items(
    candidates: list[tuple[Path, str]],
    quality_records: dict[str, QualityRecord] | None,
    allowed_statuses: tuple[str, ...],
    sonic_filter_keywords: tuple[str, ...] | None,
) -> tuple[list[_InputItem], int, list[dict[str, str]]]:
    items: list[_InputItem] = []
    filtered_by_quality = 0
    filtered_by_sonic_keywords: list[dict[str, str]] = []
    missing_quality_rows: list[str] = []
    keys: dict[str, str] = {}
    for path, relative_path in candidates:
        quality = None
        if quality_records is not None:
            quality = quality_records.get(relative_path.casefold())
            if quality is None:
                missing_quality_rows.append(relative_path)
                continue
            if quality.status not in allowed_statuses:
                filtered_by_quality += 1
                continue
        if sonic_filter_keywords is not None:
            matched_keyword = _matching_sonic_filter_keyword(
                relative_path, sonic_filter_keywords
            )
            if matched_keyword is not None:
                filtered_by_sonic_keywords.append(
                    {"source": relative_path, "matched_keyword": matched_keyword}
                )
                continue
        key = motion_key_from_relative_path(relative_path)
        if key in keys:
            raise KengoConversionError(
                f"Motion key collision: {relative_path!r} and {keys[key]!r} both map to {key!r}"
            )
        keys[key] = relative_path
        items.append(
            _InputItem(path=path, relative_path=relative_path, key=key, quality=quality)
        )

    if missing_quality_rows:
        preview = ", ".join(repr(path) for path in missing_quality_rows[:10])
        remainder = len(missing_quality_rows) - 10
        suffix = f" (and {remainder} more)" if remainder > 0 else ""
        raise KengoConversionError(
            f"Quality CSV has no row for {len(missing_quality_rows)} discovered NPZ files: "
            f"{preview}{suffix}"
        )
    return items, filtered_by_quality, filtered_by_sonic_keywords


def _convert_input_item(item: _InputItem, contract: MjcfContract) -> _ConvertedItem:
    motion = load_kengo_npz(item.path, contract)
    return _ConvertedItem(
        item=item,
        entry=motion_to_motion_lib(motion, contract),
        frames=motion.joint_pos.shape[0],
        fps=motion.framerate,
        source_sha256=_sha256_file(item.path),
    )


def _convert_items(
    items: list[_InputItem], contract: MjcfContract, workers: int
) -> list[_ConvertedItem]:
    results: list[_ConvertedItem | None] = [None] * len(items)
    errors: list[tuple[str, BaseException]] = []

    if workers == 1:
        for index, item in enumerate(items):
            try:
                results[index] = _convert_input_item(item, contract)
            except Exception as exc:  # preserve deterministic aggregate diagnostics
                errors.append((item.relative_path, exc))
    else:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="kengo-npz"
        ) as executor:
            future_to_index: dict[Future[_ConvertedItem], int] = {
                executor.submit(_convert_input_item, item, contract): index
                for index, item in enumerate(items)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # preserve deterministic aggregate diagnostics
                    errors.append((items[index].relative_path, exc))

    if errors:
        errors.sort(key=lambda item: (item[0].casefold(), item[0]))
        diagnostics = "\n".join(
            f"  - {relative_path}: {type(exc).__name__}: {exc}"
            for relative_path, exc in errors
        )
        raise BatchConversionError(
            f"{len(errors)} Kengo NPZ conversion(s) failed:\n{diagnostics}"
        )

    return [result for result in results if result is not None]


def _temporary_sibling(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    return Path(name)


def _atomic_joblib_dump(data: Any, target: Path, *, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    temporary = _temporary_sibling(target)
    try:
        joblib.dump(data, temporary, compress=3)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(data: Any, target: Path, *, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {target}")
    temporary = _temporary_sibling(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing manifest: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def convert_dataset(
    input_path: str | Path,
    output_path: str | Path,
    mjcf_path: str | Path,
    *,
    quality_csv: str | Path | None = None,
    quality_statuses: Sequence[str] | str = DEFAULT_QUALITY_STATUSES,
    apply_sonic_keyword_filter: bool = True,
    sonic_filter_keywords: Sequence[str] = SONIC_DEFAULT_FILTER_KEYWORDS,
    workers: int = 1,
    limit: int | None = None,
    overwrite: bool = False,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recursively convert selected NPZ files and atomically write PKL + manifest."""

    if workers < 1:
        raise KengoConversionError(f"workers must be at least 1; got {workers}")
    if limit is not None and limit < 1:
        raise KengoConversionError(f"limit must be positive; got {limit}")

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".pkl":
        raise KengoConversionError(f"Output must be a .pkl file: {output}")
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    if manifest == output:
        raise KengoConversionError("Manifest path must differ from the output PKL path")
    if not overwrite:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        if manifest.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing manifest: {manifest}"
            )

    contract = parse_mjcf_contract(mjcf_path)
    input_root, candidates = _discover_npz(Path(input_path))
    allowed_statuses = _normalize_statuses(quality_statuses)
    normalized_sonic_keywords = (
        _normalize_sonic_filter_keywords(sonic_filter_keywords)
        if apply_sonic_keyword_filter
        else None
    )
    quality_records = read_quality_csv(quality_csv) if quality_csv is not None else None
    items, filtered_by_quality, filtered_by_sonic_keywords = _make_input_items(
        candidates,
        quality_records,
        allowed_statuses,
        normalized_sonic_keywords,
    )
    selected_before_limit = len(items)
    skipped_by_limit = 0
    if limit is not None and len(items) > limit:
        skipped_by_limit = len(items) - limit
        items = items[:limit]
    if not items:
        raise KengoConversionError(
            "No Kengo NPZ files remain after applying the quality filter and limit"
        )

    converted = _convert_items(items, contract, workers)
    motion_lib = {result.item.key: result.entry for result in converted}
    _atomic_joblib_dump(motion_lib, output, overwrite=overwrite)

    motion_records = []
    for result in converted:
        quality = result.item.quality
        motion_records.append(
            {
                "key": result.item.key,
                "source": result.item.relative_path,
                "source_sha256": result.source_sha256,
                "frames": result.frames,
                "fps": result.fps,
                "quality_status": quality.status if quality is not None else None,
                "quality_reasons": (
                    [reason for reason in quality.reasons.split(";") if reason]
                    if quality is not None
                    else []
                ),
            }
        )

    manifest_data: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": "sonic_motion_lib",
        "input_root": str(input_root),
        "output": str(output),
        "mjcf": {
            "path": str(contract.path),
            "sha256": contract.sha256,
            "root_body": contract.root_body_name,
            "body_names_dfs": list(contract.body_names),
            "joint_names_dfs": list(contract.joint_names),
            "joint_axes_dfs": contract.joint_axes.tolist(),
        },
        "quality_filter": {
            "csv": str(Path(quality_csv).expanduser().resolve())
            if quality_csv is not None
            else None,
            "allowed_statuses": list(allowed_statuses),
        },
        "sonic_keyword_filter": {
            "enabled": apply_sonic_keyword_filter,
            "keywords": list(normalized_sonic_keywords or ()),
            "excluded": filtered_by_sonic_keywords,
        },
        "counts": {
            "discovered": len(candidates),
            "selected_before_limit": selected_before_limit,
            "filtered_by_quality": filtered_by_quality,
            "filtered_by_sonic_keywords": len(filtered_by_sonic_keywords),
            "skipped_by_limit": skipped_by_limit,
            "written": len(converted),
            "frames": sum(result.frames for result in converted),
        },
        "motions": motion_records,
    }
    _atomic_json_dump(manifest_data, manifest, overwrite=overwrite)
    return manifest_data


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert five-field retargeted Kengo NPZ files to a SONIC motion-lib PKL"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ file or recursively scanned directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Combined output .pkl file"
    )
    parser.add_argument(
        "--mjcf", type=Path, required=True, help="Clean 24-body Kengo MJCF"
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        help="Optional per_file_quality.csv; discovered files must all have a row",
    )
    parser.add_argument(
        "--quality-statuses",
        nargs="+",
        default=list(DEFAULT_QUALITY_STATUSES),
        help="Statuses retained when --quality-csv is set (default: PASS WARN)",
    )
    parser.add_argument(
        "--no-sonic-keyword-filter",
        dest="apply_sonic_keyword_filter",
        action="store_false",
        help=(
            "Disable the official SONIC filename-keyword filter for furniture, "
            "vehicles, sitting, acrobatics, and elevated-surface motions"
        ),
    )
    parser.add_argument(
        "--sonic-filter-keywords",
        nargs="+",
        default=list(SONIC_DEFAULT_FILTER_KEYWORDS),
        help="Filename keywords excluded by the SONIC filter",
    )
    parser.add_argument(
        "--workers",
        "--num-workers",
        dest="workers",
        type=_positive_int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="Parallel NPZ loading/conversion workers",
    )
    parser.add_argument(
        "--limit", type=_positive_int, help="Convert only the first N selected paths"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Atomically replace existing outputs"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest JSON path (default: <output>.manifest.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = convert_dataset(
        input_path=args.input,
        output_path=args.output,
        mjcf_path=args.mjcf,
        quality_csv=args.quality_csv,
        quality_statuses=args.quality_statuses,
        apply_sonic_keyword_filter=args.apply_sonic_keyword_filter,
        sonic_filter_keywords=args.sonic_filter_keywords,
        workers=args.workers,
        limit=args.limit,
        overwrite=args.overwrite,
        manifest_path=args.manifest,
    )
    counts = manifest["counts"]
    print(
        f"Wrote {counts['written']} motions / {counts['frames']} frames to {manifest['output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
