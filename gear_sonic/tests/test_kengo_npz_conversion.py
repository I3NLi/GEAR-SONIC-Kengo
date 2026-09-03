from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
import pytest

from gear_sonic.data_process.convert_kengo_npz_to_motion_lib import (
    BatchConversionError,
    KengoConversionError,
    convert_dataset,
    convert_npz,
    load_kengo_npz,
    motion_key_from_relative_path,
    parse_mjcf_contract,
)
from gear_sonic.utils.kengo_contract import KENGO_MUJOCO_JOINT_NAMES

TEST_AXES = np.asarray(
    [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, -3.0]] * 8,
    dtype=np.float32,
)[:23]
TEST_AXES /= np.linalg.norm(TEST_AXES, axis=1, keepdims=True)


def _body_name(joint_name: str) -> str:
    if joint_name == "waist_yaw_joint":
        return "pelvis_link"
    return joint_name.removesuffix("_joint") + "_link"


def _write_mjcf(path: Path, *, swap_actuators: bool = False) -> Path:
    model = ET.Element("mujoco", model="kengo_test")
    ET.SubElement(model, "compiler", angle="radian")
    worldbody = ET.SubElement(model, "worldbody")
    torso = ET.SubElement(worldbody, "body", name="torso_link")
    ET.SubElement(torso, "joint", name="floating_base_joint", type="free")

    for joint_name, axis in zip(KENGO_MUJOCO_JOINT_NAMES, TEST_AXES, strict=True):
        body = ET.SubElement(torso, "body", name=_body_name(joint_name))
        ET.SubElement(
            body,
            "joint",
            name=joint_name,
            type="hinge",
            axis=" ".join(str(float(value)) for value in axis),
        )

    actuator = ET.SubElement(model, "actuator")
    actuator_names = list(KENGO_MUJOCO_JOINT_NAMES)
    if swap_actuators:
        actuator_names[0], actuator_names[1] = actuator_names[1], actuator_names[0]
    for joint_name in actuator_names:
        ET.SubElement(actuator, "motor", name=joint_name, joint=joint_name)

    ET.ElementTree(model).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _motion_arrays(frames: int = 3) -> dict[str, np.ndarray]:
    joint_pos = np.arange(frames * 23, dtype=np.float64).reshape(frames, 23) / 100.0
    base_pos_w = np.arange(frames * 3, dtype=np.float64).reshape(frames, 3) / 10.0
    half_sqrt = np.sqrt(0.5)
    base_quat_w = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half_sqrt, 0.0, 0.0, half_sqrt],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    if frames != 3:
        base_quat_w = np.tile(base_quat_w[:1], (frames, 1))
    return {
        "joint_pos": joint_pos,
        "base_pos_w": base_pos_w,
        "base_quat_w": base_quat_w,
    }


def _write_npz(
    path: Path,
    *,
    frames: int = 3,
    joint_names: list[str] | None = None,
    quaternion_scale: float = 1.0,
    extra_field: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _motion_arrays(frames)
    fields = {
        "framerate": np.asarray(30.0003, dtype=np.float64),
        "joint_names": np.asarray(
            joint_names if joint_names is not None else KENGO_MUJOCO_JOINT_NAMES
        ),
        "joint_pos": arrays["joint_pos"],
        "base_pos_w": arrays["base_pos_w"],
        "base_quat_w": arrays["base_quat_w"] * quaternion_scale,
    }
    if extra_field:
        fields["unexpected"] = np.asarray(1)
    np.savez(path, **fields)
    return path


def _write_quality_csv(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["relative_path", "status", "reasons"]
        )
        writer.writeheader()
        for relative_path, status, reasons in rows:
            writer.writerow(
                {"relative_path": relative_path, "status": status, "reasons": reasons}
            )
    return path


def test_parse_and_convert_exact_kengo_contract(tmp_path: Path) -> None:
    mjcf = _write_mjcf(tmp_path / "kengo.xml")
    source = _write_npz(tmp_path / "walk.npz")

    contract = parse_mjcf_contract(mjcf)
    entry = convert_npz(source, contract)
    source_arrays = _motion_arrays()

    assert contract.root_body_name == "torso_link"
    assert contract.joint_names == tuple(KENGO_MUJOCO_JOINT_NAMES)
    assert contract.body_names[11] == "pelvis_link"
    assert contract.joint_axes.shape == (23, 3)
    np.testing.assert_allclose(contract.joint_axes, TEST_AXES)

    assert set(entry) == {
        "root_trans_offset",
        "pose_aa",
        "dof",
        "root_rot",
        "smpl_joints",
        "fps",
    }
    assert entry["root_trans_offset"].shape == (3, 3)
    assert entry["pose_aa"].shape == (3, 24, 3)
    assert entry["dof"].shape == (3, 23)
    assert entry["root_rot"].shape == (3, 4)
    assert entry["smpl_joints"].shape == (3, 24, 3)
    for field in ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints"):
        assert entry[field].dtype == np.float32

    np.testing.assert_allclose(entry["root_trans_offset"], source_arrays["base_pos_w"])
    np.testing.assert_allclose(entry["dof"], source_arrays["joint_pos"])
    np.testing.assert_allclose(
        entry["pose_aa"][:, 1:, :],
        source_arrays["joint_pos"][:, :, None] * TEST_AXES[None, :, :],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        entry["pose_aa"][:, 0, :],
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], [0.0, 0.0, 0.0]]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        entry["root_rot"], source_arrays["base_quat_w"][:, [1, 2, 3, 0]], atol=1e-7
    )
    np.testing.assert_array_equal(
        entry["smpl_joints"], np.zeros((3, 24, 3), dtype=np.float32)
    )
    assert entry["fps"] == pytest.approx(30.0003)


def test_npz_schema_and_joint_order_are_strict(tmp_path: Path) -> None:
    contract = parse_mjcf_contract(_write_mjcf(tmp_path / "kengo.xml"))

    swapped_names = list(KENGO_MUJOCO_JOINT_NAMES)
    swapped_names[0], swapped_names[1] = swapped_names[1], swapped_names[0]
    swapped = _write_npz(tmp_path / "swapped.npz", joint_names=swapped_names)
    with pytest.raises(KengoConversionError, match=r"joint_names\[0\].*expected"):
        load_kengo_npz(swapped, contract)

    extra = _write_npz(tmp_path / "extra.npz", extra_field=True)
    with pytest.raises(KengoConversionError, match="unexpected=.*unexpected"):
        load_kengo_npz(extra, contract)

    unnormalized = _write_npz(tmp_path / "bad_quaternion.npz", quaternion_scale=2.0)
    with pytest.raises(KengoConversionError, match="norm is"):
        load_kengo_npz(unnormalized, contract)


def test_mjcf_actuator_order_must_match_dfs_contract(tmp_path: Path) -> None:
    mjcf = _write_mjcf(tmp_path / "bad_actuator.xml", swap_actuators=True)
    with pytest.raises(KengoConversionError, match=r"actuator joint order\[0\]"):
        parse_mjcf_contract(mjcf)


def test_recursive_quality_filter_unique_keys_manifest_and_overwrite(
    tmp_path: Path,
) -> None:
    mjcf = _write_mjcf(tmp_path / "kengo.xml")
    input_root = tmp_path / "motions"
    _write_npz(input_root / "a" / "same.npz", frames=2)
    _write_npz(input_root / "b" / "same.npz", frames=4)
    _write_npz(input_root / "c" / "drop.npz", frames=5)
    quality_csv = _write_quality_csv(
        tmp_path / "quality.csv",
        [
            ("a/same.npz", "PASS", ""),
            ("b\\same.npz", "warn", "foot_slip;joint_velocity"),
            ("c/drop.npz", "FAIL", "bad_clip"),
        ],
    )
    output = tmp_path / "converted" / "kengo.pkl"

    manifest = convert_dataset(
        input_root,
        output,
        mjcf,
        quality_csv=quality_csv,
        workers=2,
    )

    key_a = motion_key_from_relative_path("a/same.npz")
    key_b = motion_key_from_relative_path("b/same.npz")
    assert key_a != key_b
    assert list(joblib.load(output)) == [key_a, key_b]
    assert manifest["counts"] == {
        "discovered": 3,
        "selected_before_limit": 2,
        "filtered_by_quality": 1,
        "filtered_by_sonic_keywords": 0,
        "skipped_by_limit": 0,
        "written": 2,
        "frames": 6,
    }
    assert [motion["source"] for motion in manifest["motions"]] == [
        "a/same.npz",
        "b/same.npz",
    ]
    assert manifest["motions"][1]["quality_status"] == "WARN"
    assert manifest["motions"][1]["quality_reasons"] == ["foot_slip", "joint_velocity"]

    manifest_path = output.with_suffix(".pkl.manifest.json")
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert not list(tmp_path.rglob("*.tmp"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        convert_dataset(input_root, output, mjcf, quality_csv=quality_csv)

    limited = convert_dataset(
        input_root,
        output,
        mjcf,
        quality_csv=quality_csv,
        workers=2,
        limit=1,
        overwrite=True,
    )
    assert list(joblib.load(output)) == [key_a]
    assert limited["counts"]["selected_before_limit"] == 2
    assert limited["counts"]["skipped_by_limit"] == 1
    assert limited["counts"]["written"] == 1


def test_official_sonic_filename_filter_is_enabled_and_auditable(
    tmp_path: Path,
) -> None:
    mjcf = _write_mjcf(tmp_path / "kengo.xml")
    input_root = tmp_path / "motions"
    kept = _write_npz(input_root / "walk" / "walk_001.npz")
    dropped = _write_npz(input_root / "acrobatics" / "cartwheel_R_003_retargeted.npz")
    output = tmp_path / "converted.pkl"

    manifest = convert_dataset(input_root, output, mjcf, workers=1)

    assert list(joblib.load(output)) == [
        motion_key_from_relative_path(kept.relative_to(input_root).as_posix())
    ]
    assert manifest["counts"]["filtered_by_sonic_keywords"] == 1
    assert manifest["sonic_keyword_filter"]["enabled"] is True
    assert manifest["sonic_keyword_filter"]["excluded"] == [
        {
            "source": dropped.relative_to(input_root).as_posix(),
            "matched_keyword": "cartwheel",
        }
    ]

    unfiltered = convert_dataset(
        input_root,
        output,
        mjcf,
        apply_sonic_keyword_filter=False,
        overwrite=True,
    )
    assert unfiltered["counts"]["filtered_by_sonic_keywords"] == 0
    assert unfiltered["counts"]["written"] == 2


def test_batch_validation_failure_writes_no_partial_output(tmp_path: Path) -> None:
    mjcf = _write_mjcf(tmp_path / "kengo.xml")
    input_root = tmp_path / "motions"
    _write_npz(input_root / "good.npz")
    _write_npz(input_root / "bad.npz", quaternion_scale=2.0)
    output = tmp_path / "converted.pkl"

    with pytest.raises(BatchConversionError, match="bad.npz"):
        convert_dataset(input_root, output, mjcf, workers=2)

    assert not output.exists()
    assert not output.with_suffix(".pkl.manifest.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))
