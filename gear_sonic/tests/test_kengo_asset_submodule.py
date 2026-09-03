from __future__ import annotations

import configparser
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_PATH = Path("external_dependencies/kengo_robot_description")
ASSET_ROOT = REPO_ROOT / SUBMODULE_PATH
LOCK_PATH = REPO_ROOT / "KENGO_ASSETS.lock.json"


def test_kengo_asset_gitlink_matches_lock() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["submodule_path"] == SUBMODULE_PATH.as_posix()
    assert lock["classification"] == "restricted"
    assert lock["redistributable"] is False

    modules = configparser.ConfigParser()
    modules.read(REPO_ROOT / ".gitmodules", encoding="utf-8")
    section = f'submodule "{SUBMODULE_PATH.as_posix()}"'
    assert modules[section]["path"] == SUBMODULE_PATH.as_posix()
    assert modules[section]["url"] == lock["repository"]

    indexed = subprocess.run(
        ["git", "ls-files", "--stage", "--", SUBMODULE_PATH.as_posix()],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    mode, commit, stage_and_path = indexed.split(maxsplit=2)
    assert mode == "160000"
    assert commit == lock["commit"]
    assert stage_and_path == f"0\t{SUBMODULE_PATH.as_posix()}"


def test_initialized_kengo_asset_bundle_validates() -> None:
    validator = ASSET_ROOT / "scripts" / "validate_assets.py"
    if not validator.is_file():
        pytest.skip("private Kengo asset submodule is not initialized")
    completed = subprocess.run(
        [sys.executable, str(validator)],
        cwd=ASSET_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["valid"] is True
    assert result["actuated_joint_count"] == 23
    assert result["mesh_count"] == 27


def test_legacy_scattered_kengo_assets_are_absent() -> None:
    legacy_root = REPO_ROOT / "gear_sonic/data/assets/robot_description"
    assert not (legacy_root / "urdf/kengo").exists()
    assert not (legacy_root / "meshes/kengo").exists()
    assert not (legacy_root / "mjcf/kengo_with_fist.xml").exists()
