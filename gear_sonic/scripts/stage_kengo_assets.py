#!/usr/bin/env python3
"""Validate the pinned private Kengo asset submodule.

This legacy entry-point name is retained for operator compatibility. It no
longer copies assets from a sibling checkout: the parent repository must use
the exact private submodule commit recorded in its gitlink.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_PATH = Path("external_dependencies/kengo_robot_description")
DEFAULT_ASSET_ROOT = REPO_ROOT / SUBMODULE_PATH
LOCK_PATH = REPO_ROOT / "KENGO_ASSETS.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(asset_root: Path) -> dict[str, object]:
    asset_root = asset_root.expanduser().resolve()
    validator = asset_root / "scripts" / "validate_assets.py"
    if not validator.is_file():
        raise FileNotFoundError(
            "Kengo asset submodule is not initialized. Run: "
            "git submodule update --init --recursive "
            "external_dependencies/kengo_robot_description"
        )

    if asset_root == DEFAULT_ASSET_ROOT.resolve():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if lock.get("submodule_path") != SUBMODULE_PATH.as_posix():
            raise ValueError("Kengo asset lock has an unexpected submodule path")
        status = subprocess.run(
            ["git", "submodule", "status", "--", SUBMODULE_PATH.as_posix()],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        line = status.stdout.rstrip("\r\n")
        if status.returncode != 0 or not line or line[0] in "-+U":
            detail = (status.stderr or status.stdout).strip()
            raise RuntimeError(f"Kengo submodule is not at its pinned commit: {detail}")
        actual_commit = line[1:].split(maxsplit=1)[0]
        if actual_commit != lock.get("commit"):
            raise ValueError(
                f"Kengo asset lock commit mismatch: {actual_commit} != {lock.get('commit')}"
            )
        contract = asset_root / str(lock.get("asset_contract", ""))
        actual_contract_sha = _sha256(contract)
        if actual_contract_sha != lock.get("asset_contract_sha256"):
            raise ValueError(
                "Kengo asset contract SHA-256 mismatch: "
                f"{actual_contract_sha} != {lock.get('asset_contract_sha256')}"
            )

    completed = subprocess.run(
        [sys.executable, str(validator)],
        cwd=asset_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    args = parser.parse_args()
    result = verify(args.asset_root)
    result["asset_root"] = str(args.asset_root.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
