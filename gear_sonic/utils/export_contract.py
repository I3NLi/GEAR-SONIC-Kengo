"""Small, dependency-free helpers for embodiment-aware policy exports."""

from __future__ import annotations

from pathlib import Path


_EXPORT_TAG_ALIASES = {
    "galaxea_kengo": "kengo",
    "unitree_g1": "g1",
    "g1_model_12_dex": "g1",
    "unitree_h2": "h2",
}


def normalize_export_tag(robot_type: str) -> str:
    """Return a stable filename tag without changing model-internal ABI keys."""

    normalized = str(robot_type).strip().casefold().replace("-", "_")
    if not normalized:
        raise ValueError("robot_type must be non-empty for ONNX export")
    tag = _EXPORT_TAG_ALIASES.get(normalized, normalized)
    if not tag.replace("_", "").isalnum():
        raise ValueError(f"robot_type contains unsafe filename characters: {robot_type!r}")
    return tag


def combined_policy_onnx_name(base_name: str, robot_type: str) -> str:
    """Name the combined robot-policy ONNX for its physical embodiment."""

    path = Path(base_name)
    if path.suffix.casefold() != ".onnx":
        raise ValueError(f"expected an .onnx base name, got {base_name!r}")
    return f"{path.stem}_{normalize_export_tag(robot_type)}.onnx"
