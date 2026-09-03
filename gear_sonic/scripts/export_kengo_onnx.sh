#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

checkpoint="${1:-}"
if [[ -z "$checkpoint" ]]; then
    printf 'Usage: %s /absolute/path/to/checkpoint.pt\n' "$0" >&2
    exit 2
fi
if [[ ! -s "$checkpoint" ]]; then
    printf 'Checkpoint is missing or empty: %s\n' "$checkpoint" >&2
    exit 1
fi

python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    printf 'Missing evaluation interpreter: %s\n' "$python_bin" >&2
    exit 1
fi

"$python_bin" gear_sonic/scripts/stage_kengo_assets.py

"$python_bin" gear_sonic/eval_agent_trl.py \
    "+checkpoint=$checkpoint" \
    +headless=True \
    ++num_envs=1 \
    ++export_onnx_only=true \
    ++export_primary_onnx_only=true \
    ++export_embodiment=kengo

export_dir="$(dirname "$checkpoint")/exported"
mapfile -t exported_models < <(
    find "$export_dir" -maxdepth 1 -type f -name 'model_step_*_kengo.onnx' -printf '%T@ %p\n' \
        | sort -nr \
        | cut -d' ' -f2-
)
if (( ${#exported_models[@]} == 0 )); then
    printf 'No combined Kengo ONNX was produced under %s\n' "$export_dir" >&2
    exit 1
fi

printf 'Combined Kengo ONNX: %s\n' "${exported_models[0]}"
sha256sum "${exported_models[0]}"
