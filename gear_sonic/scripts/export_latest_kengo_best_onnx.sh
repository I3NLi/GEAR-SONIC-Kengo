#!/usr/bin/env bash
set -Eeuo pipefail

# Export the checkpoint selected by backup_best_checkpoints.py without racing
# its atomic best_so_far.pt replacement.  Successful invocations write exactly
# one machine-readable line to stdout; diagnostics and exporter logs stay on
# stderr so an SSH caller can parse the final result reliably.

umask 077

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
backup_root="$repo_root/checkpoint_backups/hourly_best"
best_metadata="$backup_root/best_so_far.json"
best_checkpoint="$backup_root/best_so_far.pt"
export_root="$repo_root/sim2sim_exports"
canonical_dir=""
export_script="$repo_root/gear_sonic/scripts/export_kengo_onnx.sh"
python_bin="$repo_root/.venv/bin/python"
training_pid_file="$repo_root/run_logs/kengo_full8gpu.pid"
minimum_free_gpu_mib=8192

temporary_bundle=""
publish_temporary=""

cleanup() {
    if [[ -n "$publish_temporary" && -f "$publish_temporary" ]]; then
        rm -f -- "$publish_temporary"
    fi
    if [[ -n "$temporary_bundle" && -d "$temporary_bundle" ]]; then
        rm -rf -- "$temporary_bundle"
    fi
}
trap cleanup EXIT

fail() {
    local exit_code="${2:-1}"
    printf '[ERROR] %s\n' "$1" >&2
    exit "$exit_code"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command is unavailable: $1"
}

for required_command in flock nvidia-smi nice pgrep timeout; do
    require_command "$required_command"
done
[[ -x "$python_bin" ]] || fail "Missing Kengo Python interpreter: $python_bin"
[[ -f "$export_script" ]] || fail "Missing Kengo ONNX exporter: $export_script"
[[ -f "$best_metadata" ]] || fail "Missing best-checkpoint metadata: $best_metadata"
[[ -f "$best_checkpoint" && ! -L "$best_checkpoint" ]] \
    || fail "Best checkpoint must be a regular non-symlink file: $best_checkpoint"

mkdir -p -- "$export_root"
[[ -d "$export_root" && ! -L "$export_root" ]] \
    || fail "Sim2sim export root must be a real directory: $export_root"
exec 9>"$backup_root/.onnx_export.lock"
flock -n 9 || fail "Another Kengo best-ONNX export is already running" 75

# Globals populated by load_best_metadata.
best_step=""
best_step_padded=""
best_sha256=""
best_bytes=""
source_run_dir=""

load_best_metadata() {
    local parsed
    if ! parsed="$($python_bin - "$best_metadata" "$best_checkpoint" "$repo_root" <<'PY'
import json
from pathlib import Path
import re
import sys

metadata_path = Path(sys.argv[1]).resolve(strict=True)
best_path = Path(sys.argv[2]).resolve(strict=True)
repo_root = Path(sys.argv[3]).resolve(strict=True)

with metadata_path.open(encoding="utf-8") as stream:
    data = json.load(stream)

step = data.get("step")
if isinstance(step, bool) or not isinstance(step, int) or step < 1:
    raise ValueError(f"invalid best checkpoint step: {step!r}")
checkpoint_sha = str(data.get("sha256", "")).lower()
if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is None:
    raise ValueError("best checkpoint metadata has an invalid sha256")
checkpoint_bytes = data.get("bytes")
if (
    isinstance(checkpoint_bytes, bool)
    or not isinstance(checkpoint_bytes, int)
    or checkpoint_bytes < 1
):
    raise ValueError(f"invalid best checkpoint byte count: {checkpoint_bytes!r}")

cached = Path(str(data.get("cached_checkpoint", "")))
if not cached.is_absolute():
    cached = repo_root / cached
if cached.resolve(strict=False) != best_path:
    raise ValueError(
        f"metadata cached_checkpoint does not identify {best_path}: {cached}"
    )

source = Path(str(data.get("source_checkpoint", "")))
if not source.is_absolute():
    source = repo_root / source
source = source.resolve(strict=False)
try:
    source.relative_to(repo_root)
except ValueError as exc:
    raise ValueError(f"source checkpoint escapes repository: {source}") from exc
source_run = source.parent.resolve(strict=True)
config = source_run / "config.yaml"
if not config.is_file():
    raise ValueError(f"source run has no config.yaml: {config}")
for value in (step, f"{step:06d}", checkpoint_sha, checkpoint_bytes, source_run):
    text = str(value)
    if "\n" in text or "\r" in text:
        raise ValueError("metadata paths must not contain newlines")
    print(text)
PY
)"; then
        return 1
    fi
    local fields=()
    mapfile -t fields <<<"$parsed"
    [[ "${#fields[@]}" -eq 5 ]] || return 1
    best_step="${fields[0]}"
    best_step_padded="${fields[1]}"
    best_sha256="${fields[2]}"
    best_bytes="${fields[3]}"
    source_run_dir="${fields[4]}"
}

best_binding() {
    "$python_bin" - "$best_metadata" <<'PY'
import json
from pathlib import Path
import sys

with Path(sys.argv[1]).open(encoding="utf-8") as stream:
    data = json.load(stream)
print(f"{data.get('step')}:{str(data.get('sha256', '')).lower()}")
PY
}

file_sha_and_bytes() {
    "$python_bin" - "$1" <<'PY'
from pathlib import Path
import hashlib
import sys

path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as stream:
    while chunk := stream.read(4 * 1024 * 1024):
        digest.update(chunk)
print(digest.hexdigest(), path.stat().st_size)
PY
}

validate_onnx() {
    "$python_bin" - "$1" <<'PY'
from pathlib import Path
import hashlib
import sys

import numpy as np
import onnx
import onnxruntime as ort

path = Path(sys.argv[1]).resolve(strict=True)
if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
    raise ValueError(f"ONNX must be a non-empty regular non-symlink file: {path}")

model = onnx.load(str(path), load_external_data=True)
onnx.checker.check_model(model)

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session = ort.InferenceSession(
    str(path), sess_options=options, providers=["CPUExecutionProvider"]
)
inputs = session.get_inputs()
outputs = session.get_outputs()
if len(inputs) != 1 or len(outputs) < 1:
    raise ValueError(
        f"combined Kengo ONNX must have one input and at least one output, "
        f"got {len(inputs)} and {len(outputs)}"
    )
policy_input = inputs[0]
policy_output = outputs[0]
if policy_input.type != "tensor(float)" or policy_output.type != "tensor(float)":
    raise ValueError(
        f"combined Kengo ONNX must use float32, got "
        f"{policy_input.type} -> {policy_output.type}"
    )
if list(policy_input.shape) != [1, 1270] or list(policy_output.shape) != [1, 23]:
    raise ValueError(
        f"combined Kengo ONNX shape mismatch: "
        f"{policy_input.shape} -> {policy_output.shape}"
    )
result = session.run(
    [policy_output.name],
    {policy_input.name: np.zeros((1, 1270), dtype=np.float32)},
)[0]
result = np.asarray(result)
if result.shape != (1, 23) or not np.isfinite(result).all():
    raise ValueError(
        f"combined Kengo ONNX returned invalid output: "
        f"shape={result.shape}, finite={bool(np.isfinite(result).all())}"
    )

digest = hashlib.sha256()
with path.open("rb") as stream:
    while chunk := stream.read(4 * 1024 * 1024):
        digest.update(chunk)
print(digest.hexdigest(), path.stat().st_size)
PY
}

write_manifest() {
    local manifest_path="$1"
    local onnx_path="$2"
    local onnx_sha="$3"
    local onnx_bytes="$4"
    local provenance="$5"
    "$python_bin" - \
        "$manifest_path" "$best_step" "$best_sha256" "$onnx_path" \
        "$onnx_sha" "$onnx_bytes" "$provenance" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
import tempfile

destination = Path(sys.argv[1])
data = {
    "schema_version": 1,
    "step": int(sys.argv[2]),
    "checkpoint_sha256": sys.argv[3],
    "onnx_path": str(Path(sys.argv[4]).resolve(strict=True)),
    "onnx_sha256": sys.argv[5],
    "bytes": int(sys.argv[6]),
    "provenance": sys.argv[7],
    "recorded_at": datetime.now(timezone.utc).isoformat(),
}
descriptor, temporary_name = tempfile.mkstemp(
    dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
PY
}

validate_manifest_or_adopt() {
    local manifest_path="$1"
    local onnx_path="$2"
    local onnx_sha="$3"
    local onnx_bytes="$4"
    if [[ ! -e "$manifest_path" ]]; then
        printf '[WARN] Adopting validated legacy canonical ONNX without a manifest: %s\n' \
            "$onnx_path" >&2
        write_manifest \
            "$manifest_path" "$onnx_path" "$onnx_sha" "$onnx_bytes" \
            "adopted_validated_canonical"
        return
    fi
    [[ -f "$manifest_path" && ! -L "$manifest_path" ]] \
        || fail "ONNX manifest must be a regular non-symlink file: $manifest_path"
    "$python_bin" - \
        "$manifest_path" "$best_step" "$best_sha256" "$onnx_path" \
        "$onnx_sha" "$onnx_bytes" <<'PY'
from pathlib import Path
import json
import sys

manifest_path = Path(sys.argv[1])
with manifest_path.open(encoding="utf-8") as stream:
    data = json.load(stream)
expected = {
    "step": int(sys.argv[2]),
    "checkpoint_sha256": sys.argv[3],
    "onnx_path": str(Path(sys.argv[4]).resolve(strict=True)),
    "onnx_sha256": sys.argv[5],
    "bytes": int(sys.argv[6]),
}
mismatches = {
    key: {"manifest": data.get(key), "expected": value}
    for key, value in expected.items()
    if data.get(key) != value
}
if mismatches:
    raise ValueError(
        "canonical ONNX conflicts with its checkpoint/artifact manifest: "
        + json.dumps(mismatches, sort_keys=True)
    )
PY
}

emit_result() {
    local exported="$1"
    local gpu="$2"
    local onnx_path="$3"
    local onnx_sha="$4"
    local onnx_bytes="$5"
    "$python_bin" - \
        "$best_step" "$best_sha256" "$onnx_path" "$onnx_sha" \
        "$onnx_bytes" "$exported" "$gpu" <<'PY'
import json
import sys

result = {
    "step": int(sys.argv[1]),
    "checkpoint_sha256": sys.argv[2],
    "onnx_path": sys.argv[3],
    "onnx_sha256": sys.argv[4],
    "bytes": int(sys.argv[5]),
    "exported": sys.argv[6] == "true",
    "gpu": int(sys.argv[7]) if sys.argv[7] else None,
}
print("[RESULT_JSON] " + json.dumps(result, sort_keys=True))
PY
}

if ! load_best_metadata; then
    fail "Could not parse stable best-checkpoint metadata"
fi
initial_binding="$best_step:$best_sha256"
canonical_dir="$export_root/kengo_best_step_${best_step}"
mkdir -p -- "$canonical_dir"
[[ -d "$canonical_dir" && ! -L "$canonical_dir" ]] \
    || fail "Canonical step directory must be a real directory: $canonical_dir"
canonical_onnx="$canonical_dir/model_step_${best_step_padded}_kengo.onnx"
canonical_manifest="${canonical_onnx}.manifest.json"

if [[ -e "$canonical_onnx" ]]; then
    [[ -f "$canonical_onnx" && ! -L "$canonical_onnx" ]] \
        || fail "Canonical ONNX is not a regular non-symlink file: $canonical_onnx"
    if ! validation="$(validate_onnx "$canonical_onnx")"; then
        fail "Existing canonical ONNX failed validation: $canonical_onnx"
    fi
    read -r canonical_sha canonical_bytes extra <<<"$validation"
    [[ -n "$canonical_sha" && -n "$canonical_bytes" && -z "${extra:-}" ]] \
        || fail "Could not parse canonical ONNX validation output"
    [[ "$(best_binding)" == "$initial_binding" ]] \
        || fail "Best checkpoint changed while validating its canonical ONNX; retry"
    validate_manifest_or_adopt \
        "$canonical_manifest" "$canonical_onnx" "$canonical_sha" "$canonical_bytes"
    emit_result false "" "$canonical_onnx" "$canonical_sha" "$canonical_bytes"
    exit 0
fi

# Pin the exact inode named by best_so_far.json.  The backup service may replace
# best_so_far.pt at any moment; a hard link keeps this export bound to one inode.
snapshot_checkpoint=""
for attempt in 1 2 3; do
    if [[ "$attempt" -gt 1 ]]; then
        sleep 1
        if ! load_best_metadata; then
            continue
        fi
        initial_binding="$best_step:$best_sha256"
        canonical_dir="$export_root/kengo_best_step_${best_step}"
        mkdir -p -- "$canonical_dir"
        [[ -d "$canonical_dir" && ! -L "$canonical_dir" ]] \
            || fail "Canonical step directory must be a real directory: $canonical_dir"
        canonical_onnx="$canonical_dir/model_step_${best_step_padded}_kengo.onnx"
        canonical_manifest="${canonical_onnx}.manifest.json"
        if [[ -e "$canonical_onnx" ]]; then
            fail "Canonical ONNX appeared during best-checkpoint snapshot; retry"
        fi
    fi
    temporary_bundle="$(mktemp -d \
        "$source_run_dir/.daily_kengo_export_${best_step_padded}_${best_sha256:0:12}.XXXXXX")"
    snapshot_checkpoint="$temporary_bundle/model_step_${best_step_padded}.pt"
    if ! ln -- "$best_checkpoint" "$snapshot_checkpoint"; then
        fail "Could not hard-link the selected best checkpoint into its source run"
    fi
    if ! snapshot_identity="$(file_sha_and_bytes "$snapshot_checkpoint")"; then
        fail "Could not hash the pinned best-checkpoint inode"
    fi
    read -r snapshot_sha snapshot_bytes extra <<<"$snapshot_identity"
    if [[ "$snapshot_sha" == "$best_sha256" && "$snapshot_bytes" == "$best_bytes" ]]; then
        break
    fi
    printf '[WARN] best_so_far.pt changed during snapshot attempt %s; retrying\n' \
        "$attempt" >&2
    rm -rf -- "$temporary_bundle"
    temporary_bundle=""
    snapshot_checkpoint=""
done
[[ -n "$snapshot_checkpoint" && -f "$snapshot_checkpoint" ]] \
    || fail "Could not obtain a checkpoint inode matching best_so_far.json after 3 attempts"

training_pids=""
if training_pids="$(pgrep -f '[t]rain_agent_trl.py.*sonic_kengo')"; then
    training_pids="$(printf '%s\n' "$training_pids" | sort -n -u)"
else
    pgrep_status=$?
    [[ "$pgrep_status" -eq 1 ]] \
        || fail "Could not inspect Kengo training processes" 75
    training_pids=""
fi

training_launcher_pid=""
training_snapshot=""
if [[ -n "$training_pids" ]]; then
    [[ -r "$training_pid_file" ]] \
        || fail "Training is active but its launcher PID snapshot is unavailable" 75
    training_launcher_pid="$(tr -d '[:space:]' <"$training_pid_file")"
    [[ "$training_launcher_pid" =~ ^[1-9][0-9]*$ ]] \
        || fail "Training launcher PID file is invalid: $training_pid_file" 75
    kill -0 "$training_launcher_pid" 2>/dev/null \
        || fail "Training launcher PID is no longer alive: $training_launcher_pid" 75
    while IFS= read -r training_pid; do
        [[ "$training_pid" =~ ^[1-9][0-9]*$ && -r "/proc/$training_pid/stat" ]] \
            || fail "Training PID snapshot is not readable: $training_pid" 75
        proc_stat="$(<"/proc/$training_pid/stat")"
        proc_fields="${proc_stat##*) }"
        read -r -a proc_field_array <<<"$proc_fields"
        [[ "${#proc_field_array[@]}" -ge 20 ]] \
            || fail "Training PID start-time snapshot is incomplete: $training_pid" 75
        training_snapshot+="$training_pid:${proc_field_array[19]};"
    done <<<"$training_pids"
fi

selected_gpu=""
selected_gpu_free_mib=-1
if ! gpu_inventory="$(nvidia-smi \
    --query-gpu=index,memory.free --format=csv,noheader,nounits)"; then
    fail "Could not query GPU free memory" 75
fi
while IFS=',' read -r gpu_index gpu_free_mib; do
    gpu_index="${gpu_index//[[:space:]]/}"
    gpu_free_mib="${gpu_free_mib//[[:space:]]/}"
    [[ "$gpu_index" =~ ^[0-9]+$ && "$gpu_free_mib" =~ ^[0-9]+$ ]] || continue
    if (( gpu_free_mib > selected_gpu_free_mib )); then
        selected_gpu="$gpu_index"
        selected_gpu_free_mib="$gpu_free_mib"
    fi
done <<<"$gpu_inventory"
[[ -n "$selected_gpu" ]] || fail "No usable NVIDIA GPU was reported" 75
if [[ -n "$training_pids" && "$selected_gpu_free_mib" -lt "$minimum_free_gpu_mib" ]]; then
    fail "Training is active and no GPU has ${minimum_free_gpu_mib} MiB free (best: GPU $selected_gpu, ${selected_gpu_free_mib} MiB)" 75
fi

printf '[EXPORT] step=%s checkpoint_sha256=%s gpu=%s free_mib=%s training_active=%s\n' \
    "$best_step" "$best_sha256" "$selected_gpu" "$selected_gpu_free_mib" \
    "$([[ -n "$training_pids" ]] && printf true || printf false)" >&2

export ACCEPT_EULA=Y
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$selected_gpu"
export HYDRA_FULL_ERROR=1
export MKL_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NUMEXPR_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

export_log="$temporary_bundle/export.log"
export_command=(
    nice -n 19
)
if command -v ionice >/dev/null 2>&1; then
    export_command+=(ionice -c 3)
fi
export_command+=(
    timeout --signal=TERM --kill-after=30s 15m
    bash "$export_script" "$snapshot_checkpoint"
)

if "${export_command[@]}" >"$export_log" 2>&1; then
    :
else
    export_status=$?
    printf '[ERROR] Kengo ONNX export failed or timed out (status %s). Tail follows:\n' \
        "$export_status" >&2
    tail -n 80 "$export_log" >&2 || true
    exit 1
fi

produced_onnx="$temporary_bundle/exported/model_step_${best_step_padded}_kengo.onnx"
[[ -f "$produced_onnx" && ! -L "$produced_onnx" ]] \
    || fail "Exporter did not produce the exact expected ONNX: $produced_onnx"
if ! validation="$(validate_onnx "$produced_onnx")"; then
    fail "New Kengo ONNX failed checker/shape/finite-inference validation"
fi
read -r produced_sha produced_bytes extra <<<"$validation"
[[ -n "$produced_sha" && -n "$produced_bytes" && -z "${extra:-}" ]] \
    || fail "Could not parse new ONNX validation output"

[[ "$(best_binding)" == "$initial_binding" ]] \
    || fail "Best checkpoint changed during ONNX export; refusing stale publication" 75

if [[ -n "$training_pids" ]]; then
    current_training_pids=""
    if current_training_pids="$(pgrep -f '[t]rain_agent_trl.py.*sonic_kengo')"; then
        current_training_pids="$(printf '%s\n' "$current_training_pids" | sort -n -u)"
    else
        pgrep_status=$?
        [[ "$pgrep_status" -eq 1 ]] \
            || fail "Could not re-check Kengo training processes" 75
    fi
    [[ "$current_training_pids" == "$training_pids" ]] \
        || fail "Training PID set changed during export; refusing publication" 75
    kill -0 "$training_launcher_pid" 2>/dev/null \
        || fail "Training launcher exited during export; refusing publication" 75
    current_training_snapshot=""
    while IFS= read -r training_pid; do
        [[ -r "/proc/$training_pid/stat" ]] \
            || fail "Training PID vanished during export: $training_pid" 75
        proc_stat="$(<"/proc/$training_pid/stat")"
        proc_fields="${proc_stat##*) }"
        read -r -a proc_field_array <<<"$proc_fields"
        [[ "${#proc_field_array[@]}" -ge 20 ]] \
            || fail "Training PID snapshot became incomplete: $training_pid" 75
        current_training_snapshot+="$training_pid:${proc_field_array[19]};"
    done <<<"$current_training_pids"
    [[ "$current_training_snapshot" == "$training_snapshot" ]] \
        || fail "Training PID identity changed during export; refusing publication" 75
fi

if [[ -e "$canonical_onnx" || -e "$canonical_manifest" ]]; then
    fail "Canonical step artifact appeared during export; refusing to overwrite it"
fi
publish_temporary="$canonical_dir/.model_step_${best_step_padded}_kengo.${best_sha256:0:12}.$$.tmp.onnx"
mv -- "$produced_onnx" "$publish_temporary"
if [[ -e "$canonical_onnx" ]]; then
    fail "Canonical ONNX conflict detected immediately before publication"
fi
mv -- "$publish_temporary" "$canonical_onnx"
publish_temporary=""

if ! validation="$(validate_onnx "$canonical_onnx")"; then
    fail "Published canonical ONNX failed post-publication validation"
fi
read -r canonical_sha canonical_bytes extra <<<"$validation"
[[ "$canonical_sha" == "$produced_sha" && "$canonical_bytes" == "$produced_bytes" ]] \
    || fail "Published canonical ONNX digest/size differs from validated export"
write_manifest \
    "$canonical_manifest" "$canonical_onnx" "$canonical_sha" "$canonical_bytes" \
    "exported_from_pinned_checkpoint_inode"

emit_result true "$selected_gpu" "$canonical_onnx" "$canonical_sha" "$canonical_bytes"
