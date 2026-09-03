#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

num_envs="${1:-4096}"
iterations="${2:-100000}"
run_tag="${3:-full8gpu_sonic_filtered}"

if [[ ! "$num_envs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'num_envs must be a positive integer: %s\n' "$num_envs" >&2
    exit 2
fi
if [[ ! "$iterations" =~ ^[1-9][0-9]*$ ]]; then
    printf 'iterations must be a positive integer: %s\n' "$iterations" >&2
    exit 2
fi
if [[ ! "$run_tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    printf 'run_tag may contain only letters, digits, dot, underscore, and dash: %s\n' "$run_tag" >&2
    exit 2
fi

python_bin="$repo_root/.venv/bin/python"
motion_file="$repo_root/data/kengo_motion_lib/robot_filtered/kengo_sonic_filtered.pkl"
manifest_file="$motion_file.manifest.json"
if [[ ! -x "$python_bin" ]]; then
    printf 'Missing training interpreter: %s\n' "$python_bin" >&2
    exit 1
fi
if ! "$python_bin" gear_sonic/scripts/stage_kengo_assets.py; then
    printf 'Pinned Kengo asset submodule validation failed.\n' >&2
    exit 1
fi
if [[ ! -s "$motion_file" || ! -s "$manifest_file" ]]; then
    printf 'Missing filtered Kengo motion library or manifest.\n' >&2
    exit 1
fi

log_dir="$repo_root/run_logs"
pid_file="$log_dir/kengo_full8gpu.pid"
mkdir -p "$log_dir"

if [[ -r "$pid_file" ]]; then
    existing_pid="$(<"$pid_file")"
    if [[ "$existing_pid" =~ ^[1-9][0-9]*$ ]] \
        && kill -0 "$existing_pid" 2>/dev/null \
        && ps -p "$existing_pid" -o args= | grep -q 'accelerate.commands.launch'; then
        printf 'Kengo training is already running as PID %s.\n' "$existing_pid" >&2
        exit 1
    fi
fi
if pgrep -af '[t]rain_agent_trl.py.*sonic_kengo' >/dev/null; then
    printf 'Another Kengo SONIC training process is already running:\n' >&2
    pgrep -af '[t]rain_agent_trl.py.*sonic_kengo' >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%d_%H%M%S)"
stdout_log="$log_dir/kengo_${run_tag}_${timestamp}.log"

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=12

command=(
    "$python_bin" -m accelerate.commands.launch
    --multi_gpu
    --num_processes=8
    --num_machines=1
    --gpu_ids=0,1,2,3,4,5,6,7
    --main_process_port=0
    --mixed_precision=no
    --dynamo_backend=no
    --num_cpu_threads_per_process=12
    gear_sonic/train_agent_trl.py
    +exp=manager/universal_token/all_modes/sonic_kengo
    "exp_var=$run_tag"
    "num_envs=$num_envs"
    headless=true
    use_wandb=false
    "++algo.config.num_learning_iterations=$iterations"
)

nohup setsid "${command[@]}" >"$stdout_log" 2>&1 </dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"$pid_file"
printf '%s\n' "$stdout_log" >"$log_dir/kengo_full8gpu.latest_log"

sleep 2
if ! kill -0 "$launcher_pid" 2>/dev/null; then
    printf 'Training launcher exited during startup; inspect %s\n' "$stdout_log" >&2
    exit 1
fi

printf 'Started Kengo SONIC training.\n'
printf 'PID: %s\n' "$launcher_pid"
printf 'Log: %s\n' "$stdout_log"
printf 'Environments: %s per GPU, %s total\n' "$num_envs" "$((num_envs * 8))"
printf 'Iterations: %s\n' "$iterations"
