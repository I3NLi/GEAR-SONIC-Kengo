#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="$repo_root/.venv/bin/python"
latest_log_file="$repo_root/run_logs/kengo_full8gpu.latest_log"
backup_root="$repo_root/checkpoint_backups/hourly_best"
pid_file="$repo_root/run_logs/kengo_checkpoint_backup.pid"
mkdir -p "$repo_root/run_logs" "$backup_root"

if [[ ! -x "$python_bin" ]]; then
    printf 'Missing Python interpreter: %s\n' "$python_bin" >&2
    exit 1
fi
if [[ ! -s "$latest_log_file" ]]; then
    printf 'Missing current-training log pointer: %s\n' "$latest_log_file" >&2
    exit 1
fi
if [[ -r "$pid_file" ]]; then
    existing_pid="$(<"$pid_file")"
    if [[ "$existing_pid" =~ ^[1-9][0-9]*$ ]] \
        && kill -0 "$existing_pid" 2>/dev/null \
        && ps -p "$existing_pid" -o args= | grep -q 'backup_best_checkpoints.py'; then
        printf 'Hourly checkpoint backup service is already running as PID %s.\n' "$existing_pid" >&2
        exit 1
    fi
fi
if pgrep -af '[b]ackup_best_checkpoints.py' >/dev/null; then
    printf 'Another checkpoint backup service is already running:\n' >&2
    pgrep -af '[b]ackup_best_checkpoints.py' >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%d_%H%M%S)"
stdout_log="$repo_root/run_logs/kengo_checkpoint_backup_${timestamp}.log"
command=(
    "$python_bin"
    gear_sonic/scripts/backup_best_checkpoints.py
    --repo-root "$repo_root"
    --latest-log-file "$latest_log_file"
    --backup-root "$backup_root"
    --check-interval-seconds 3600
    --training-pid-file "$repo_root/run_logs/kengo_full8gpu.pid"
)

nohup setsid "${command[@]}" >"$stdout_log" 2>&1 </dev/null &
service_pid=$!
printf '%s\n' "$service_pid" >"$pid_file"
printf '%s\n' "$stdout_log" >"$repo_root/run_logs/kengo_checkpoint_backup.latest_log"

sleep 2
if ! kill -0 "$service_pid" 2>/dev/null; then
    printf 'Checkpoint backup service exited during startup; inspect %s\n' "$stdout_log" >&2
    exit 1
fi

printf 'Started hourly best-checkpoint backup service.\n'
printf 'PID: %s\n' "$service_pid"
printf 'Log: %s\n' "$stdout_log"
printf 'Backup root: %s\n' "$backup_root"
printf 'Retention: one best_so_far.pt plus small JSON metadata/history.\n'
