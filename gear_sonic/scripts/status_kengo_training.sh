#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pid_file="$repo_root/run_logs/kengo_full8gpu.pid"
latest_log_file="$repo_root/run_logs/kengo_full8gpu.latest_log"
tail_lines="${1:-80}"

if [[ ! "$tail_lines" =~ ^[1-9][0-9]*$ ]]; then
    printf 'tail_lines must be a positive integer: %s\n' "$tail_lines" >&2
    exit 2
fi

if [[ -r "$pid_file" ]]; then
    launcher_pid="$(<"$pid_file")"
    if [[ "$launcher_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$launcher_pid" 2>/dev/null; then
        printf 'Launcher PID %s is running.\n' "$launcher_pid"
        ps -p "$launcher_pid" -o pid=,ppid=,etime=,%cpu=,%mem=,args=
    else
        printf 'Recorded launcher PID %s is not running.\n' "$launcher_pid"
    fi
else
    printf 'No launcher PID file found.\n'
fi

printf '\nTraining processes:\n'
pgrep -af '[t]rain_agent_trl.py.*sonic_kengo' || true

printf '\nGPU state:\n'
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader

if [[ -r "$latest_log_file" ]]; then
    latest_log="$(<"$latest_log_file")"
    printf '\nLatest log (%s):\n' "$latest_log"
    tail -n "$tail_lines" "$latest_log"
fi

backup_pid_file="$repo_root/run_logs/kengo_checkpoint_backup.pid"
backup_state="$repo_root/checkpoint_backups/hourly_best/state.json"
printf '\nHourly best-checkpoint backup:\n'
if [[ -r "$backup_pid_file" ]]; then
    backup_pid="$(<"$backup_pid_file")"
    if [[ "$backup_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$backup_pid" 2>/dev/null; then
        ps -p "$backup_pid" -o pid=,ppid=,etime=,%cpu=,%mem=,args=
    else
        printf 'Recorded backup PID %s is not running.\n' "$backup_pid"
    fi
else
    printf 'No hourly backup service PID file found.\n'
fi
if [[ -r "$backup_state" ]]; then
    "$repo_root/.venv/bin/python" - "$backup_state" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    state = json.load(stream)
summary = {
    "checks_completed": state.get("checks_completed"),
    "last_check_at": state.get("last_check_at"),
    "next_check_at": state.get("next_check_at"),
    "best": state.get("best"),
}
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY
fi
if [[ -f "$repo_root/checkpoint_backups/hourly_best/best_so_far.pt" ]]; then
    printf '\nRetained checkpoint (only one full backup is kept):\n'
    du -h "$repo_root/checkpoint_backups/hourly_best/best_so_far.pt"
fi
