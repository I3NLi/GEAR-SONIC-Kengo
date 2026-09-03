#!/usr/bin/env python3
"""Check training once per hour and retain only the best sampled checkpoint.

The training callback atomically replaces ``last.pt`` every 50 iterations.  At
each hourly check this service hard-links the current inode, validates its real
``global_step``, associates that step with the log's ``Mean rewards`` value,
and atomically replaces ``best_so_far.pt`` only when the sampled score improves.
Only one full backup is retained; metadata and a tiny JSONL improvement history
provide an audit trail without accumulating hourly checkpoint copies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import tempfile
import threading
import time
from typing import Any, Callable

STATE_SCHEMA_VERSION = 2
RECENT_EVENT_LIMIT = 128
ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)")
MEAN_REWARD_RE = re.compile(
    r"Mean rewards:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
LAST_CHECKPOINT_RE = re.compile(r"Saved model checkpoint to\s+(.+?/last\.pt)\s*$")


@dataclass(frozen=True)
class CheckpointEvent:
    step: int
    mean_reward: float
    checkpoint_path: str


@dataclass(frozen=True)
class ScanResult:
    offset: int
    current_step: int | None
    current_reward: float | None
    events: tuple[CheckpointEvent, ...]


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint_step(path: Path) -> int:
    """Load a locked checkpoint inode and validate its recoverable core state."""

    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "policy_state_dict",
        "value_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "state",
        "args",
        "env_state_dict",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    step = int(checkpoint["state"].global_step)
    if step < 1:
        raise ValueError(f"Checkpoint global_step must be positive: {step}")
    for name in ("policy_state_dict", "value_state_dict"):
        state_dict = checkpoint[name]
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"Checkpoint {name} must be a non-empty dictionary")
        for key, value in state_dict.items():
            if torch.is_tensor(value) and (
                value.is_floating_point() or value.is_complex()
            ):
                if not torch.isfinite(value).all().item():
                    raise ValueError(
                        f"Checkpoint contains non-finite tensor {name}.{key}"
                    )
    return step


def scan_log(
    log_path: Path,
    *,
    offset: int = 0,
    current_step: int | None = None,
    current_reward: float | None = None,
) -> ScanResult:
    """Scan newly appended log bytes and return completed ``last.pt`` events."""

    size = log_path.stat().st_size
    if offset < 0 or offset > size:
        offset = 0
        current_step = None
        current_reward = None

    events: list[CheckpointEvent] = []
    with log_path.open("rb") as stream:
        stream.seek(offset)
        while True:
            line_start = stream.tell()
            raw_line = stream.readline()
            if not raw_line:
                break
            # A writer can be between two writes when the hourly read begins.
            # Do not commit a partial final line; reread it next hour.
            if not raw_line.endswith(b"\n"):
                stream.seek(line_start)
                break
            line = raw_line.decode("utf-8", errors="replace")
            if match := ITERATION_RE.search(line):
                current_step = int(match.group(1))
                current_reward = None
            if match := MEAN_REWARD_RE.search(line):
                reward = float(match.group(1))
                if math.isfinite(reward):
                    current_reward = reward
            if match := LAST_CHECKPOINT_RE.search(line):
                if current_step is not None and current_reward is not None:
                    events.append(
                        CheckpointEvent(
                            step=current_step,
                            mean_reward=current_reward,
                            checkpoint_path=match.group(1).strip(),
                        )
                    )
        new_offset = stream.tell()

    return ScanResult(
        offset=new_offset,
        current_step=current_step,
        current_reward=current_reward,
        events=tuple(events),
    )


def link_checkpoint_inode(source: Path, destination_dir: Path) -> Path:
    """Hard-link the current atomic ``last.pt`` inode without copying 413 MiB."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    candidate = destination_dir / (f".candidate.{os.getpid()}.{time.time_ns()}.pt")
    os.link(source, candidate)
    if candidate.stat().st_size < 1:
        candidate.unlink(missing_ok=True)
        raise ValueError(f"Checkpoint candidate is empty: {source}")
    return candidate


def atomic_json(data: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
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


def append_json_line(data: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class BestCheckpointBackup:
    def __init__(
        self,
        *,
        repo_root: Path,
        latest_log_file: Path,
        backup_root: Path,
        check_interval_seconds: int = 3600,
        checkpoint_step_reader: Callable[[Path], int] = read_checkpoint_step,
    ) -> None:
        self.repo_root = repo_root.expanduser().resolve(strict=True)
        self.latest_log_file = self._resolve_inside_repo(latest_log_file)
        self.backup_root = self._resolve_inside_repo(backup_root, must_exist=False)
        self.check_interval_seconds = check_interval_seconds
        self.checkpoint_step_reader = checkpoint_step_reader
        if check_interval_seconds < 1:
            raise ValueError("check_interval_seconds must be positive")

        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint = self.backup_root / "best_so_far.pt"
        self.best_metadata = self.backup_root / "best_so_far.json"
        self.pending_update = self.backup_root / ".best_update.pending.json"
        self.history_path = self.backup_root / "improvement_history.jsonl"
        self.state_path = self.backup_root / "state.json"
        self.state = self._load_state()
        self._reconcile_storage()

    def _resolve_inside_repo(self, path: Path, *, must_exist: bool = True) -> Path:
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        resolved = candidate.resolve(strict=must_exist)
        if not resolved.is_relative_to(self.repo_root):
            raise ValueError(f"Path must stay inside repository: {resolved}")
        return resolved

    def _default_state(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "log_path": None,
            "log_device": None,
            "log_inode": None,
            "log_offset": 0,
            "current_step": None,
            "current_reward": None,
            "current_checkpoint_path": None,
            "recent_events": [],
            "best": None,
            "last_check_at": None,
            "next_check_at": None,
            "checks_completed": 0,
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported backup state schema: {state.get('schema_version')}"
            )
        return state

    def _read_current_log(self) -> Path:
        raw_path = self.latest_log_file.read_text(encoding="utf-8").strip()
        if not raw_path:
            raise ValueError(f"Latest-log pointer is empty: {self.latest_log_file}")
        return self._resolve_inside_repo(Path(raw_path))

    def _resolve_checkpoint(self, raw_path: str) -> Path:
        return self._resolve_inside_repo(Path(raw_path))

    @staticmethod
    def _metadata_matches_checkpoint(metadata: dict[str, Any], path: Path) -> bool:
        try:
            return (
                path.is_file()
                and path.stat().st_size == int(metadata["bytes"])
                and sha256_file(path) == str(metadata["sha256"])
            )
        except (KeyError, OSError, TypeError, ValueError):
            return False

    def _append_history_once(self, metadata: dict[str, Any]) -> None:
        if self.history_path.is_file():
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
            if lines:
                try:
                    if json.loads(lines[-1]).get("sha256") == metadata["sha256"]:
                        return
                except (json.JSONDecodeError, KeyError):
                    pass
        append_json_line(metadata, self.history_path)

    def _reconcile_storage(self) -> None:
        """Recover an interrupted best update and remove orphan hard links."""

        committed: dict[str, Any] | None = None
        if self.pending_update.is_file():
            try:
                transaction = json.loads(
                    self.pending_update.read_text(encoding="utf-8")
                )
                metadata = transaction["metadata"]
                candidate_name = str(transaction["candidate_name"])
                if (
                    not candidate_name.startswith(".candidate.")
                    or "/" in candidate_name
                ):
                    raise ValueError("Invalid pending candidate name")
                candidate = self.backup_root / candidate_name
                if self._metadata_matches_checkpoint(metadata, self.best_checkpoint):
                    committed = metadata
                elif self._metadata_matches_checkpoint(metadata, candidate):
                    os.replace(candidate, self.best_checkpoint)
                    committed = metadata
                if committed is not None:
                    atomic_json(committed, self.best_metadata)
                    print(
                        f"Recovered interrupted best update at step {committed['step']}.",
                        flush=True,
                    )
            except (
                KeyError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                print(f"Discarding invalid pending update: {exc}", flush=True)
            finally:
                self.pending_update.unlink(missing_ok=True)

        for orphan in self.backup_root.glob(".candidate.*.pt"):
            orphan.unlink(missing_ok=True)

        if committed is None and self.best_metadata.is_file():
            try:
                metadata = json.loads(self.best_metadata.read_text(encoding="utf-8"))
                if self._metadata_matches_checkpoint(metadata, self.best_checkpoint):
                    committed = metadata
            except (OSError, json.JSONDecodeError):
                committed = None

        if committed is None:
            state_best = self.state.get("best")
            if isinstance(state_best, dict) and self._metadata_matches_checkpoint(
                state_best, self.best_checkpoint
            ):
                committed = state_best
                atomic_json(committed, self.best_metadata)

        self.state["best"] = committed
        if committed is not None:
            self._append_history_once(committed)
        atomic_json(self.state, self.state_path)

    def _update_best(
        self,
        event: CheckpointEvent,
        source: Path,
        candidate: Path,
        log_path: Path,
    ) -> bool:
        best = self.state.get("best")
        if (
            best is not None
            and self.best_checkpoint.is_file()
            and event.mean_reward <= float(best["mean_reward"])
        ):
            return False

        digest = sha256_file(candidate)
        metadata = {
            "metric": "training_mean_reward",
            "mean_reward": event.mean_reward,
            "step": event.step,
            "source_checkpoint": str(source),
            "source_log": str(log_path),
            "cached_checkpoint": str(self.best_checkpoint),
            "bytes": candidate.stat().st_size,
            "sha256": digest,
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(
            {"candidate_name": candidate.name, "metadata": metadata},
            self.pending_update,
        )
        os.replace(candidate, self.best_checkpoint)
        atomic_json(metadata, self.best_metadata)
        self.pending_update.unlink(missing_ok=True)
        self._append_history_once(metadata)
        self.state["best"] = metadata
        print(
            f"New best checkpoint: step={event.step} "
            f"mean_reward={event.mean_reward:.8g} sha256={digest}",
            flush=True,
        )
        return True

    def run_cycle(self, *, now: float | None = None) -> None:
        cycle_time = time.time() if now is None else now
        log_path = self._read_current_log()
        log_stat = log_path.stat()
        identity_changed = (
            self.state.get("log_path") != str(log_path)
            or self.state.get("log_device") != log_stat.st_dev
            or self.state.get("log_inode") != log_stat.st_ino
            or int(self.state.get("log_offset", 0)) > log_stat.st_size
        )
        if identity_changed:
            self.state.update(
                {
                    "log_path": str(log_path),
                    "log_device": log_stat.st_dev,
                    "log_inode": log_stat.st_ino,
                    "log_offset": 0,
                    "current_step": None,
                    "current_reward": None,
                    "current_checkpoint_path": None,
                    "recent_events": [],
                }
            )

        result = scan_log(
            log_path,
            offset=int(self.state["log_offset"]),
            current_step=self.state.get("current_step"),
            current_reward=self.state.get("current_reward"),
        )
        recent_by_step = {
            int(event["step"]): event for event in self.state.get("recent_events", [])
        }

        def merge_scan(scan: ScanResult) -> None:
            self.state.update(
                {
                    "log_offset": scan.offset,
                    "current_step": scan.current_step,
                    "current_reward": scan.current_reward,
                }
            )
            for event in scan.events:
                recent_by_step[event.step] = {
                    "step": event.step,
                    "mean_reward": event.mean_reward,
                    "checkpoint_path": event.checkpoint_path,
                }
                self.state["current_checkpoint_path"] = event.checkpoint_path
            self.state["recent_events"] = sorted(
                recent_by_step.values(), key=lambda event: event["step"]
            )[-RECENT_EVENT_LIMIT:]

        merge_scan(result)

        # Check exactly one current checkpoint per cycle.  The hard link pins the
        # inode even if training atomically replaces ``last.pt`` during loading.
        current_checkpoint_path = self.state.get("current_checkpoint_path")
        if current_checkpoint_path is not None:
            checkpoint = self._resolve_checkpoint(current_checkpoint_path)
            candidate = link_checkpoint_inode(checkpoint, self.backup_root)
            try:
                checkpoint_step = self.checkpoint_step_reader(candidate)
                event_data = recent_by_step.get(checkpoint_step)
                if event_data is None:
                    # Loading a real checkpoint takes several seconds. Rescan
                    # once in case its save marker landed just after the first
                    # scan; this is still one hourly check, not polling.
                    retry = scan_log(
                        log_path,
                        offset=int(self.state["log_offset"]),
                        current_step=self.state.get("current_step"),
                        current_reward=self.state.get("current_reward"),
                    )
                    merge_scan(retry)
                    event_data = recent_by_step.get(checkpoint_step)
                if event_data is None:
                    print(
                        f"Checkpoint step {checkpoint_step} has no completed log event yet; "
                        "leaving the retained best unchanged.",
                        flush=True,
                    )
                else:
                    event = CheckpointEvent(
                        step=int(event_data["step"]),
                        mean_reward=float(event_data["mean_reward"]),
                        checkpoint_path=str(event_data["checkpoint_path"]),
                    )
                    self._update_best(
                        event,
                        checkpoint,
                        candidate,
                        log_path,
                    )
            finally:
                candidate.unlink(missing_ok=True)

        self.state["last_check_at"] = datetime.fromtimestamp(
            cycle_time, timezone.utc
        ).isoformat()
        self.state["next_check_at"] = datetime.fromtimestamp(
            cycle_time + self.check_interval_seconds, timezone.utc
        ).isoformat()
        self.state["checks_completed"] = int(self.state["checks_completed"]) + 1
        atomic_json(self.state, self.state_path)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def read_live_pid(pid_file: Path) -> tuple[int | None, bool]:
    if not pid_file.is_file():
        return None, False
    text = pid_file.read_text(encoding="utf-8").strip()
    if not text.isdecimal() or int(text) < 1:
        raise ValueError(f"Invalid training PID file: {pid_file}")
    pid = int(text)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return pid, False
    except PermissionError:
        return pid, True
    return pid, True


def acquire_instance_lock(lock_path: Path):
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"Another checkpoint backup service holds {lock_path}")
    stream.seek(0)
    stream.truncate()
    stream.write(f"{os.getpid()}\n")
    stream.flush()
    return stream


def seconds_until_next_check(
    state: dict[str, Any], *, now: float | None = None
) -> float:
    value = state.get("next_check_at")
    if not value:
        return 0.0
    try:
        due = datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0
    return max(0.0, due - (time.time() if now is None else now))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--latest-log-file",
        type=Path,
        default=Path("run_logs/kengo_full8gpu.latest_log"),
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("checkpoint_backups/hourly_best"),
    )
    parser.add_argument("--check-interval-seconds", type=positive_int, default=3600)
    parser.add_argument(
        "--training-pid-file",
        type=Path,
        default=Path("run_logs/kengo_full8gpu.pid"),
        help="Stop the hourly service after the recorded training process exits",
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve(strict=True)
    backup_root = args.backup_root.expanduser()
    if not backup_root.is_absolute():
        backup_root = repo_root / backup_root
    backup_root = backup_root.resolve(strict=False)
    if not backup_root.is_relative_to(repo_root):
        raise ValueError(f"Path must stay inside repository: {backup_root}")
    instance_lock = acquire_instance_lock(backup_root / ".service.lock")
    service = BestCheckpointBackup(
        repo_root=args.repo_root,
        latest_log_file=args.latest_log_file,
        backup_root=args.backup_root,
        check_interval_seconds=args.check_interval_seconds,
    )
    training_pid_file = service._resolve_inside_repo(
        args.training_pid_file, must_exist=False
    )
    if args.once:
        service.run_cycle()
        instance_lock.close()
        return 0

    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(
        f"Checking {service.latest_log_file} every "
        f"{args.check_interval_seconds} seconds; retaining only "
        f"{service.best_checkpoint}",
        flush=True,
    )

    initial_delay = seconds_until_next_check(service.state)
    if initial_delay > 0:
        print(
            f"Preserving the existing schedule; next check in {initial_delay:.0f} seconds.",
            flush=True,
        )
        if stop_event.wait(initial_delay):
            instance_lock.close()
            print("Hourly checkpoint backup service stopped.", flush=True)
            return 0

    while not stop_event.is_set():
        cycle_started = time.time()
        try:
            training_pid, training_was_alive = read_live_pid(training_pid_file)
            service.run_cycle(now=cycle_started)
            training_pid, training_alive = read_live_pid(training_pid_file)
            if training_pid is not None and not training_alive:
                if training_was_alive and not stop_event.wait(2.0):
                    print(
                        "Training exited during the hourly check; performing one final check.",
                        flush=True,
                    )
                    service.run_cycle()
                print(
                    f"Training PID {training_pid} has exited; stopping backup service.",
                    flush=True,
                )
                break
        except Exception as exc:  # keep the hourly service alive across transient races
            print(f"Backup cycle failed: {type(exc).__name__}: {exc}", flush=True)
        wait_seconds = max(
            1.0,
            cycle_started + args.check_interval_seconds - time.time(),
        )
        stop_event.wait(wait_seconds)
    instance_lock.close()
    print("Hourly checkpoint backup service stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
