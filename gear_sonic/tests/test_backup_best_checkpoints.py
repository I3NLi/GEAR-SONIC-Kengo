from __future__ import annotations

import json
from pathlib import Path

from gear_sonic.scripts.backup_best_checkpoints import (
    BestCheckpointBackup,
    scan_log,
    seconds_until_next_check,
    sha256_file,
)


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _save_block(step: int, reward: float, checkpoint: str) -> str:
    return (
        f"Learning iteration {step}\n"
        f"Mean rewards: {reward:.5f}\n"
        f"Saved model checkpoint to {checkpoint}\n"
    )


def _replace_checkpoint(path: Path, step: int, label: str) -> None:
    replacement = path.with_suffix(".next")
    replacement.write_bytes(f"{step}:{label}".encode())
    replacement.replace(path)


def test_scan_log_is_incremental_and_emits_only_completed_saves(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        _save_block(50, 0.5, "logs/run/last.pt")
        + "Learning iteration 51\nMean rewards: 0.7\n",
        encoding="utf-8",
    )

    first = scan_log(log)
    assert [(event.step, event.mean_reward) for event in first.events] == [(50, 0.5)]
    assert first.current_step == 51
    assert first.current_reward == 0.7

    _append(log, _save_block(100, 1.25, "logs/run/last.pt"))
    second = scan_log(
        log,
        offset=first.offset,
        current_step=first.current_step,
        current_reward=first.current_reward,
    )
    assert [(event.step, event.mean_reward) for event in second.events] == [(100, 1.25)]


def test_scan_log_does_not_commit_an_incomplete_final_line(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    prefix = "Learning iteration 50\nMean rewards: 0.5\n"
    partial_marker = "Saved model checkpoint to logs/run/last.pt"
    log.write_bytes((prefix + partial_marker).encode())

    first = scan_log(log)
    assert not first.events
    assert first.offset == len(prefix.encode())

    _append(log, "\n")
    second = scan_log(
        log,
        offset=first.offset,
        current_step=first.current_step,
        current_reward=first.current_reward,
    )
    assert [(event.step, event.mean_reward) for event in second.events] == [(50, 0.5)]


def test_hourly_sampling_retains_only_one_best_checkpoint(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    log_dir = repo / "run_logs"
    checkpoint_dir = repo / "logs_rl" / "run"
    log_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    log = log_dir / "train.log"
    checkpoint = checkpoint_dir / "last.pt"
    log_dir.joinpath("kengo_full8gpu.latest_log").write_text(str(log), encoding="utf-8")
    _replace_checkpoint(checkpoint, 50, "checkpoint-step-50")
    log.write_text(_save_block(50, 0.5, "logs_rl/run/last.pt"), encoding="utf-8")

    service = BestCheckpointBackup(
        repo_root=repo,
        latest_log_file=Path("run_logs/kengo_full8gpu.latest_log"),
        backup_root=Path("checkpoint_backups/hourly_best"),
        check_interval_seconds=3600,
        checkpoint_step_reader=lambda path: int(path.read_bytes().split(b":", 1)[0]),
    )
    service.run_cycle(now=1_000.0)

    cached = repo / "checkpoint_backups/hourly_best/best_so_far.pt"
    assert cached.read_bytes() == b"50:checkpoint-step-50"
    assert list(service.backup_root.rglob("*.pt")) == [cached]

    _replace_checkpoint(checkpoint, 100, "checkpoint-step-100-worse")
    _append(log, _save_block(100, 0.4, "logs_rl/run/last.pt"))
    service.run_cycle(now=2_000.0)
    assert cached.read_bytes() == b"50:checkpoint-step-50"

    _replace_checkpoint(checkpoint, 150, "checkpoint-step-150-best")
    _append(log, _save_block(150, 0.8, "logs_rl/run/last.pt"))
    service.run_cycle(now=3_000.0)
    assert cached.read_bytes() == b"150:checkpoint-step-150-best"
    assert list(service.backup_root.rglob("*.pt")) == [cached]

    service.run_cycle(now=4_600.0)
    assert list(service.backup_root.rglob("*.pt")) == [cached]
    metadata = json.loads(service.best_metadata.read_text(encoding="utf-8"))
    assert metadata["step"] == 150
    assert metadata["mean_reward"] == 0.8
    assert metadata["sha256"] == sha256_file(cached)
    history = [
        json.loads(line)
        for line in service.history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(item["step"], item["mean_reward"]) for item in history] == [
        (50, 0.5),
        (150, 0.8),
    ]
    state = json.loads(service.state_path.read_text(encoding="utf-8"))
    assert state["checks_completed"] == 4
    assert state["best"]["step"] == 150


def test_checkpoint_step_must_match_a_completed_log_event(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    log_dir = repo / "run_logs"
    checkpoint_dir = repo / "logs_rl" / "run"
    log_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    log = log_dir / "train.log"
    checkpoint = checkpoint_dir / "last.pt"
    log_dir.joinpath("kengo_full8gpu.latest_log").write_text(str(log), encoding="utf-8")
    _replace_checkpoint(checkpoint, 100, "checkpoint-step-100")
    log.write_text(_save_block(50, 0.5, "logs_rl/run/last.pt"), encoding="utf-8")

    service = BestCheckpointBackup(
        repo_root=repo,
        latest_log_file=Path("run_logs/kengo_full8gpu.latest_log"),
        backup_root=Path("checkpoint_backups/hourly_best"),
        check_interval_seconds=3600,
        checkpoint_step_reader=lambda path: int(path.read_bytes().split(b":", 1)[0]),
    )
    service.run_cycle(now=1_000.0)
    assert not service.best_checkpoint.exists()
    assert not service.history_path.exists()

    _append(log, _save_block(100, 0.75, "logs_rl/run/last.pt"))
    service.run_cycle(now=4_600.0)
    assert service.best_checkpoint.read_bytes() == b"100:checkpoint-step-100"
    metadata = json.loads(service.best_metadata.read_text(encoding="utf-8"))
    assert metadata["step"] == 100
    assert metadata["mean_reward"] == 0.75


def test_startup_reconciles_manifest_and_cleans_orphan_candidate(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    log_dir = repo / "run_logs"
    checkpoint_dir = repo / "logs_rl" / "run"
    log_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    log = log_dir / "train.log"
    checkpoint = checkpoint_dir / "last.pt"
    log_dir.joinpath("kengo_full8gpu.latest_log").write_text(str(log), encoding="utf-8")
    _replace_checkpoint(checkpoint, 50, "checkpoint-step-50")
    log.write_text(_save_block(50, 0.5, "logs_rl/run/last.pt"), encoding="utf-8")

    service = BestCheckpointBackup(
        repo_root=repo,
        latest_log_file=Path("run_logs/kengo_full8gpu.latest_log"),
        backup_root=Path("checkpoint_backups/hourly_best"),
        checkpoint_step_reader=lambda path: int(path.read_bytes().split(b":", 1)[0]),
    )
    service.run_cycle(now=1_000.0)
    state = json.loads(service.state_path.read_text(encoding="utf-8"))
    state["best"] = None
    service.state_path.write_text(json.dumps(state), encoding="utf-8")
    orphan = service.backup_root / ".candidate.999.1.pt"
    orphan.write_bytes(b"orphaned checkpoint blocks")

    recovered = BestCheckpointBackup(
        repo_root=repo,
        latest_log_file=Path("run_logs/kengo_full8gpu.latest_log"),
        backup_root=Path("checkpoint_backups/hourly_best"),
        checkpoint_step_reader=lambda path: int(path.read_bytes().split(b":", 1)[0]),
    )
    assert recovered.state["best"]["step"] == 50
    assert not orphan.exists()
    assert list(recovered.backup_root.rglob("*.pt")) == [recovered.best_checkpoint]


def test_startup_finishes_a_pending_best_update(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    log_dir = repo / "run_logs"
    log_dir.mkdir(parents=True)
    log = log_dir / "train.log"
    log.write_text("", encoding="utf-8")
    log_dir.joinpath("kengo_full8gpu.latest_log").write_text(str(log), encoding="utf-8")
    service = BestCheckpointBackup(
        repo_root=repo,
        latest_log_file=Path("run_logs/kengo_full8gpu.latest_log"),
        backup_root=Path("checkpoint_backups/hourly_best"),
        checkpoint_step_reader=lambda path: int(path.read_bytes().split(b":", 1)[0]),
    )
    service.best_checkpoint.write_bytes(b"50:previous-best")
    previous = {
        "bytes": service.best_checkpoint.stat().st_size,
        "mean_reward": 0.5,
        "sha256": sha256_file(service.best_checkpoint),
        "step": 50,
    }
    service.state["best"] = previous
    service.state_path.write_text(json.dumps(service.state), encoding="utf-8")
    service.best_metadata.write_text(json.dumps(previous), encoding="utf-8")

    candidate = service.backup_root / ".candidate.123.456.pt"
    candidate.write_bytes(b"100:new-best")
    pending = {
        "bytes": candidate.stat().st_size,
        "mean_reward": 0.9,
        "sha256": sha256_file(candidate),
        "step": 100,
    }
    service.pending_update.write_text(
        json.dumps({"candidate_name": candidate.name, "metadata": pending}),
        encoding="utf-8",
    )

    recovered = BestCheckpointBackup(
        repo_root=repo,
        latest_log_file=Path("run_logs/kengo_full8gpu.latest_log"),
        backup_root=Path("checkpoint_backups/hourly_best"),
        checkpoint_step_reader=lambda path: int(path.read_bytes().split(b":", 1)[0]),
    )
    assert recovered.best_checkpoint.read_bytes() == b"100:new-best"
    assert recovered.state["best"]["step"] == 100
    assert not recovered.pending_update.exists()
    assert list(recovered.backup_root.rglob("*.pt")) == [recovered.best_checkpoint]


def test_restart_preserves_the_next_hourly_check_time() -> None:
    state = {"next_check_at": "1970-01-01T01:00:00+00:00"}
    assert seconds_until_next_check(state, now=1_000.0) == 2_600.0
    assert seconds_until_next_check(state, now=4_000.0) == 0.0
