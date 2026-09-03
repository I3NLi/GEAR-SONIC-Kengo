from __future__ import annotations

import sys

import numpy as np
import pytest

from gear_sonic.scripts.run_kengo_sonic_sim2sim import (
    FUTURE_FRAMES,
    FUTURE_JOINT_DIM,
    HISTORY_LENGTH,
    NUM_JOINTS,
    POLICY_INPUT_DIM,
    PROPRIOCEPTION_DIM,
    TOKENIZER_DIM,
    BinaryFloat32PolicyClient,
    MotionReference,
    RemoteSshOnnxSession,
    SonicHistory,
    _full_reference_step_budget,
    _parse_args,
    assemble_sonic_observation,
    load_motion_npz,
    relative_rotation_6d,
    training_forward_difference,
)
from gear_sonic.utils.kengo_contract import (
    KENGO_ISAACLAB_JOINT_NAMES,
    KENGO_MUJOCO_JOINT_NAMES,
)


def test_forward_difference_preserves_training_tail_convention() -> None:
    values = np.asarray([[0.0], [1.0], [3.0], [6.0]], dtype=np.float32)
    velocity = training_forward_difference(values, fps=2.0)
    np.testing.assert_allclose(velocity[:, 0], [2.0, 4.0, 6.0, 4.0])


def test_relative_rotation_6d_uses_first_two_matrix_columns() -> None:
    identity = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    yaw_90 = np.asarray((np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), dtype=np.float32)
    result = relative_rotation_6d(identity, np.stack((identity, yaw_90)))
    np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[1], [0.0, -1.0, 1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_combined_observation_has_exact_1270_layout() -> None:
    future = np.concatenate(
        (
            np.full(FUTURE_JOINT_DIM, 1.0, dtype=np.float32),
            np.full(FUTURE_JOINT_DIM, 2.0, dtype=np.float32),
            np.full(FUTURE_FRAMES * 6, 3.0, dtype=np.float32),
        )
    )
    gyro = np.full((HISTORY_LENGTH, 3), 4.0, dtype=np.float32)
    joint_pos = np.full((HISTORY_LENGTH, NUM_JOINTS), 5.0, dtype=np.float32)
    joint_vel = np.full((HISTORY_LENGTH, NUM_JOINTS), 6.0, dtype=np.float32)
    actions = np.full((HISTORY_LENGTH, NUM_JOINTS), 7.0, dtype=np.float32)
    gravity = np.full((HISTORY_LENGTH, 3), 8.0, dtype=np.float32)
    observation = assemble_sonic_observation(
        future, gyro, joint_pos, joint_vel, actions, gravity
    )
    assert observation.shape == (POLICY_INPUT_DIM,)
    np.testing.assert_array_equal(observation[:FUTURE_JOINT_DIM], 1.0)
    np.testing.assert_array_equal(
        observation[FUTURE_JOINT_DIM : 2 * FUTURE_JOINT_DIM], 2.0
    )
    np.testing.assert_array_equal(
        observation[2 * FUTURE_JOINT_DIM : TOKENIZER_DIM], 3.0
    )
    actor = observation[TOKENIZER_DIM:]
    assert actor.shape == (PROPRIOCEPTION_DIM,)
    boundaries = np.cumsum([0, 30, 230, 230, 230, 30])
    for start, end, expected in zip(boundaries[:-1], boundaries[1:], range(4, 9)):
        np.testing.assert_array_equal(actor[start:end], float(expected))


def test_history_is_oldest_to_newest_and_actions_are_previous_actions() -> None:
    history = SonicHistory()
    future = np.zeros(TOKENIZER_DIM, dtype=np.float32)
    history.build(
        future,
        np.ones(3, dtype=np.float32),
        np.ones(NUM_JOINTS, dtype=np.float32),
        np.ones(NUM_JOINTS, dtype=np.float32),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
    )
    history.record_action(np.full(NUM_JOINTS, 9.0, dtype=np.float32))
    observation = history.build(
        future,
        np.full(3, 2.0, dtype=np.float32),
        np.full(NUM_JOINTS, 2.0, dtype=np.float32),
        np.full(NUM_JOINTS, 2.0, dtype=np.float32),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
    )
    np.testing.assert_array_equal(history.base_angular_velocity[:-1], 1.0)
    np.testing.assert_array_equal(history.base_angular_velocity[-1], 2.0)
    np.testing.assert_array_equal(history.action[:-1], 0.0)
    np.testing.assert_array_equal(history.action[-1], 9.0)
    action_start = TOKENIZER_DIM + 30 + 230 + 230
    np.testing.assert_array_equal(
        observation[action_start : action_start + 9 * NUM_JOINTS], 0.0
    )
    np.testing.assert_array_equal(
        observation[action_start + 9 * NUM_JOINTS : action_start + 10 * NUM_JOINTS],
        9.0,
    )


def test_motion_loader_reorders_mujoco_names_and_builds_future_contract(
    tmp_path,
) -> None:
    frames = 60
    source_fps = 50.0
    raw_joint_pos = np.empty((frames, NUM_JOINTS), dtype=np.float64)
    for column in range(NUM_JOINTS):
        raw_joint_pos[:, column] = column + 0.001 * np.arange(frames)
    root_pos = np.zeros((frames, 3), dtype=np.float64)
    root_pos[:, 2] = 0.9
    root_quat = np.zeros((frames, 4), dtype=np.float64)
    root_quat[:, 0] = 1.0
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        framerate=np.asarray(source_fps),
        joint_names=np.asarray(KENGO_MUJOCO_JOINT_NAMES),
        joint_pos=raw_joint_pos,
        base_pos_w=root_pos,
        base_quat_w=root_quat,
    )
    clip = load_motion_npz(path)
    assert clip.num_frames == frames
    expected_first = np.asarray(
        [KENGO_MUJOCO_JOINT_NAMES.index(name) for name in KENGO_ISAACLAB_JOINT_NAMES]
    )
    np.testing.assert_allclose(clip.joint_pos[0], expected_first, atol=1e-6)
    np.testing.assert_allclose(clip.joint_vel[:-1], 0.05, atol=2e-5)

    reference = MotionReference(clip)
    future = reference.future_observation(np.asarray((1.0, 0.0, 0.0, 0.0)))
    assert future.shape == (TOKENIZER_DIM,)
    expected_indices = np.arange(FUTURE_FRAMES) * 5
    expected_joint_pos = clip.joint_pos[expected_indices].reshape(-1)
    expected_joint_vel = clip.joint_vel[expected_indices].reshape(-1)
    np.testing.assert_allclose(future[:FUTURE_JOINT_DIM], expected_joint_pos)
    np.testing.assert_allclose(
        future[FUTURE_JOINT_DIM : 2 * FUTURE_JOINT_DIM], expected_joint_vel
    )
    expected_identity_6d = np.tile([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], FUTURE_FRAMES)
    np.testing.assert_allclose(future[2 * FUTURE_JOINT_DIM :], expected_identity_6d)


def test_motion_loader_accepts_hiyio_wbt_body_schema(tmp_path) -> None:
    frames = 10
    body_names = np.asarray((b"decoy_link", b"torso_link"))
    body_pos = np.zeros((frames, len(body_names), 3), dtype=np.float32)
    body_pos[:, 0, :] = 42.0
    body_pos[:, 1, 0] = np.linspace(0.0, 0.09, frames)
    body_pos[:, 1, 2] = 0.86
    body_quat = np.zeros((frames, len(body_names), 4), dtype=np.float32)
    body_quat[:, :, 0] = 1.0
    yaw_90 = np.asarray((np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), dtype=np.float32)
    body_quat[:, 1, :] = yaw_90
    body_quat[1::2, 1, :] *= -1.0
    canonical_joint_pos = np.tile(
        np.arange(NUM_JOINTS, dtype=np.float32), (frames, 1)
    )
    canonical_joint_pos += 0.001 * np.arange(frames, dtype=np.float32)[:, None]
    permutation = np.arange(NUM_JOINTS)[::-1]
    encoded_joint_names = np.asarray(
        [KENGO_ISAACLAB_JOINT_NAMES[index].encode("utf-8") for index in permutation]
    )
    path = tmp_path / "hiyio_wbt.npz"
    np.savez(
        path,
        fps=np.asarray(50.0),
        joint_names=encoded_joint_names,
        joint_pos=canonical_joint_pos[:, permutation],
        body_names=body_names,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
    )

    clip = load_motion_npz(path)

    assert clip.num_frames == frames
    np.testing.assert_allclose(clip.joint_pos, canonical_joint_pos)
    np.testing.assert_allclose(clip.root_pos_w, body_pos[:, 1, :])
    np.testing.assert_allclose(clip.root_quat_w, np.tile(yaw_90, (frames, 1)))
    np.testing.assert_allclose(clip.root_lin_vel_w[:-1, 0], 0.5, atol=1e-6)


_FAKE_BINARY_POLICY = r"""
import sys
import numpy as np

REQUEST_BYTES = 1270 * 4

def read_exact(size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            return None if remaining == size else (_ for _ in ()).throw(EOFError())
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)

while True:
    payload = read_exact(REQUEST_BYTES)
    if payload is None:
        break
    values = np.frombuffer(payload, dtype='<f4')
    response = (values[:23] + np.float32(0.5)).astype('<f4').tobytes()
    # Force the client to exercise read_exact instead of relying on one read.
    sys.stdout.buffer.write(response[:17])
    sys.stdout.buffer.flush()
    sys.stdout.buffer.write(response[17:])
    sys.stdout.buffer.flush()
"""


def test_binary_policy_protocol_is_persistent_and_handles_short_reads() -> None:
    client = BinaryFloat32PolicyClient(
        lambda: [sys.executable, "-u", "-c", _FAKE_BINARY_POLICY],
        label="local fake policy",
    )
    try:
        first = np.arange(POLICY_INPUT_DIM, dtype=np.float32)
        second = first + 100.0
        np.testing.assert_array_equal(client.infer(first), first[:23] + 0.5)
        np.testing.assert_array_equal(client.infer(second), second[:23] + 0.5)
        assert client.starts == 1
        assert client.reconnects == 0
    finally:
        client.close()


def test_binary_policy_allows_three_reconnects_and_replays_same_observation() -> None:
    fail_after_request = (
        "import sys; sys.stdin.buffer.read(1270 * 4); raise SystemExit(7)"
    )
    commands = [
        [sys.executable, "-u", "-c", fail_after_request],
        [sys.executable, "-u", "-c", fail_after_request],
        [sys.executable, "-u", "-c", fail_after_request],
        [sys.executable, "-u", "-c", _FAKE_BINARY_POLICY],
    ]
    starts = 0

    def command_factory() -> list[str]:
        nonlocal starts
        command = commands[min(starts, len(commands) - 1)]
        starts += 1
        return command

    client = BinaryFloat32PolicyClient(
        command_factory, label="reconnecting fake policy"
    )
    observation = np.linspace(-1.0, 1.0, POLICY_INPUT_DIM, dtype=np.float32)
    try:
        np.testing.assert_allclose(client.infer(observation), observation[:23] + 0.5)
        assert client.starts == 4
        assert client.reconnects == 3
    finally:
        client.close()


def test_remote_session_builds_noninteractive_ssh_command_and_cli_needs_no_policy() -> None:
    session = RemoteSshOnnxSession(
        "trainer@192.0.2.10",
        "/srv/gear-sonic/checkpoints/best combined.onnx",
    )
    command = session._command()
    assert command[0] == "ssh"
    assert "-T" in command
    assert "BatchMode=yes" in command
    assert "NumberOfPasswordPrompts=0" in command
    assert "ConnectTimeout=60" in command
    assert "ConnectionAttempts=4" in command
    assert "TCPKeepAlive=yes" in command
    assert "IPQoS=throughput" in command
    assert "RekeyLimit=1G" in command
    assert command[-2] == "trainer@192.0.2.10"
    assert "best combined.onnx" in command[-1]
    args = _parse_args(
        [
            "--remote-policy-ssh",
            "trainer@192.0.2.10",
            "--remote-policy-path",
            "/srv/gear-sonic/checkpoints/best.onnx",
            "--headless",
        ]
    )
    assert args.policy is None
    assert args.remote_policy_path == "/srv/gear-sonic/checkpoints/best.onnx"


def test_full_reference_uses_an_exact_integer_step_budget() -> None:
    assert _full_reference_step_budget(5411, 0) == (5411, 21644)
    assert _full_reference_step_budget(5411, 11) == (5400, 21600)


def test_full_reference_cli_forces_fall_continuation_and_rejects_other_end_modes(
) -> None:
    remote = [
        "--remote-policy-ssh",
        "trainer@192.0.2.10",
        "--remote-policy-path",
        "/srv/gear-sonic/checkpoints/best.onnx",
        "--headless",
    ]
    args = _parse_args([*remote, "--full-reference", "--stop-on-fall"])
    assert args.full_reference
    assert not args.stop_on_fall

    with pytest.raises(SystemExit):
        _parse_args([*remote, "--full-reference", "--max-sim-seconds", "1"])
    with pytest.raises(SystemExit):
        _parse_args([*remote, "--full-reference", "--loop"])
