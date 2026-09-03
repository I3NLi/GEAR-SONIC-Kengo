#!/usr/bin/env python3
"""Serve one combined Kengo SONIC ONNX model over a binary stdio stream.

Protocol (little-endian, with no framing or stdout text):

* request: 1270 contiguous float32 values (5080 bytes)
* response: 23 contiguous float32 values (92 bytes)

The protocol is deliberately stateless: every response depends only on its
request.  That makes it safe for a client to reconnect and replay the current
observation after a broken SSH pipe.  Diagnostics are written only to stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import BinaryIO

import numpy as np


INPUT_DIM = 1270
OUTPUT_DIM = 23
FLOAT_DTYPE = np.dtype("<f4")
REQUEST_BYTES = INPUT_DIM * FLOAT_DTYPE.itemsize
RESPONSE_BYTES = OUTPUT_DIM * FLOAT_DTYPE.itemsize


def _read_exact_or_eof(stream: BinaryIO, size: int) -> bytes | None:
    """Read one fixed-size message, returning None only for clean stream EOF."""

    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return None
            received = size - remaining
            raise EOFError(f"truncated request: received {received} of {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args(argv)
    args.policy = args.policy.expanduser().resolve()
    if not args.policy.is_file():
        parser.error(f"--policy file not found: {args.policy}")
    return args


def run(policy_path: Path) -> int:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed; install onnxruntime>=1.18,<2"
        ) from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(policy_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or not outputs:
        raise RuntimeError("combined SONIC ONNX must expose one input and an output")
    policy_input = inputs[0]
    policy_output = outputs[0]
    if policy_input.type != "tensor(float)" or policy_output.type != "tensor(float)":
        raise RuntimeError(
            "combined SONIC ONNX input/output must both use float32; "
            f"got {policy_input.type} -> {policy_output.type}"
        )
    if len(policy_input.shape) != 2 or len(policy_output.shape) != 2:
        raise RuntimeError(
            "combined SONIC ONNX must be rank two; "
            f"got {policy_input.shape} -> {policy_output.shape}"
        )
    expected_dimensions = (
        ("input batch", policy_input.shape[0], 1),
        ("input feature", policy_input.shape[1], INPUT_DIM),
        ("output batch", policy_output.shape[0], 1),
        ("output feature", policy_output.shape[1], OUTPUT_DIM),
    )
    mismatches = [
        f"{label}={actual} (expected {expected})"
        for label, actual, expected in expected_dimensions
        if isinstance(actual, int) and actual != expected
    ]
    if mismatches:
        raise RuntimeError(
            f"ONNX shape mismatch {policy_input.shape} -> {policy_output.shape}: "
            + ", ".join(mismatches)
        )

    source = sys.stdin.buffer
    destination = sys.stdout.buffer
    while True:
        payload = _read_exact_or_eof(source, REQUEST_BYTES)
        if payload is None:
            return 0
        observation = np.frombuffer(payload, dtype=FLOAT_DTYPE).reshape(1, INPUT_DIM)
        raw_output = session.run(
            [policy_output.name], {policy_input.name: observation}
        )[0]
        action = np.asarray(raw_output, dtype=np.float32).reshape(-1)
        if action.shape != (OUTPUT_DIM,) or not np.isfinite(action).all():
            raise RuntimeError(
                f"policy returned invalid action shape/value: {action.shape}, "
                f"finite={np.isfinite(action).all()}"
            )
        destination.write(action.astype(FLOAT_DTYPE, copy=False).tobytes(order="C"))
        destination.flush()


def main(argv: list[str] | None = None) -> int:
    try:
        return run(_parse_args(argv).policy)
    except BrokenPipeError:
        # Normal when the local simulator closes its SSH process.
        return 0
    except Exception as exc:
        print(f"[REMOTE_ONNX_ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
