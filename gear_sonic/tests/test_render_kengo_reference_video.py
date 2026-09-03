from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.scripts.render_kengo_reference_video import (
    VIDEO_FPS,
    reference_video_timeline,
)


def test_timeline_covers_full_50hz_reference_at_exact_30fps() -> None:
    reference_frames = 2591
    presentation_times, reference_indices = reference_video_timeline(
        reference_frames, 50.0
    )

    assert len(presentation_times) == 1555
    assert len(reference_indices) == 1555
    assert presentation_times[0] == 0.0
    np.testing.assert_allclose(np.diff(presentation_times), 1.0 / VIDEO_FPS)
    assert reference_indices[0] == 0
    assert reference_indices[-1] == reference_frames - 1
    assert np.all(np.diff(reference_indices) >= 0)
    assert reference_indices.min() == 0
    assert reference_indices.max() == reference_frames - 1

    reference_duration = reference_frames / 50.0
    video_duration = len(presentation_times) / VIDEO_FPS
    assert video_duration >= reference_duration
    assert video_duration - reference_duration < 1.0 / VIDEO_FPS


def test_timeline_uses_midpoint_mapping_without_integer_duration_drift() -> None:
    presentation_times, reference_indices = reference_video_timeline(60, 50.0)

    assert len(presentation_times) == 36
    assert len(presentation_times) / VIDEO_FPS == pytest.approx(60 / 50.0)
    np.testing.assert_array_equal(reference_indices[:6], [0, 2, 4, 5, 7, 9])
    assert reference_indices[-1] == 59


@pytest.mark.parametrize(
    ("frames", "reference_fps", "video_fps"),
    [
        (0, 50.0, 30.0),
        (3, 0.0, 30.0),
        (3, float("nan"), 30.0),
        (3, 50.0, float("inf")),
    ],
)
def test_timeline_rejects_invalid_inputs(
    frames: int, reference_fps: float, video_fps: float
) -> None:
    with pytest.raises(ValueError):
        reference_video_timeline(frames, reference_fps, video_fps)
