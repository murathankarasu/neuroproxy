"""Blink detection: find real dips, refuse to invent them."""
from __future__ import annotations

import numpy as np
import pytest

from neuroproxy.features.ocular import (
    BLINK_DROP_RATIO,
    detect_blinks,
    extract as ocular_features,
)
from neuroproxy.vision.detector import FaceBox
from neuroproxy.vision.eyes import eye_region, measure
from training.evaluation.ocular import evaluate, gt_blink_onsets, match_events

FS = 30.0


def _series(n=600, base=40.0, blinks=(), width=5, noise=0.0, seed=0):
    x = np.full(n, base, dtype=float)
    if noise:
        x += np.random.default_rng(seed).normal(0, noise, n)
    for on in blinks:
        x[on : on + width] = base * 0.4
    return x


def test_finds_injected_blinks():
    x = _series(blinks=(60, 200, 350, 500), noise=1.0)
    f = ocular_features(x, FS)
    assert f.blink_count == 4
    assert f.blink_rate_per_min == pytest.approx(12.0, abs=0.1)


def test_flat_signal_yields_no_blinks():
    """A rare-event detector must not manufacture events from noise."""
    assert ocular_features(_series(noise=1.0), FS).blink_count == 0


def test_sparse_events_survive_the_scale_estimate():
    """Regression: a MAD-based threshold collapsed on exactly this case.

    Blinks are rare and brief, so the spread of the series is dominated by the
    open-eye baseline. With a statistical threshold the scale estimate went to
    zero and the detector found nothing on a perfectly clean signal.
    """
    assert ocular_features(_series(blinks=(60, 200, 350, 500), noise=0.0), FS).blink_count == 4


@pytest.mark.parametrize("width,expected", [(1, 0), (2, 0), (5, 1), (60, 0)])
def test_duration_bounds_are_enforced(width, expected):
    """Too brief is noise; too long is an occlusion or a downward gaze.

    A blink lasts 100-400 ms, i.e. 3-12 frames at 30 fps. One or two frames is
    sensor noise.
    """
    assert ocular_features(_series(blinks=(200,), width=width), FS).blink_count == expected


def test_refractory_period_rejects_doubles():
    x = _series(blinks=(200, 203), width=3)
    assert ocular_features(x, FS).blink_count == 1


def test_unusable_eye_region_reports_a_reason():
    x = _series()
    x[: int(0.8 * x.size)] = np.nan
    f = ocular_features(x, FS)
    assert f.blink_count == 0
    assert f.reason and "unusable" in f.reason


def test_eye_region_rejects_a_box_that_does_not_fit():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    assert eye_region(frame, FaceBox(0, 0, 4, 2, 0.9, "test")) is None


def test_openness_is_higher_for_structured_eyes_than_a_blank_lid():
    """The proxy is edge energy, so structure must score above smoothness.

    The band is period-4, not period-2: a 3x3 vertical Sobel is exactly zero on
    a pattern that alternates every row, because the rows above and below any
    given row are then identical. That is a real null of the operator at
    Nyquist, and a period-2 test image measures it instead of the code.
    """
    box = FaceBox(0, 0, 100, 100, 0.9, "test")
    blank = np.full((100, 100, 3), 128, dtype=np.uint8)
    striped = blank.copy()
    striped[34:52:4, :, :] = 40          # horizontal structure in the eye band
    assert measure(blank, box).openness == pytest.approx(0.0)
    assert measure(striped, box).openness > 1.0


def test_f1_is_zero_not_missing_when_nothing_is_detected():
    """Otherwise a detector that fires never is dropped from the mean.

    The first run of this evaluation reported mean F1 0.75 while the detector
    had a recall of 0.13, because the eight subjects it found nothing for were
    excluded as 'undefined' instead of scored as zero.
    """
    au45 = np.zeros(600)
    au45[100:110] = 1.0
    ev = evaluate("s", [], au45, FS)
    assert ev.n_gt == 1
    assert ev.f1 == 0.0


def test_event_matching_respects_the_tolerance():
    truth = [100, 300]
    assert match_events([103, 303], truth, FS) == 2          # within 0.3 s
    assert match_events([160, 360], truth, FS) == 0          # 2 s away
    # One detection cannot claim two ground-truth events.
    assert match_events([100], truth, FS) == 1


def test_gt_onsets_count_transitions_not_frames():
    au45 = np.zeros(100)
    au45[10:20] = 1.0
    au45[50:62] = 1.0
    assert gt_blink_onsets(au45).tolist() == [10, 50]
