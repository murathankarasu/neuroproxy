"""End-to-end properties of the offline pipeline and the quality gate."""
from __future__ import annotations

import numpy as np
import pytest

from neuroproxy.pipeline.offline import _fill_short_gaps, analyze, extract_traces
from neuroproxy.rppg.base import get_method
from neuroproxy.vision import FaceDetector, extract as roi_extract
from training.datasets.base import resample_to
from training.datasets.synthetic import SyntheticConfig, generate


def _rec(**kw):
    cfg = SyntheticConfig(duration_s=kw.pop("duration_s", 30.0), **kw)
    return generate(cfg)


def test_pos_recovers_injected_pulse():
    """The correctness floor: POS must find a pulse we painted into the pixels.

    Failing this means a harness bug, never a physiology result.
    """
    rec = _rec(hr_bpm=72.0)
    results = analyze(rec, get_method("pos"), window_s=20.0, stride_s=5.0)
    errors = [w.abs_error for w in results if w.abs_error is not None]
    assert errors, "no window produced a comparable HR"
    assert float(np.median(errors)) < 2.0


def test_darkness_suppresses_output():
    """With no visible face the engine must refuse, not guess."""
    black = np.zeros((96, 128, 3), dtype=np.uint8)
    rec = _rec(duration_s=30.0)
    rec.frame_source = lambda: iter([black] * rec.n_frames)
    results = analyze(rec, get_method("pos"), window_s=20.0, stride_s=5.0)
    assert results
    assert all(not w.valid for w in results)
    assert all(w.hr_pred_bpm is None for w in results)
    assert all(w.reason for w in results)


def test_long_gap_is_not_interpolated():
    """Short dropouts may be filled; a long one must invalidate the window."""
    rgb = np.tile(np.array([100.0, 90.0, 80.0]), (300, 1))
    valid = np.ones(300, dtype=bool)
    valid[100:105] = False                       # ~0.17 s at 30 fps
    assert _fill_short_gaps(rgb, valid, 30.0) is not None
    valid[100:140] = False                       # ~1.3 s
    assert _fill_short_gaps(rgb, valid, 30.0) is None


def test_ground_truth_aligned_by_time_not_index():
    """GT at a different rate must be resampled on the time axis."""
    src_fps, dst_fps, seconds = 64.0, 30.0, 10.0
    t = np.arange(int(src_fps * seconds)) / src_fps
    gt = np.sin(2 * np.pi * 1.2 * t)
    out = resample_to(gt, src_fps, dst_fps, int(dst_fps * seconds))
    t2 = np.arange(out.size) / dst_fps
    assert out.size == int(dst_fps * seconds)
    assert np.allclose(out, np.sin(2 * np.pi * 1.2 * t2), atol=0.02)


def test_roi_rejects_non_skin_frame():
    """A uniformly blue frame contains no skin and must yield no ROI."""
    blue = np.zeros((96, 128, 3), dtype=np.uint8)
    blue[:, :, 2] = 220
    det = FaceDetector("skin")
    assert det.detect(blue) is None


def test_quality_drops_when_frames_are_dropped():
    """An unstable capture rate must lower window quality."""
    from neuroproxy.quality.metrics import fps_stability

    steady = np.arange(300) / 30.0
    jittery = steady + np.random.default_rng(0).normal(0, 0.02, steady.size)
    assert fps_stability(steady) > fps_stability(jittery)


@pytest.mark.parametrize("method", ["pos", "chrom", "green"])
def test_methods_return_normalised_signal(method):
    """Every method returns a zero-mean, unit-variance BVP estimate."""
    rec = _rec(duration_s=10.0)
    traces = extract_traces(rec)
    bvp = get_method(method)(traces.rgb, rec.fps)
    assert bvp.shape[0] == traces.rgb.shape[0]
    assert abs(float(bvp.mean())) < 1e-6
    assert float(bvp.std()) == pytest.approx(1.0, abs=1e-6)
