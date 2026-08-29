"""Confidence must refuse when there is no pulse to measure.

These pin a defect found by measurement, not by review: with the design doc's
single weighted sum, a well-lit still recording of a face containing no pulse
scored 0.637 and answered 100% of its windows with a mean error of 25.2 bpm.
Image quality outvoted the complete absence of a signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuroproxy.confidence import pulse_quality, score
from neuroproxy.pipeline.offline import analyze
from neuroproxy.quality.metrics import WindowQuality
from neuroproxy.rppg.base import get_method
from neuroproxy.rppg.signal import bandpass, hr_stability
from training.datasets.synthetic import SyntheticConfig, generate
from training.evaluation.harness import counterfactual_error


def _pristine_image_quality(**overrides) -> WindowQuality:
    kw = dict(
        face=0.95,
        lighting=0.95,
        motion=0.95,
        compression=1.0,
        fps_stability=1.0,
        valid_frame_ratio=1.0,
        pulse_snr_db=16.0,
        hr_stability_bpm=0.2,
    )
    kw.update(overrides)
    return WindowQuality(**kw)


def test_perfect_image_cannot_rescue_absent_pulse():
    """The regression this whole module exists for."""
    good = score(_pristine_image_quality())
    no_pulse = score(_pristine_image_quality(pulse_snr_db=-1.0, hr_stability_bpm=30.0))
    assert good.value > 0.6 and not good.abstain
    assert no_pulse.value < 0.1
    assert no_pulse.abstain


def test_missing_stability_is_evidence_of_absence():
    """A sub-window with no peak means no pulse, not 'unknown'."""
    assert pulse_quality(16.0, None) == 0.0
    assert score(_pristine_image_quality(hr_stability_bpm=None)).abstain


def test_hr_stability_separates_pulse_from_filtered_noise():
    """SNR cannot make this distinction; sub-window agreement can."""
    fs, n = 30.0, 600
    t = np.arange(n) / fs
    pulse = bandpass(np.sin(2 * np.pi * 1.2 * t), fs)
    noise = bandpass(np.random.default_rng(21).normal(size=n), fs)
    assert hr_stability(pulse, fs) < 1.0
    assert hr_stability(noise, fs) > 5.0


def test_engine_abstains_on_pulseless_video():
    """End-to-end: no pulse painted into the pixels, so no answers."""
    rec = generate(SyntheticConfig(duration_s=40.0, pulse_amplitude=0.0, noise_sigma=6.0))
    windows = analyze(rec, get_method("pos"), window_s=20.0, stride_s=5.0)
    assert windows
    assert all(not w.valid for w in windows)
    # And the errors it avoided were large, not incidental.
    errors = [e for e in (counterfactual_error(w) for w in windows) if e is not None]
    assert errors and float(np.mean(errors)) > 10.0


def test_engine_still_answers_a_real_pulse():
    """Abstention must not be achieved by refusing everything."""
    rec = generate(SyntheticConfig(duration_s=40.0, pulse_amplitude=0.012))
    windows = analyze(rec, get_method("pos"), window_s=20.0, stride_s=5.0)
    answered = [w for w in windows if w.valid]
    assert len(answered) == len(windows)
    assert float(np.median([w.abs_error for w in answered])) < 2.0


@pytest.mark.parametrize("term", ["face", "lighting", "motion", "compression"])
def test_image_terms_reduce_but_do_not_zero_confidence(term):
    """Image degradation should lower confidence without acting as a gate."""
    base = score(_pristine_image_quality()).value
    worse = score(_pristine_image_quality(**{term: 0.1})).value
    assert worse < base
    assert worse > 0.0


def test_stability_subwindows_must_be_distinct():
    """A fixed sub-window length equal to the analysis window is a silent no-op.

    With sub_seconds=10 and a 10 s window the three sub-windows coincided, the
    spread was always exactly 0.0, and the pulse gate passed everything. The
    default is now a fraction of the window, and a degenerate request returns
    None rather than a flattering zero.
    """
    fs = 30.0
    noise = bandpass(np.random.default_rng(3).normal(size=int(fs * 10)), fs)
    # Explicitly degenerate: sub-window as long as the window itself.
    assert hr_stability(noise, fs, sub_seconds=10.0) is None
    # Default scales with the window and separates noise from a pulse.
    t = np.arange(int(fs * 10)) / fs
    pulse = bandpass(np.sin(2 * np.pi * 1.2 * t), fs)
    assert hr_stability(pulse, fs) is not None
    assert hr_stability(pulse, fs) < hr_stability(noise, fs)
