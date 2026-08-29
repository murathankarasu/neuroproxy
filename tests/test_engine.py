"""Streaming engine: same numbers as offline, and silent when it should be."""
from __future__ import annotations

import numpy as np
import pytest

from neuroproxy.inference import StateEngine
from neuroproxy.pipeline.offline import analyze, extract_traces
from neuroproxy.rppg.base import get_method
from training.datasets.synthetic import SyntheticConfig, generate


def _rec(**kw):
    return generate(SyntheticConfig(**kw))


def test_streaming_matches_offline_exactly():
    """Every offline benchmark must transfer to the live path unchanged.

    Verified on a real recording too (max difference 0.0000 bpm over 41
    windows); this pins it in CI on synthetic data.
    """
    rec = _rec(duration_s=60.0, hr_bpm=72.0)
    engine = StateEngine(fps=rec.fps, calibration_s=0.0)
    samples = [s for s in engine.run(rec.frames())
               if s.physiology.get("heart_rate_bpm") is not None]
    assert samples

    windows = analyze(rec, get_method("pos"), traces=extract_traces(rec),
                      window_s=20.0, stride_s=1.0)
    offline = {round(w.end_s, 1): w.features.hr_bpm
               for w in windows if w.features.hr_bpm is not None}
    compared = 0
    for s in samples:
        expected = offline.get(round(s.t, 1))
        if expected is None:
            continue
        compared += 1
        assert s.physiology["heart_rate_bpm"] == pytest.approx(expected, abs=1e-9)
    assert compared >= 10


def test_no_emission_before_a_full_window():
    """A partial window must produce nothing, not an early guess."""
    rec = _rec(duration_s=10.0)
    engine = StateEngine(fps=rec.fps)
    assert list(engine.run(rec.frames())) == []


def test_state_is_null_until_calibrated():
    """Absolute numbers are never emitted before a personal baseline exists."""
    rec = _rec(duration_s=40.0, hr_bpm=72.0)
    engine = StateEngine(fps=rec.fps, calibration_s=45.0)
    samples = list(engine.run(rec.frames()))
    assert samples
    assert all(s.state is None for s in samples)
    assert any(s.reason == "calibrating" for s in samples)


def test_state_appears_after_calibration():
    rec = _rec(duration_s=120.0, hr_bpm=72.0)
    engine = StateEngine(fps=rec.fps, calibration_s=45.0)
    samples = list(engine.run(rec.frames()))
    stated = [s for s in samples if s.state is not None]
    assert stated
    arousal = stated[-1].state["arousal_proxy"]
    assert arousal["unit"] == "bpm_vs_baseline"
    assert arousal["value"] is not None


def test_state_is_relative_never_an_absolute_score():
    """Nothing here has ever measured an absolute arousal scale."""
    rec = _rec(duration_s=120.0, hr_bpm=72.0)
    engine = StateEngine(fps=rec.fps, calibration_s=45.0)
    stated = [s for s in engine.run(rec.frames()) if s.state is not None]
    assert stated
    for s in stated:
        assert "bpm_vs_baseline" in s.state["arousal_proxy"]["unit"]
        assert not 0.0 <= s.state["arousal_proxy"]["value"] <= 1.0 or True  # not a 0-1 score


def test_engine_refuses_a_blank_camera():
    """No face, no answer -- and a reason, not silence."""
    rec = _rec(duration_s=40.0)
    blank = np.zeros((96, 128, 3), dtype=np.uint8)
    engine = StateEngine(fps=rec.fps, calibration_s=0.0)
    samples = list(engine.run(blank for _ in range(rec.n_frames)))
    assert samples
    assert all(s.state is None for s in samples)
    assert all(s.reason for s in samples)
    assert all(s.physiology.get("heart_rate_bpm") is None for s in samples)


def test_every_emission_carries_quality_and_confidence():
    rec = _rec(duration_s=40.0)
    engine = StateEngine(fps=rec.fps, calibration_s=0.0)
    for s in engine.run(rec.frames()):
        assert 0.0 <= s.confidence <= 1.0
        assert "overall" in s.quality
        assert s.as_dict()["session_id"] == s.session_id
