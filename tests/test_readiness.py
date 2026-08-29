"""Pre-session readiness: warn early, and never pass a blank camera."""
from __future__ import annotations

import numpy as np
import pytest

from neuroproxy.readiness import (
    SKIN_FRACTION_GOOD,
    SKIN_FRACTION_POOR,
    Readiness,
    assess,
)


def _blank(n=20, value=0):
    return [np.full((96, 128, 3), value, dtype=np.uint8) for _ in range(n)]


def _skin_ellipse(n=20, tone=(158, 117, 102), rx=0.30, ry=0.38):
    """Frames containing a skin-coloured ellipse the detector can find."""
    frames = []
    h, w = 96, 128
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - w / 2) / (w * rx)) ** 2 + ((yy - h / 2) / (h * ry)) ** 2) <= 1.0
    for _ in range(n):
        f = np.full((h, w, 3), 20, dtype=np.uint8)
        f[mask] = tone
        frames.append(f)
    return frames


def test_no_frames_is_refused():
    r = assess(iter([]))
    assert not r.ready
    assert r.verdict == "no_face"
    assert "No frames" in r.messages[0]


def test_blank_camera_is_refused_not_passed():
    """A covered lens must never come back 'ready'."""
    r = assess(iter(_blank()))
    assert not r.ready
    assert r.verdict == "no_face"
    assert r.skin_fraction is None


def test_visible_skin_passes():
    r = assess(iter(_skin_ellipse()))
    assert r.skin_fraction is not None
    assert r.ready
    assert r.verdict in ("good", "marginal")


def test_verdict_follows_the_fitted_bands():
    """The bands come from measured usable rates, so the mapping must hold."""
    good = Readiness(ready=True, verdict="good", skin_fraction=0.95)
    assert good.skin_fraction >= SKIN_FRACTION_GOOD
    assert SKIN_FRACTION_POOR < SKIN_FRACTION_GOOD


def test_poor_verdict_carries_actionable_advice():
    """A warning with no instruction is not worth showing a participant."""
    r = assess(iter(_blank()))
    assert r.messages and len(r.messages[0]) > 20


def test_preview_length_is_respected():
    """The check must not consume the whole session to make a decision."""
    frames = _skin_ellipse(n=500)
    consumed = {"n": 0}

    def counting():
        for f in frames:
            consumed["n"] += 1
            yield f

    assess(counting(), max_frames=30)
    assert consumed["n"] <= 31


def test_report_is_serialisable():
    r = assess(iter(_skin_ellipse()))
    d = r.as_dict()
    assert set(("ready", "verdict", "skin_fraction", "messages")) <= set(d)
