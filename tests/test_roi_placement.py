"""ROI placement is per-ROI, and the forehead is the one that moves.

Shifting every ROI by a single hairline offset was measured and rejected
(MAE_all 4.00 -> 4.28). Moving only the forehead was measured and kept
(4.00 -> 2.87). These pin that distinction so it is not undone by a later
"simplification".
"""
from __future__ import annotations

import numpy as np
import pytest

from neuroproxy.vision.detector import FaceBox
from neuroproxy.vision.roi import (
    FOREHEAD_BELOW_HAIRLINE,
    ROI_BOXES,
    extract,
    forehead_from_hairline,
    hairline_row,
)


def _face_with_hair(hair_rows=30, h=200, w=160):
    """A skin mask whose top `hair_rows` rows are hair, not skin."""
    mask = np.zeros((h, w), dtype=bool)
    mask[hair_rows:, int(w * 0.15) : int(w * 0.85)] = True
    face = FaceBox(0, 0, w, h, 0.9, "test")
    return mask, face


def test_hairline_is_found_below_the_box_top():
    mask, face = _face_with_hair(hair_rows=30)
    row = hairline_row(mask, face)
    assert row is not None
    assert 28 <= row <= 34


def test_forehead_band_sits_below_the_hairline():
    mask, face = _face_with_hair(hair_rows=30)
    band = forehead_from_hairline(mask, face)
    assert band is not None
    x0, y0, x1, y1 = band
    hair_frac = 30 / face.h
    assert y0 == pytest.approx(hair_frac + FOREHEAD_BELOW_HAIRLINE[0], abs=0.02)
    assert y1 > y0
    assert y1 <= 0.45          # must never reach the eyes


def test_forehead_band_is_refused_when_it_would_reach_the_eyes():
    """A very low hairline must fall back rather than sample the eyes."""
    mask, face = _face_with_hair(hair_rows=90)   # hairline at 45% of face height
    assert forehead_from_hairline(mask, face) is None


def test_no_hairline_falls_back_to_the_default_box():
    mask = np.ones((200, 160), dtype=bool)       # skin everywhere, no hair
    face = FaceBox(0, 0, 160, 200, 0.9, "test")
    band = forehead_from_hairline(mask, face)
    # Either no shift is needed, or the band starts essentially at the top.
    assert band is None or band[1] <= FOREHEAD_BELOW_HAIRLINE[0] + 0.02


def test_cheeks_are_never_moved_by_the_hairline():
    """The regression that made things worse: cheeks sliding onto the jaw."""
    frame = np.zeros((200, 160, 3), dtype=np.uint8)
    frame[:, :] = (150, 110, 95)
    mask, face = _face_with_hair(hair_rows=40)
    sample = extract(frame, face, mask=mask, anchor_forehead=True)
    assert sample is not None
    # extract() must not have mutated the shared default boxes.
    assert ROI_BOXES["left_cheek"] == (0.14, 0.52, 0.40, 0.76)
    assert ROI_BOXES["right_cheek"] == (0.60, 0.52, 0.86, 0.76)


def test_anchoring_can_be_switched_off():
    frame = np.zeros((200, 160, 3), dtype=np.uint8)
    frame[:, :] = (150, 110, 95)
    mask, face = _face_with_hair(hair_rows=40)
    on = extract(frame, face, mask=mask, anchor_forehead=True)
    off = extract(frame, face, mask=mask, anchor_forehead=False)
    assert on is not None and off is not None
