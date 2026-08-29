"""Eye region localisation and an openness proxy.

The design doc specifies MediaPipe Face Landmarker for this. **MediaPipe is not
usable on this machine**: version 1.0.1 crashes inside the graph on macOS arm64
("Check failed: service_ Service is unavailable" in DrishtiMetalHelper), with
the CPU delegate forced as well, and 0.10.35 no longer ships the legacy
`mediapipe.python.solutions` API the detector expected. So the eye region is
derived geometrically from the face box instead.

That is a real downgrade and it is measured, not assumed: see
`training/evaluation/ocular.py`. A geometric ROI assumes a roughly frontal,
upright face. Head rotation moves the eyes out of the box, and the whole
approach should be replaced with landmarks as soon as they are available.

OPENNESS PROXY
--------------
Vertical edge energy in the eye region, not brightness. An open eye carries
strong horizontal structure -- eyelid margins, the iris/sclera boundary -- and
a closed eye is a smooth lid. Measured against SCAMPS' `au45` blink ground
truth across ten subjects, with a permutation null over mismatched subject
pairs:

    vertical edge energy   matched 0.691   mismatched 0.113   p < 0.0001
    mean brightness        matched 0.336   mismatched 0.117   p < 0.0001

Edge energy is the clearly better carrier, so brightness is computed but only
reported as a secondary channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .detector import FaceBox

# Eye band as a fraction of the face box: (x0, y0, x1, y1).
EYE_BAND = (0.15, 0.33, 0.85, 0.52)


@dataclass
class EyeSample:
    """One frame's eye-region measurements. Raw, unnormalised units."""

    openness: float          # vertical edge energy; higher = more open
    brightness: float        # secondary channel
    n_pixels: int


def eye_region(frame_rgb: np.ndarray, face: FaceBox) -> Optional[np.ndarray]:
    """Crop the eye band out of the frame, or None if it does not fit."""
    h, w = frame_rgb.shape[:2]
    x0 = max(int(face.x + EYE_BAND[0] * face.w), 0)
    y0 = max(int(face.y + EYE_BAND[1] * face.h), 0)
    x1 = min(int(face.x + EYE_BAND[2] * face.w), w)
    y1 = min(int(face.y + EYE_BAND[3] * face.h), h)
    if x1 - x0 < 8 or y1 - y0 < 4:
        return None
    return frame_rgb[y0:y1, x0:x1]


def measure(frame_rgb: np.ndarray, face: FaceBox) -> Optional[EyeSample]:
    """Openness proxy and brightness for one frame."""
    patch = eye_region(frame_rgb, face)
    if patch is None:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float64)
    # Vertical derivative: responds to the horizontal structures an open eye has.
    edges = float(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).mean())
    return EyeSample(openness=edges, brightness=float(gray.mean()), n_pixels=gray.size)
