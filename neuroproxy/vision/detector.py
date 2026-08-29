"""Face localisation with pluggable, degradable backends.

Backends, in order of fidelity:
  mediapipe -- landmark-accurate, optional dependency
  haar      -- ships with OpenCV, adequate for frontal seated subjects
  skin      -- YCrCb skin segmentation; no face model, works on synthetic
  static    -- fixed central box, for fully controlled captures

The pipeline degrades rather than crashes: a missing optional dependency drops
to the next backend and lowers the reported face quality, which then flows into
the confidence score instead of silently producing a confident wrong answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Empirical YCrCb skin locus, used ONLY to *find* a face when no face model is
# available. It must not be used to select ROI pixels once a face is known --
# see `adaptive_skin_mask` for why.
_CR_RANGE = (133, 177)
_CB_RANGE = (77, 127)

# A detection covering more than this fraction of the frame is the detector
# failing, not a face. Measured on SCAMPS: nine of ten clips gave boxes at
# 0.28-0.34 of frame area, while one gave 0.93 -- effectively the whole frame.
# That bad box put the eye band on the wrong part of the image (blink F1 0.00
# against 0.86-1.00 elsewhere) and cost the subject 91% of its cardiac windows.
MAX_FACE_AREA_FRACTION = 0.60

# Adaptive mask parameters. Radius in (Cr, Cb) around the face's own median,
# derived from the face's own spread and clamped so that neither an extremely
# uniform nor an extremely noisy face degenerates.
_ADAPTIVE_K = 3.0
_ADAPTIVE_MIN_RADIUS = 6.0
_ADAPTIVE_MAX_RADIUS = 28.0
# Pixels far darker or brighter than the face's median luma are shadow, hair or
# specular blowout, not usable skin.
_LUMA_TOLERANCE = 0.55


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    backend: str

    def clip(self, width: int, height: int) -> "FaceBox":
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        w = max(1, min(self.w, width - x))
        h = max(1, min(self.h, height - y))
        return FaceBox(x, y, w, h, self.confidence, self.backend)


def skin_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Boolean skin mask from YCrCb thresholding."""
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    mask = (
        (cr >= _CR_RANGE[0])
        & (cr <= _CR_RANGE[1])
        & (cb >= _CB_RANGE[0])
        & (cb <= _CB_RANGE[1])
    )
    # Close small holes so noise speckle does not fragment the region.
    m = (mask.astype(np.uint8)) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m > 0


def adaptive_skin_mask(frame_rgb: np.ndarray, face: "FaceBox") -> np.ndarray:
    """Skin mask derived from the detected face's own colour, not a fixed locus.

    WHY THIS REPLACED THE FIXED LOCUS FOR ROI SELECTION
    ---------------------------------------------------
    A fixed YCrCb box systematically excludes dark skin in dim light. Low
    luminance compresses chroma toward the neutral point (128, 128), so the
    same person who sits comfortably inside the locus under bright light falls
    outside it under dim light -- and darker skin starts closer to the boundary.

    Measured on SCAMPS, fraction of inner-face pixels inside the fixed locus:
    86-98% for eight of ten subjects, and **0%** for the one dark-skinned
    subject in dim lighting (median Cr 131 against a floor of 133, median Cb
    131 against a ceiling of 127 -- outside on both axes by a hair). The sensor
    layer produced no output at all for that subject.

    Widening the box would paper over that specific case while leaving the
    design wrong. The face box already tells us where skin is, so the mask is
    built from the face's own chroma distribution instead: median as centre,
    spread as radius. That is tone-agnostic and luminance-agnostic by
    construction.

    Design doc section 14 lists skin tone / device bias as a project risk;
    docs/limitations.md section 7 tracks what is and is not yet measured.
    """
    h, w = frame_rgb.shape[:2]
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)

    # Inner face region: avoids hair, ears, collar and background.
    x0 = max(int(face.x + 0.25 * face.w), 0)
    x1 = min(int(face.x + 0.75 * face.w), w)
    y0 = max(int(face.y + 0.35 * face.h), 0)
    y1 = min(int(face.y + 0.80 * face.h), h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return skin_mask(frame_rgb)

    sample = ycrcb[y0:y1, x0:x1].reshape(-1, 3)
    luma_c = float(np.median(sample[:, 0]))
    cr_c = float(np.median(sample[:, 1]))
    cb_c = float(np.median(sample[:, 2]))

    # Robust spread of the face's own chroma sets the acceptance radius.
    spread = float(
        np.median(np.abs(sample[:, 1] - cr_c) + np.abs(sample[:, 2] - cb_c))
    )
    radius = float(
        np.clip(_ADAPTIVE_K * spread, _ADAPTIVE_MIN_RADIUS, _ADAPTIVE_MAX_RADIUS)
    )

    dist = np.abs(ycrcb[:, :, 1] - cr_c) + np.abs(ycrcb[:, :, 2] - cb_c)
    luma_ok = np.abs(ycrcb[:, :, 0] - luma_c) <= max(_LUMA_TOLERANCE * luma_c, 25.0)
    mask = (dist <= radius) & luma_ok

    m = cv2.morphologyEx(
        (mask.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    return m > 0


class FaceDetector:
    """Locate a face box per frame using the requested backend."""

    def __init__(self, backend: str = "auto") -> None:
        self.requested = backend
        self.backend = backend
        self._haar: Optional[cv2.CascadeClassifier] = None
        self._mp = None
        if backend in ("auto", "mediapipe"):
            self._mp = self._try_mediapipe()
            if self._mp is not None:
                self.backend = "mediapipe"
            elif backend == "mediapipe":
                raise RuntimeError("mediapipe backend requested but not installed")
        if self.backend in ("auto", "haar"):
            self._haar = self._load_haar()
            if self._haar is not None and self.backend == "auto":
                self.backend = "haar"
        if self.backend == "auto":
            self.backend = "skin"

    @staticmethod
    def _try_mediapipe():
        try:
            import mediapipe as mp  # noqa: F401
        except Exception:
            return None
        try:
            from mediapipe.python.solutions import face_detection

            return face_detection.FaceDetection(min_detection_confidence=0.5)
        except Exception:
            return None

    @staticmethod
    def _load_haar() -> Optional[cv2.CascadeClassifier]:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        clf = cv2.CascadeClassifier(path)
        return None if clf.empty() else clf

    def detect(self, frame_rgb: np.ndarray) -> Optional[FaceBox]:
        h, w = frame_rgb.shape[:2]
        if self.backend == "mediapipe":
            box = self._detect_mediapipe(frame_rgb)
            if box is not None:
                return box.clip(w, h)
        if self.backend in ("mediapipe", "haar") and self._haar is not None:
            box = self._detect_haar(frame_rgb)
            if box is not None:
                return box.clip(w, h)
        if self.backend == "static":
            return FaceBox(
                int(w * 0.25), int(h * 0.15), int(w * 0.5), int(h * 0.7), 0.5, "static"
            ).clip(w, h)
        box = self._detect_skin(frame_rgb)
        return box.clip(w, h) if box is not None else None

    def _detect_mediapipe(self, frame_rgb: np.ndarray) -> Optional[FaceBox]:
        res = self._mp.process(frame_rgb)
        if not getattr(res, "detections", None):
            return None
        det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
        rel = det.location_data.relative_bounding_box
        h, w = frame_rgb.shape[:2]
        score = float(det.score[0]) if det.score else 0.9
        return FaceBox(
            int(rel.xmin * w), int(rel.ymin * h),
            int(rel.width * w), int(rel.height * h),
            score, "mediapipe",
        )

    def _detect_haar(self, frame_rgb: np.ndarray) -> Optional[FaceBox]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        frame_area = float(frame_rgb.shape[0] * frame_rgb.shape[1])
        # Take the largest *plausible* detection, not simply the largest. A
        # whole-frame false positive is otherwise always the biggest one, and
        # everything downstream -- ROI, eye band, quality -- is measured
        # relative to this box, so one bad box corrupts every feature at once.
        plausible = [
            f for f in faces if (f[2] * f[3]) / frame_area <= MAX_FACE_AREA_FRACTION
        ]
        if not plausible:
            return None
        x, y, w, h = max(plausible, key=lambda f: f[2] * f[3])
        # Haar gives no score; treat a clean single detection as moderately
        # confident and let downstream quality metrics do the real gating.
        return FaceBox(int(x), int(y), int(w), int(h), 0.85, "haar")

    def _detect_skin(self, frame_rgb: np.ndarray) -> Optional[FaceBox]:
        mask = skin_mask(frame_rgb)
        if mask.sum() < 32:
            return None
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if num <= 1:
            return None
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[idx]
        frame_area = frame_rgb.shape[0] * frame_rgb.shape[1]
        # Confidence from how compact and how plausibly sized the blob is.
        fill = area / float(max(w * h, 1))
        size_ok = 0.01 <= area / float(frame_area) <= 0.8
        conf = float(np.clip(fill, 0.0, 1.0)) * (1.0 if size_ok else 0.3)
        return FaceBox(int(x), int(y), int(w), int(h), conf * 0.8, "skin")
