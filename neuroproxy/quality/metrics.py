"""Signal-quality gate (design doc section 4.3).

The product rule this implements: when the camera cannot see well, the engine
must widen its uncertainty or refuse to answer -- never emit a confident state.
Quality is therefore computed per frame, aggregated per window, and multiplied
into confidence downstream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

# Luminance outside this range means under/over-exposed skin.
GOOD_LUMA = (60.0, 200.0)
# Laplacian variance below this reads as defocus or heavy compression blur.
BLUR_FLOOR = 8.0
# Per-frame face-box displacement, as a fraction of box width.
MOTION_TOLERANCE = 0.02

# Compression detection. Chroma subsampling is the single largest measured
# threat to POS/CHROM (docs/limitations.md section 3), and until now nothing in
# the quality gate could see it. Reference values measured on the synthetic
# generator: chroma/luma high-frequency ratio is ~0.07 uncompressed and drops
# to ~0.003 at any JPEG quality, i.e. it detects the *presence* of chroma
# subsampling sharply; blockiness then grades its severity.
CHROMA_RATIO_FLOOR = 0.001    # scores 0
CHROMA_RATIO_GOOD = 0.05      # scores 1
BLOCKINESS_SCALE = 8.0


@dataclass
class FrameQuality:
    face: float = 0.0
    lighting: float = 0.0
    sharpness: float = 0.0
    motion: float = 1.0
    skin_fraction: float = 0.0
    clipped_fraction: float = 0.0
    compression: float = 1.0

    @property
    def overall(self) -> float:
        return float(
            np.clip(
                0.30 * self.face
                + 0.20 * self.lighting
                + 0.18 * self.motion
                + 0.15 * self.sharpness
                + 0.17 * self.compression,
                0.0,
                1.0,
            )
        )


@dataclass
class WindowQuality:
    """Window-level aggregate, plus the reason a window was rejected."""

    face: float = 0.0
    lighting: float = 0.0
    sharpness: float = 0.0
    motion: float = 0.0
    skin_fraction: float = 0.0
    compression: float = 1.0
    fps_stability: float = 1.0
    valid_frame_ratio: float = 0.0
    pulse_snr_db: Optional[float] = None
    hr_stability_bpm: Optional[float] = None
    reason: Optional[str] = None

    @property
    def overall(self) -> float:
        base = float(
            np.clip(
                0.25 * self.face
                + 0.16 * self.lighting
                + 0.16 * self.motion
                + 0.12 * self.sharpness
                + 0.19 * self.compression
                + 0.12 * self.fps_stability,
                0.0,
                1.0,
            )
        )
        return base * float(np.clip(self.valid_frame_ratio, 0.0, 1.0))

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["overall"] = self.overall
        return d


def _luma(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)


def _crop_to_face(frame_rgb: np.ndarray, face=None) -> np.ndarray:
    """Crop to the face box when one is known.

    Exposure and focus must be judged on the face, not the frame. Measured on
    SCAMPS: a dark-skinned subject against a blown-out white background scored
    0.66 on frame-level lighting because 34% of the *background* was clipped,
    while the face itself was 6% clipped and scored 0.94. All three subjects
    the engine could not measure were being penalised for their backdrop --
    and all three were dark-skin, dim-light or bright-background cases, i.e.
    the error ran consistently in one direction.
    """
    if face is None:
        return frame_rgb
    h, w = frame_rgb.shape[:2]
    x0, y0 = max(int(face.x), 0), max(int(face.y), 0)
    x1, y1 = min(int(face.x + face.w), w), min(int(face.y + face.h), h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame_rgb
    return frame_rgb[y0:y1, x0:x1]


def lighting_score(frame_rgb: np.ndarray, face=None) -> "tuple":
    """Score exposure on the face, and report the fraction of clipped pixels."""
    gray = _luma(_crop_to_face(frame_rgb, face))
    mean = float(gray.mean())
    clipped = float(((gray <= 2) | (gray >= 253)).mean())
    lo, hi = GOOD_LUMA
    if mean < lo:
        score = mean / lo
    elif mean > hi:
        score = max(0.0, 1.0 - (mean - hi) / (255.0 - hi))
    else:
        score = 1.0
    return float(np.clip(score * (1.0 - clipped), 0.0, 1.0)), clipped


def sharpness_score(frame_rgb: np.ndarray, face=None) -> float:
    """Laplacian variance on the face, squashed to [0, 1].

    Judged on the face for the same reason as exposure: a sharp face in front
    of a defocused background is a good capture, not a bad one.
    """
    gray = _luma(_crop_to_face(frame_rgb, face))
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(var / (var + BLUR_FLOOR), 0.0, 1.0))


def blockiness(frame_rgb: np.ndarray) -> float:
    """8x8 DCT block-edge energy relative to non-edge energy.

    Zero on uncompressed frames, rising with compression severity.
    """
    g = _luma(frame_rgb).astype(np.float64)
    if g.shape[0] < 16 or g.shape[1] < 16:
        return 0.0
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    jb = np.arange(dh.shape[1]) % 8 == 7
    ib = np.arange(dv.shape[0]) % 8 == 7
    if not jb.any() or not ib.any() or (~jb).sum() == 0 or (~ib).sum() == 0:
        return 0.0
    on = (dh[:, jb].mean() + dv[ib, :].mean()) / 2.0
    off = (dh[:, ~jb].mean() + dv[~ib, :].mean()) / 2.0
    return float(max((on - off) / (off + 1e-6), 0.0))


def chroma_detail_ratio(frame_rgb: np.ndarray) -> float:
    """High-frequency energy in Cr/Cb relative to Y.

    Chroma subsampling removes chroma detail while leaving luma intact, so
    this collapses as soon as a frame has been through a 4:2:0 codec. That is
    precisely the information POS and CHROM read the pulse out of.
    """
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    lv = lambda c: float(cv2.Laplacian(c, cv2.CV_64F).var())
    luma = lv(ycrcb[:, :, 0])
    chroma = (lv(ycrcb[:, :, 1]) + lv(ycrcb[:, :, 2])) / 2.0
    return float(chroma / (luma + 1e-6))


def compression_score(frame_rgb: np.ndarray) -> float:
    """1.0 for pristine frames, falling as codec damage appears.

    The two indicators compound rather than compete: the chroma ratio detects
    *whether* a frame went through a subsampling codec (it saturates almost
    immediately, so it cannot grade severity), while blockiness grades *how
    hard* it was compressed. Taking the minimum lets the saturated chroma term
    swallow the severity information, so they are multiplied.

    NOTE for real data: essentially every webcam and every browser stream is
    already compressed, so this term will be low across the board on real
    recordings. That is not a bug -- it reflects real risk to the pulse signal
    -- but it means absolute confidence values are not comparable between
    compressed and uncompressed sources. Ranking within a source still holds.
    """
    ratio = chroma_detail_ratio(frame_rgb)
    lo, hi = np.log10(CHROMA_RATIO_FLOOR), np.log10(CHROMA_RATIO_GOOD)
    chroma_term = float(np.clip((np.log10(max(ratio, 1e-9)) - lo) / (hi - lo), 0.0, 1.0))
    block_term = float(1.0 / (1.0 + BLOCKINESS_SCALE * blockiness(frame_rgb)))
    return float(chroma_term * block_term)


def motion_score(displacement_px: float, face_width: float) -> float:
    """1.0 for a still head, decaying as the box moves between frames."""
    if face_width <= 0:
        return 0.0
    rel = displacement_px / face_width
    return float(np.clip(1.0 - rel / (MOTION_TOLERANCE * 4.0), 0.0, 1.0))


def fps_stability(timestamps: np.ndarray) -> float:
    """1 - coefficient of variation of frame intervals.

    Dropped frames break the constant-rate assumption every filter here makes,
    so an unstable capture must lower confidence even if the image looks fine.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size < 3:
        return 0.0
    dt = np.diff(ts)
    if dt.mean() <= 0:
        return 0.0
    cv_ = dt.std() / dt.mean()
    return float(np.clip(1.0 - cv_, 0.0, 1.0))


def aggregate(frames: List[FrameQuality], timestamps: np.ndarray) -> WindowQuality:
    """Combine per-frame quality into a window verdict.

    Uses the 10th percentile rather than the mean for face/motion: a window is
    only as good as its worst stretch, and a two-second occlusion in a
    twenty-second window destroys the spectrum regardless of the average.
    """
    if not frames:
        return WindowQuality(reason="no_valid_frames")

    def p10(vals: List[float]) -> float:
        return float(np.percentile(vals, 10))

    return WindowQuality(
        face=p10([f.face for f in frames]),
        lighting=float(np.mean([f.lighting for f in frames])),
        sharpness=float(np.mean([f.sharpness for f in frames])),
        motion=p10([f.motion for f in frames]),
        skin_fraction=float(np.mean([f.skin_fraction for f in frames])),
        compression=float(np.mean([f.compression for f in frames])),
        fps_stability=fps_stability(timestamps),
        valid_frame_ratio=1.0,
    )
