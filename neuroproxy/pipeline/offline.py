"""Offline video -> windowed state pipeline.

This is the batch twin of the realtime loop in design doc section 4.2. Running
offline first is deliberate: it is reproducible, it can be compared against
contact ground truth, and it fails loudly instead of dropping frames.

    Recording -> per-frame ROI colour + quality
              -> rolling windows (20 s, 1 s stride)
              -> rPPG method -> BVP -> HR / SNR
              -> paired ground-truth HR from the same window
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..confidence import Confidence
from ..confidence import score as confidence_score
from ..features.cardiac import CardiacFeatures
from ..features.cardiac import extract as cardiac_features
from ..quality import metrics as qm
from ..quality.metrics import FrameQuality, WindowQuality
from ..rppg.base import RPPGMethod
from ..rppg.signal import hr_from_psd, welch_psd
from ..vision.detector import FaceDetector, adaptive_skin_mask
from ..vision.eyes import measure as measure_eyes
from ..vision.roi import extract as roi_extract

# A window with fewer than this fraction of usable frames is not reported.
MIN_VALID_FRAME_RATIO = 0.75
# Below this window quality the engine refuses to emit a state.
MIN_WINDOW_QUALITY = 0.35
# Compression is a property of the stream, not of any single frame, so it is
# sampled rather than computed per frame and carried forward between samples.
COMPRESSION_SAMPLE_EVERY = 15

# Face-crop size for frame-based (neural) methods. 72 is baked into the
# released EfficientPhys checkpoints, not a free choice.
DEFAULT_CROP_SIZE = 72


@dataclass
class Traces:
    """Per-frame outputs for a whole recording."""

    rgb: np.ndarray                      # (T, 3), NaN where no ROI was found
    timestamps: np.ndarray               # (T,) seconds
    quality: List[FrameQuality]
    valid: np.ndarray                    # (T,) bool
    fps: float
    # (T, S, S, 3) uint8 face crops, populated only when a frame-based method
    # will need them. None keeps the classical path free of the memory cost.
    crops: Optional[np.ndarray] = None
    # Per-frame eye openness proxy, NaN where the eye region was unusable.
    openness: Optional[np.ndarray] = None

    @property
    def valid_ratio(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


@dataclass
class WindowResult:
    """One analysis window: prediction, ground truth and why to trust it."""

    subject_id: str
    start_s: float
    end_s: float
    hr_pred_bpm: Optional[float]
    hr_gt_bpm: Optional[float]
    quality: WindowQuality
    features: CardiacFeatures = field(default_factory=CardiacFeatures)
    confidence: float = 0.0
    valid: bool = False
    reason: Optional[str] = None

    @property
    def abs_error(self) -> Optional[float]:
        if self.hr_pred_bpm is None or self.hr_gt_bpm is None:
            return None
        return abs(self.hr_pred_bpm - self.hr_gt_bpm)


def extract_traces(
    recording,
    detector: Optional[FaceDetector] = None,
    crop_size: Optional[int] = None,
) -> Traces:
    """Run detection + ROI averaging over every frame of a recording.

    `crop_size` additionally stores square face crops, which frame-based
    methods consume instead of the colour trace. Requested explicitly because
    it costs roughly 28 MB per minute of video at 72x72.
    """
    if detector is None:
        hint = str(recording.metadata.get("detector", "auto"))
        detector = FaceDetector(hint)

    rgb: List[np.ndarray] = []
    quals: List[FrameQuality] = []
    valid: List[bool] = []
    openness: List[float] = []
    crops: Optional[List[np.ndarray]] = [] if crop_size else None
    prev_center: Optional[np.ndarray] = None
    compression = 1.0

    for i, frame in enumerate(recording.frames()):
        if i % COMPRESSION_SAMPLE_EVERY == 0:
            compression = qm.compression_score(frame)
        face = detector.detect(frame)
        if face is None:
            rgb.append(np.full(3, np.nan))
            quals.append(FrameQuality())
            valid.append(False)
            openness.append(np.nan)
            if crops is not None:
                crops.append(
                    crops[-1] if crops
                    else np.zeros((crop_size, crop_size, 3), np.uint8)
                )
            prev_center = None
            continue

        # Derived from this face's own colour, not a fixed locus: a fixed
        # locus excluded a dark-skinned subject entirely (see
        # vision.detector.adaptive_skin_mask).
        mask = adaptive_skin_mask(frame, face)
        sample = roi_extract(frame, face, mask=mask)
        if crops is not None:
            crops.append(_face_crop(frame, face, crop_size))
        eye = measure_eyes(frame, face)
        openness.append(np.nan if eye is None else eye.openness)
        # Judged on the face box, not the frame: a bright or busy backdrop
        # is not an exposure fault of the subject.
        light, clipped = qm.lighting_score(frame, face)
        sharp = qm.sharpness_score(frame, face)

        center = np.array([face.x + face.w / 2.0, face.y + face.h / 2.0])
        disp = 0.0 if prev_center is None else float(np.linalg.norm(center - prev_center))
        prev_center = center

        fq = FrameQuality(
            face=face.confidence,
            lighting=light,
            sharpness=sharp,
            motion=qm.motion_score(disp, face.w),
            skin_fraction=sample.skin_fraction if sample else 0.0,
            clipped_fraction=clipped,
            compression=compression,
        )
        quals.append(fq)
        if sample is None:
            rgb.append(np.full(3, np.nan))
            valid.append(False)
        else:
            rgb.append(sample.rgb)
            valid.append(True)

    arr = np.asarray(rgb, dtype=np.float64)
    ts = np.arange(len(rgb)) / recording.fps
    return Traces(
        rgb=arr,
        timestamps=ts,
        quality=quals,
        valid=np.asarray(valid, dtype=bool),
        fps=recording.fps,
        openness=np.asarray(openness, dtype=np.float64),
        crops=np.stack(crops) if crops else None,
    )


def _face_crop(frame: np.ndarray, face, size: int) -> np.ndarray:
    """Square face crop resized for a frame-based model."""
    h, w = frame.shape[:2]
    x0, y0 = max(int(face.x), 0), max(int(face.y), 0)
    x1, y1 = min(int(face.x + face.w), w), min(int(face.y + face.h), h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(frame[y0:y1, x0:x1], (size, size),
                      interpolation=cv2.INTER_AREA)


def _fill_short_gaps(rgb: np.ndarray, valid: np.ndarray, fps: float,
                     max_gap_s: float = 0.5) -> Optional[np.ndarray]:
    """Interpolate brief dropouts; refuse to invent data across long ones.

    Blinks and momentary detector misses are short. A two-second gap is a real
    loss of signal and must not be smoothed over -- it would fabricate spectral
    content exactly where the HR peak is read.
    """
    if valid.all():
        return rgb
    if not valid.any():
        return None
    max_gap = int(max_gap_s * fps)
    idx = np.arange(rgb.shape[0])
    # Reject if any single gap is too long.
    gap = 0
    for v in valid:
        gap = 0 if v else gap + 1
        if gap > max_gap:
            return None
    out = rgb.copy()
    for c in range(rgb.shape[1]):
        out[~valid, c] = np.interp(idx[~valid], idx[valid], rgb[valid, c])
    return out


def _gt_hr_for_window(recording, start: int, stop: int, fps: float) -> Optional[float]:
    """Ground-truth HR over the same window, via the same estimator.

    Using an identical estimator on both sides keeps the comparison about the
    optical signal rather than about two different peak-picking conventions.
    """
    if not recording.has_gt:
        return None
    from training.datasets.base import resample_to

    gt = resample_to(recording.gt_bvp, recording.gt_bvp_fps, fps, recording.n_frames)
    seg = gt[start:stop]
    if seg.size < int(fps * 4):
        return None
    freqs, psd = welch_psd(seg, fps)
    return hr_from_psd(freqs, psd)


def analyze(
    recording,
    method: RPPGMethod,
    traces: Optional[Traces] = None,
    window_s: float = 20.0,
    stride_s: float = 1.0,
    detector: Optional[FaceDetector] = None,
) -> List[WindowResult]:
    """Produce one WindowResult per rolling window of a recording."""
    if traces is None:
        traces = extract_traces(
            recording,
            detector=detector,
            crop_size=DEFAULT_CROP_SIZE if getattr(method, "needs_frames", False) else None,
        )
    if getattr(method, "needs_frames", False) and traces.crops is None:
        raise ValueError(
            "method {!r} needs face crops; call extract_traces(..., crop_size=...) "
            "so the crops are available".format(getattr(method, "name", method))
        )

    fps = traces.fps
    win = int(round(window_s * fps))
    stride = max(int(round(stride_s * fps)), 1)
    n = traces.rgb.shape[0]
    results: List[WindowResult] = []

    if n < win:
        return results

    for start in range(0, n - win + 1, stride):
        stop = start + win
        seg_valid = traces.valid[start:stop]
        seg_quals = [q for q, v in zip(traces.quality[start:stop], seg_valid) if v]
        wq = qm.aggregate(seg_quals, traces.timestamps[start:stop])
        wq.valid_frame_ratio = float(seg_valid.mean())

        base = WindowResult(
            subject_id=recording.subject_id,
            start_s=start / fps,
            end_s=stop / fps,
            hr_pred_bpm=None,
            hr_gt_bpm=_gt_hr_for_window(recording, start, stop, fps),
            quality=wq,
        )

        if wq.valid_frame_ratio < MIN_VALID_FRAME_RATIO:
            base.reason = "insufficient_valid_frames"
            results.append(base)
            continue

        seg_rgb = _fill_short_gaps(traces.rgb[start:stop], seg_valid, fps)
        if seg_rgb is None:
            base.reason = "signal_gap_too_long"
            results.append(base)
            continue

        if getattr(method, "needs_frames", False):
            bvp = method(traces.crops[start:stop], fps)
        else:
            bvp = method(seg_rgb, fps)
        feats = cardiac_features(bvp, fps)
        wq.pulse_snr_db = feats.pulse_snr_db
        wq.hr_stability_bpm = feats.hr_stability_bpm
        base.features = feats

        # Confidence needs the pulse SNR, so it is scored after extraction --
        # but the *prediction* is still withheld when it comes out low.
        conf = confidence_score(wq)
        base.confidence = conf.value

        if wq.overall < MIN_WINDOW_QUALITY:
            base.reason = "low_quality"
            results.append(base)
            continue
        if conf.abstain:
            base.reason = "low_confidence"
            results.append(base)
            continue

        base.hr_pred_bpm = feats.hr_bpm
        base.valid = feats.hr_bpm is not None
        if not base.valid:
            base.reason = "no_hr_peak"
        results.append(base)

    return results
