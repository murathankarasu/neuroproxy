"""Streaming state engine: frames in, state out, once per second.

The offline pipeline answers "what was this recording's heart rate". A product
needs "what is happening now, and should I believe it" -- continuously, while
the session runs, without ever waiting for the session to end.

DESIGN NOTES

* **Frame-source agnostic.** The engine consumes any iterable of RGB frames, so
  a webcam, a file and a synthetic generator all drive the same code. That is
  not only tidiness: it means the realtime path is testable without a camera,
  and every offline benchmark result transfers to it directly.

* **1 Hz output, 20 s window.** Design doc section 9: emitting state at 30 FPS
  is meaningless because the underlying physiology does not move that fast. The
  window slides continuously; only the reporting is throttled.

* **Calibration then deviation.** The first `calibration_seconds` build the
  subject's personal baseline. Before that completes the engine reports state as
  `null` with reason `calibrating` -- it does not emit absolute numbers it
  cannot contextualise (docs/limitations.md section 11 on why deviation and not
  z-scores).

* **Silence is a valid output.** Every emission carries quality and confidence,
  and `state` is `null` with a reason whenever the engine will not stand behind
  a number. On real recordings that is roughly half of all sessions
  (limitations 15-16), which is a fact about cameras, not a bug to paper over.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, Iterable, Iterator, List, Optional

import numpy as np

from ..calibration import PersonalBaseline, feature_dict
from ..calibration import fit as fit_baseline
from ..confidence import score as confidence_score
from ..features.cardiac import CardiacFeatures
from ..features.cardiac import extract as cardiac_features
from ..pipeline.offline import MIN_VALID_FRAME_RATIO, _fill_short_gaps
from ..quality import metrics as qm
from ..quality.metrics import FrameQuality, WindowQuality
from ..rppg.base import RPPGMethod, get_method
from ..vision.detector import FaceDetector, adaptive_skin_mask
from ..vision.roi import extract as roi_extract

DEFAULT_WINDOW_S = 20.0
DEFAULT_EMIT_HZ = 1.0
DEFAULT_CALIBRATION_S = 45.0
COMPRESSION_SAMPLE_EVERY = 15


@dataclass
class StateSample:
    """One emission. Mirrors the API contract in design doc section 10.1."""

    session_id: str
    t: float                                  # seconds since session start
    physiology: Dict[str, Optional[float]] = field(default_factory=dict)
    state: Optional[Dict[str, object]] = None
    quality: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    reason: Optional[str] = None
    calibrated: bool = False

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class StateEngine:
    """Rolling-window engine producing `StateSample`s from a frame stream."""

    def __init__(
        self,
        session_id: str = "session",
        fps: float = 30.0,
        method: Optional[RPPGMethod] = None,
        window_s: float = DEFAULT_WINDOW_S,
        emit_hz: float = DEFAULT_EMIT_HZ,
        calibration_s: float = DEFAULT_CALIBRATION_S,
        detector: Optional[FaceDetector] = None,
    ) -> None:
        self.session_id = session_id
        self.fps = fps
        self.method = method or get_method("pos")
        self.window = int(round(window_s * fps))
        self.emit_every = max(int(round(fps / max(emit_hz, 1e-6))), 1)
        self.calibration_s = calibration_s
        self.detector = detector or FaceDetector("auto")

        self._rgb: Deque[np.ndarray] = deque(maxlen=self.window)
        self._valid: Deque[bool] = deque(maxlen=self.window)
        self._quality: Deque[FrameQuality] = deque(maxlen=self.window)
        self._n_frames = 0
        self._compression = 1.0
        self._prev_center: Optional[np.ndarray] = None

        # Calibration accumulates completed windows until the baseline is fit.
        self._calibration_windows: List[object] = []
        self.baseline: Optional[PersonalBaseline] = None

    # -- ingestion ---------------------------------------------------------

    def push(self, frame: np.ndarray) -> Optional[StateSample]:
        """Feed one frame. Returns a sample on emission ticks, else None."""
        self._ingest(frame)
        self._n_frames += 1
        if self._n_frames < self.window:
            return None
        if self._n_frames % self.emit_every != 0:
            return None
        return self._emit()

    def run(self, frames: Iterable[np.ndarray]) -> Iterator[StateSample]:
        """Drive the engine from any frame source."""
        for frame in frames:
            sample = self.push(frame)
            if sample is not None:
                yield sample

    def _ingest(self, frame: np.ndarray) -> None:
        if self._n_frames % COMPRESSION_SAMPLE_EVERY == 0:
            self._compression = qm.compression_score(frame)

        face = self.detector.detect(frame)
        if face is None:
            self._rgb.append(np.full(3, np.nan))
            self._valid.append(False)
            self._quality.append(FrameQuality())
            self._prev_center = None
            return

        mask = adaptive_skin_mask(frame, face)
        sample = roi_extract(frame, face, mask=mask)
        light, clipped = qm.lighting_score(frame, face)
        sharp = qm.sharpness_score(frame, face)

        center = np.array([face.x + face.w / 2.0, face.y + face.h / 2.0])
        disp = 0.0 if self._prev_center is None else float(
            np.linalg.norm(center - self._prev_center))
        self._prev_center = center

        self._quality.append(FrameQuality(
            face=face.confidence,
            lighting=light,
            sharpness=sharp,
            motion=qm.motion_score(disp, face.w),
            skin_fraction=sample.skin_fraction if sample else 0.0,
            clipped_fraction=clipped,
            compression=self._compression,
        ))
        self._rgb.append(sample.rgb if sample else np.full(3, np.nan))
        self._valid.append(bool(sample is not None))

    # -- emission ----------------------------------------------------------

    def _emit(self) -> StateSample:
        t = self._n_frames / self.fps
        valid = np.asarray(self._valid, dtype=bool)
        quals = [q for q, v in zip(self._quality, valid) if v]
        timestamps = np.arange(len(valid)) / self.fps

        wq = qm.aggregate(quals, timestamps)
        wq.valid_frame_ratio = float(valid.mean()) if valid.size else 0.0

        sample = StateSample(
            session_id=self.session_id, t=t,
            calibrated=bool(self.baseline and self.baseline.ready),
        )

        if wq.valid_frame_ratio < MIN_VALID_FRAME_RATIO:
            sample.reason = "insufficient_valid_frames"
            sample.quality = _quality_dict(wq)
            return sample

        rgb = _fill_short_gaps(np.asarray(self._rgb, dtype=np.float64), valid, self.fps)
        if rgb is None:
            sample.reason = "signal_gap_too_long"
            sample.quality = _quality_dict(wq)
            return sample

        bvp = self.method(rgb, self.fps)
        feats = cardiac_features(bvp, self.fps)
        wq.pulse_snr_db = feats.pulse_snr_db
        wq.hr_stability_bpm = feats.hr_stability_bpm

        conf = confidence_score(wq)
        sample.confidence = conf.value
        sample.quality = _quality_dict(wq)
        sample.physiology = {
            "heart_rate_bpm": feats.hr_bpm,
            "pulse_snr_db": feats.pulse_snr_db,
        }

        if conf.abstain or feats.hr_bpm is None:
            sample.reason = "low_confidence" if conf.abstain else "no_hr_peak"
            return sample

        self._maybe_calibrate(t, feats, conf.value)
        if not (self.baseline and self.baseline.ready):
            sample.reason = "calibrating"
            return sample

        sample.calibrated = True
        sample.state = self._state_from(feats)
        return sample

    def _maybe_calibrate(self, t: float, feats: CardiacFeatures, confidence: float) -> None:
        """Collect calm opening windows, then fit the baseline once."""
        if self.baseline and self.baseline.ready:
            return
        if t <= self.calibration_s:
            self._calibration_windows.append(_CalibrationWindow(
                start_s=t - self.window / self.fps, end_s=t,
                features=feats, valid=True, confidence=confidence))
            return
        if self._calibration_windows:
            self.baseline = fit_baseline(
                self._calibration_windows, calibration_seconds=self.calibration_s)

    def _state_from(self, feats: CardiacFeatures) -> Dict[str, object]:
        """State as deviation from the subject's own baseline.

        Deliberately NOT an absolute 0-1 score. Nothing in this project has
        ever measured an absolute arousal scale, and reporting one would imply
        a calibration that does not exist. `arousal_proxy` is the baseline
        deviation in bpm, named so that it cannot be mistaken for a measurement
        of arousal itself.
        """
        assert self.baseline is not None
        deltas = self.baseline.transform(feature_dict(feats))
        hr_delta = deltas.get("hr_bpm")
        return {
            "arousal_proxy": {
                "value": hr_delta,
                "unit": "bpm_vs_baseline",
                "baseline_bpm": self.baseline.features["hr_bpm"].location
                if "hr_bpm" in self.baseline.features else None,
            }
        }


@dataclass
class _CalibrationWindow:
    """Shape `calibration.fit` expects, without importing the offline result."""

    start_s: float
    end_s: float
    features: CardiacFeatures
    valid: bool
    confidence: float


def _quality_dict(wq: WindowQuality) -> Dict[str, float]:
    return {
        "overall": wq.overall,
        "face": wq.face,
        "lighting": wq.lighting,
        "motion": wq.motion,
        "compression": wq.compression,
        "valid_frame_ratio": wq.valid_frame_ratio,
    }
