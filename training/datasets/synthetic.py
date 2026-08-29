"""Synthetic rPPG recordings with exactly known ground truth.

WHY THIS EXISTS
---------------
Before any public dataset is downloaded we need to answer one question:
"is the harness itself correct?" A synthetic recording whose pulse we injected
ourselves separates harness bugs (wrong resampling, wrong window alignment,
wrong PSD scaling) from real-world signal problems. If POS cannot recover a
pulse we literally painted into the pixels, no dataset result is interpretable.

WHAT IT DOES NOT SHOW
---------------------
A low MAE here is NOT evidence of real-world accuracy. Real skin is not a flat
ellipse, real motion is non-rigid, real auto-exposure fights back, and real
subjects vary in skin tone, ballistocardiographic motion and makeup. Treat
synthetic numbers as a correctness floor and nothing else -- see
docs/limitations.md.

The optical model follows the dichromatic reflection model that POS and CHROM
are derived from, so this is a fair test of those methods' algebra:

    C(t) = I(t) * ( u_skin * (1 + s(t)) + u_pbv * a * p(t) ) + n(t)

with I(t) slow illumination drift, s(t) motion-linked specular variation,
u_pbv the blood-volume-pulse colour signature, and n(t) sensor noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .base import Dataset, Recording, register

# Blood-volume-pulse colour signature for RGB cameras (de Haan & van Leest,
# 2014). Green carries most of the pulse because haemoglobin absorbs there.
PBV_RGB = np.array([0.33, 0.78, 0.53])
PBV_RGB = PBV_RGB / np.linalg.norm(PBV_RGB)


@dataclass
class SyntheticConfig:
    """Knobs for generating a recording. Each maps to a real-world failure."""

    duration_s: float = 60.0
    fps: float = 30.0
    width: int = 128
    height: int = 96
    # Physiology
    hr_bpm: float = 72.0
    hr_drift_bpm: float = 6.0        # slow HR wander over the session
    # Drift must not be phase-locked to the session, or it becomes a confound
    # rather than a nuisance. With a period equal to the recording length, the
    # drift's positive half always covered rest and its negative half always
    # covered the task, which inverted task-vs-rest separability entirely at
    # task responses below the drift amplitude. Period and phase are now
    # independent of the protocol, and the phase is drawn per subject.
    hr_drift_period_s: float = 70.0
    hr_drift_phase: Optional[float] = None   # None -> derived from `seed`
    rsa_bpm: float = 3.0             # respiratory sinus arrhythmia depth
    # Task-induced HR rise: (start_s, end_s, delta_bpm). Models the rest ->
    # task -> recovery protocol in the explainer document section 10, which is
    # what a personal baseline is supposed to make comparable across people.
    hr_event: Optional[Tuple[float, float, float]] = None
    hr_event_ramp_s: float = 4.0     # sympathetic response is not a step
    respiration_hz: float = 0.25     # 15 breaths/min
    pulse_amplitude: float = 0.012   # ~1% modulation, realistic for skin
    # Degradations
    noise_sigma: float = 1.5         # additive sensor noise, in 8-bit levels
    illum_drift: float = 0.06        # fractional low-frequency light drift
    motion_px: float = 0.0           # peak head translation in pixels
    specular_gain: float = 0.006     # shading change per pixel of motion
    jpeg_quality: Optional[int] = None  # None = no compression round-trip
    # Appearance
    skin_tone: Optional[np.ndarray] = None  # RGB triple in [0, 1]
    seed: int = 0


def _instantaneous_hr(cfg: SyntheticConfig, t: np.ndarray) -> np.ndarray:
    """Non-stationary HR: slow wander, RSA, and an optional task response.

    A constant HR would let a broken estimator look perfect, so we never use one.
    """
    phase = cfg.hr_drift_phase
    if phase is None:
        phase = float(np.random.default_rng(cfg.seed + 9973).uniform(0.0, 2 * np.pi))
    drift = cfg.hr_drift_bpm * np.sin(
        2 * np.pi * t / max(cfg.hr_drift_period_s, 1e-6) + phase
    )
    rsa = cfg.rsa_bpm * np.sin(2 * np.pi * cfg.respiration_hz * t)
    hr = cfg.hr_bpm + drift + rsa
    if cfg.hr_event is not None:
        hr = hr + _event_profile(cfg, t)
    return hr


def _event_profile(cfg: SyntheticConfig, t: np.ndarray) -> np.ndarray:
    """Smooth rise-hold-recover profile for a task-induced HR change."""
    start, end, delta = cfg.hr_event
    ramp = max(cfg.hr_event_ramp_s, 1e-3)
    rise = np.clip((t - start) / ramp, 0.0, 1.0)
    fall = np.clip((end - t) / ramp, 0.0, 1.0)
    return delta * np.clip(np.minimum(rise, fall), 0.0, 1.0)


def _pulse_waveform(phase: np.ndarray) -> np.ndarray:
    """Asymmetric PPG-like pulse: fundamental plus two harmonics."""
    p = (
        np.sin(phase)
        + 0.25 * np.sin(2 * phase + 0.6)
        + 0.12 * np.sin(3 * phase + 1.1)
    )
    return p / np.abs(p).max()


# Head motion is not a sum of two slow tones. Real head-motion spectra are
# broadband with a 1/f-like roll-off: postural sway dominates in amplitude, but
# there is continuous power up through several Hz from micro-adjustments,
# breathing and ballistocardiographic recoil. A generator with only sub-0.2 Hz
# motion is silently trivial -- the band-pass deletes it before any rPPG method
# sees it -- and a generator with two in-band tones is worse than trivial,
# because a tone that lands near the true HR makes errors cancel instead of
# accumulate. Broadband is both more realistic and more diagnostic.
_SWAY_HZ = (0.11, 0.17)
_SWAY_WEIGHTS = (0.6, 0.4)
_MOTION_BAND_HZ = (0.05, 4.0)


def _broadband_motion(n: int, fps: float, rng) -> np.ndarray:
    """Unit-variance motion with a 1/f-like spectrum over the motion band."""
    from scipy import signal as sps

    x = rng.normal(0.0, 1.0, n)
    nyq = fps / 2.0
    lo = _MOTION_BAND_HZ[0] / nyq
    hi = min(_MOTION_BAND_HZ[1] / nyq, 0.99)
    if n > 32 and lo < hi:
        b, a = sps.butter(2, [lo, hi], btype="bandpass")
        x = sps.filtfilt(b, a, x, method="gust")
    # 1/f shaping: cumulative sum then re-band-pass emphasises low frequencies
    # while leaving real power in the HR band.
    x = np.cumsum(x)
    if n > 32 and lo < hi:
        b, a = sps.butter(2, [lo, hi], btype="bandpass")
        x = sps.filtfilt(b, a, x, method="gust")
    sd = x.std()
    return x / sd if sd > 1e-12 else x


def _motion_and_specular(cfg: SyntheticConfig, t: np.ndarray, rng):
    """Head translation and the shading artefact it induces.

    The artefact scales the skin colour vector, i.e. it is a common-mode
    multiplicative intensity change. That is exactly the component POS and
    CHROM are constructed to cancel and the raw green channel is not, so this
    axis separates the methods for the right physical reason rather than by
    construction.

    NOT modelled: non-rigid facial deformation, tracker failure, partial
    occlusion, and rolling-shutter interaction. Those need real video; see
    docs/limitations.md.
    """
    n = t.size
    sway = np.zeros(n)
    for f, w in zip(_SWAY_HZ, _SWAY_WEIGHTS):
        sway += w * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))

    fps = 1.0 / np.median(np.diff(t)) if n > 2 else 30.0
    mx = cfg.motion_px * (0.6 * sway + 0.4 * _broadband_motion(n, fps, rng))
    my = cfg.motion_px * (0.6 * sway + 0.4 * _broadband_motion(n, fps, rng))

    if cfg.motion_px <= 0:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    shading = (mx + my) / (2.0 * cfg.motion_px)
    specular = cfg.specular_gain * cfg.motion_px * shading
    return mx, my, specular


def generate(cfg: SyntheticConfig, subject_id: str = "synth") -> Recording:
    """Render one synthetic recording and its exact ground-truth BVP."""
    rng = np.random.default_rng(cfg.seed)
    n = int(round(cfg.duration_s * cfg.fps))
    t = np.arange(n) / cfg.fps

    # Ground truth: integrate instantaneous frequency into phase, so the BVP
    # and the HR track are consistent by construction.
    hr_track = _instantaneous_hr(cfg, t)
    phase = 2 * np.pi * np.cumsum(hr_track / 60.0) / cfg.fps
    bvp = _pulse_waveform(phase)

    skin = cfg.skin_tone
    if skin is None:
        skin = np.array([0.62, 0.46, 0.40])
    skin = np.asarray(skin, dtype=np.float64)

    illum = 1.0 + cfg.illum_drift * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6.28))
    mx, my, specular = _motion_and_specular(cfg, t, rng)

    yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width].astype(np.float64)
    cx, cy = cfg.width / 2.0, cfg.height / 2.0
    rx, ry = cfg.width * 0.30, cfg.height * 0.38
    background = np.array([0.10, 0.11, 0.13])

    frames: List[np.ndarray] = []
    for i in range(n):
        mask = (((xx - cx - mx[i]) / rx) ** 2 + ((yy - cy - my[i]) / ry) ** 2) <= 1.0
        colour = illum[i] * (
            skin * (1.0 + specular[i]) + PBV_RGB * cfg.pulse_amplitude * bvp[i]
        )
        img = np.empty((cfg.height, cfg.width, 3), dtype=np.float64)
        img[:] = background * illum[i]
        img[mask] = colour
        img = img * 255.0
        if cfg.noise_sigma > 0:
            img += rng.normal(0.0, cfg.noise_sigma, img.shape)
        frame = np.clip(img, 0, 255).astype(np.uint8)
        if cfg.jpeg_quality is not None:
            frame = _jpeg_roundtrip(frame, cfg.jpeg_quality)
        frames.append(frame)

    stacked = np.stack(frames)

    return Recording(
        subject_id=subject_id,
        fps=cfg.fps,
        n_frames=n,
        frame_source=lambda: iter(stacked),
        gt_bvp=bvp,
        gt_bvp_fps=cfg.fps,
        labels={
            "hr_bpm_mean": float(hr_track.mean()),
            "hr_bpm_resting": float(cfg.hr_bpm),
            "hr_event": cfg.hr_event,
        },
        metadata={
            "synthetic": True,
            "detector": "skin",  # a painted ellipse is not a Haar-detectable face
            "hr_track": hr_track,
            "config": cfg,
        },
    )


def _jpeg_roundtrip(frame: np.ndarray, quality: int) -> np.ndarray:
    """Encode/decode as JPEG to emulate webcam compression loss.

    This is the most under-appreciated degradation: chroma subsampling and
    quantisation attack exactly the ~1% colour modulation rPPG depends on.
    """
    import cv2

    bgr = frame[:, :, ::-1]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return frame
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded[:, :, ::-1].copy()


@register
class SyntheticDataset(Dataset):
    """A small cohort of synthetic subjects with differing tone and HR."""

    name = "synthetic"

    def __init__(
        self,
        root: Optional[Path] = None,
        n_subjects: int = 4,
        duration_s: float = 60.0,
        **overrides,
    ) -> None:
        super().__init__(root)
        self.n_subjects = n_subjects
        self.duration_s = duration_s
        self.overrides = overrides

    def is_available(self) -> bool:
        return True  # generated on demand; needs nothing on disk

    def recordings(self, limit: Optional[int] = None) -> List[Recording]:
        n = self.n_subjects if limit is None else min(limit, self.n_subjects)
        out = []
        # Deliberately spread skin tone and HR so a method that only works for
        # one appearance shows up as high between-subject variance.
        tones = [
            np.array([0.78, 0.62, 0.55]),
            np.array([0.62, 0.46, 0.40]),
            np.array([0.45, 0.32, 0.27]),
            np.array([0.30, 0.21, 0.17]),
        ]
        for i in range(n):
            cfg = SyntheticConfig(
                duration_s=self.duration_s,
                hr_bpm=58.0 + 11.0 * i,
                skin_tone=tones[i % len(tones)],
                seed=i,
                **self.overrides,
            )
            out.append(generate(cfg, subject_id="synth{:02d}".format(i)))
        return out


# --- Mixed-quality cohort -------------------------------------------------

# Degradation levels spanning "good webcam in a lit room" to "bad laptop
# camera in a dim room on a compressed call". Confidence is only testable
# against a cohort whose quality actually varies; a clean cohort produces
# near-zero error everywhere and nothing to rank.
STRESS_LEVELS = [
    # (label, noise_sigma, illum_drift, motion_px, jpeg_quality)
    ("clean", 0.5, 0.02, 0.0, None),
    ("good", 1.5, 0.06, 2.0, 95),
    ("fair", 3.0, 0.12, 5.0, 85),
    ("poor", 6.0, 0.25, 9.0, 70),
    ("bad", 12.0, 0.45, 14.0, 50),
]


@register
class SyntheticStressDataset(Dataset):
    """Synthetic subjects spread across capture-quality levels.

    Used by the confidence-calibration evaluation, which needs windows that
    genuinely differ in difficulty. Each subject is assigned one quality level,
    so between-subject variation in error is driven by capture conditions
    rather than physiology.
    """

    name = "synthetic_stress"

    def __init__(
        self,
        root: Optional[Path] = None,
        subjects_per_level: int = 2,
        duration_s: float = 60.0,
    ) -> None:
        super().__init__(root)
        self.subjects_per_level = subjects_per_level
        self.duration_s = duration_s

    def is_available(self) -> bool:
        return True

    def recordings(self, limit: Optional[int] = None) -> List[Recording]:
        tones = [
            np.array([0.78, 0.62, 0.55]),
            np.array([0.62, 0.46, 0.40]),
            np.array([0.45, 0.32, 0.27]),
            np.array([0.30, 0.21, 0.17]),
        ]
        out: List[Recording] = []
        k = 0
        for label, noise, drift, motion, jpeg in STRESS_LEVELS:
            for j in range(self.subjects_per_level):
                cfg = SyntheticConfig(
                    duration_s=self.duration_s,
                    hr_bpm=58.0 + 9.0 * (k % 5),
                    skin_tone=tones[k % len(tones)],
                    noise_sigma=noise,
                    illum_drift=drift,
                    motion_px=motion,
                    jpeg_quality=jpeg,
                    seed=k,
                )
                rec = generate(cfg, subject_id="{}{:02d}".format(label, j))
                rec.labels["quality_level"] = label
                out.append(rec)
                k += 1
                if limit is not None and len(out) >= limit:
                    return out
        return out


# --- Rest / task / recovery protocol --------------------------------------

@register
class SyntheticProtocolDataset(Dataset):
    """Subjects running the rest -> task -> recovery protocol.

    Explainer document section 10, Phase 2. Resting HR is spread wider across
    subjects (58-88 bpm) than the task response is tall (+12 bpm), which is the
    realistic situation and the whole reason a personal baseline is needed: a
    subject-independent model reading absolute HR sees person differences, not
    task differences.
    """

    name = "synthetic_protocol"

    def __init__(
        self,
        root: Optional[Path] = None,
        n_subjects: int = 6,
        rest_s: float = 60.0,
        task_s: float = 60.0,
        recovery_s: float = 60.0,
        task_delta_bpm: float = 12.0,
        **overrides,
    ) -> None:
        super().__init__(root)
        self.n_subjects = n_subjects
        self.rest_s = rest_s
        self.task_s = task_s
        self.recovery_s = recovery_s
        self.task_delta_bpm = task_delta_bpm
        self.overrides = overrides

    def is_available(self) -> bool:
        return True

    def recordings(self, limit: Optional[int] = None) -> List[Recording]:
        tones = [
            np.array([0.78, 0.62, 0.55]),
            np.array([0.62, 0.46, 0.40]),
            np.array([0.45, 0.32, 0.27]),
            np.array([0.30, 0.21, 0.17]),
        ]
        n = self.n_subjects if limit is None else min(limit, self.n_subjects)
        duration = self.rest_s + self.task_s + self.recovery_s
        event = (self.rest_s, self.rest_s + self.task_s, self.task_delta_bpm)
        out = []
        for i in range(n):
            # Resting HR spread deliberately exceeds the task response.
            resting = 58.0 + 6.0 * i
            cfg = SyntheticConfig(
                duration_s=duration,
                hr_bpm=resting,
                hr_event=event,
                skin_tone=tones[i % len(tones)],
                seed=100 + i,
                **self.overrides,
            )
            rec = generate(cfg, subject_id="prot{:02d}".format(i))
            rec.labels.update(
                {
                    "rest_s": self.rest_s,
                    "task_s": self.task_s,
                    "recovery_s": self.recovery_s,
                    "task_delta_bpm": self.task_delta_bpm,
                }
            )
            out.append(rec)
        return out


def window_condition(window, labels) -> Optional[str]:
    """Label a window rest / task / recovery by where its centre falls.

    Windows straddling a boundary are returned as None and excluded from
    evaluation: a 20 s window spanning the task onset is genuinely neither
    condition, and forcing it into one of them would blur the comparison.
    """
    rest_s = float(labels.get("rest_s", 0.0))
    task_s = float(labels.get("task_s", 0.0))
    centre = (window.start_s + window.end_s) / 2.0
    half = (window.end_s - window.start_s) / 2.0
    task_start, task_end = rest_s, rest_s + task_s
    if window.end_s <= task_start:
        return "rest"
    if window.start_s >= task_start + half and window.end_s <= task_end:
        return "task"
    if window.start_s >= task_end + half:
        return "recovery"
    return None
