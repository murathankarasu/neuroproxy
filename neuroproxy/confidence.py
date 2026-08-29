"""Confidence scoring (design doc v1.0 section 9.1).

The product claim this backs is "confidence-first design: abstain when the
signal is bad" -- listed as a core differentiator in the explainer document
(section 17). A claim like that is only worth making if confidence provably
predicts error, which is what `training/evaluation/calibration.py` measures.

Until that evidence exists this is explicitly a *heuristic*. The design doc
plans to replace it with a learned confidence head; the scaffolding here keeps
the interface stable so that swap changes nothing downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# Weights from design doc section 9.1.
# STRUCTURAL DEVIATION FROM DESIGN DOC 9.1.
#
# The design doc combines every term in one weighted sum. Measured consequence:
# a well-lit, still, uncompressed recording of a face containing *no pulse at
# all* scored 0.637 and answered 100% of its windows with a mean error of 25.2
# bpm. Four of the five terms describe the image, so image quality outvoted the
# complete absence of a signal.
#
# Image quality and pulse quality are not interchangeable. Good lighting cannot
# compensate for an absent pulse, so they gate multiplicatively rather than
# voting additively: image quality is necessary but not sufficient.
#
#     confidence = image_quality * pulse_quality
#
# `compression` joins the image terms: it is the largest measured error source
# in the pipeline (docs/limitations.md section 3) and the original formula had
# no term that could see it.
IMAGE_WEIGHTS = {
    "face": 0.34,
    "lighting": 0.20,
    "motion": 0.20,
    "compression": 0.26,
}

# Retained for reference; superseded by IMAGE_WEIGHTS plus the pulse gate.
WEIGHTS = {
    "face": 0.30,
    "lighting": 0.20,
    "motion": 0.20,
    "pulse": 0.20,
    "model": 0.10,
}

# HR spread across sub-windows, in bpm, at which the pulse gate scores ~0.37.
# Legitimate HR drift within a 20 s window is a few bpm; noise gives 10-40.
STABILITY_SCALE_BPM = 6.0

# Logistic mapping from pulse SNR (dB) to [0, 1]. A window at SNR_MID scores
# 0.5. These are engineering defaults, not fitted values.
SNR_MID_DB = 3.0
SNR_SCALE_DB = 3.0

# Below this confidence the engine emits no state (design doc: "model susuyor").
#
# DERIVED, NOT CHOSEN. `neuroproxy.cli threshold` sweeps this against measured
# error. On SCAMPS (110 windows, 10 rendered subjects) every window whose HR
# error exceeded 5 bpm had confidence at or below 0.164, so 0.20 sits just
# above the observed cliff:
#
#     threshold   coverage   MAE    max error
#     0.00        100%       10.15  99.07
#     0.15         70%        1.64  49.37
#     0.20         66%        0.35   1.35
#     0.45         42%        0.35   1.35   <- the previous hand-picked value
#
# The old 0.45 bought no accuracy over 0.20 and cost 24 points of coverage.
#
# THIS VALUE IS FITTED TO SCAMPS, which is rendered, not recorded. Re-derive it
# with `neuroproxy.cli threshold --dataset ubfc_rppg` before it ships.
ABSTAIN_BELOW = 0.20


def snr_to_quality(snr_db: Optional[float]) -> float:
    """Squash pulse SNR in dB to a [0, 1] quality term."""
    if snr_db is None or not np.isfinite(snr_db):
        return 0.0
    return float(1.0 / (1.0 + np.exp(-(snr_db - SNR_MID_DB) / SNR_SCALE_DB)))


def pulse_quality(
    snr_db: Optional[float], hr_stability_bpm: Optional[float]
) -> float:
    """Evidence that a genuine cardiac pulse is present in this window.

    Both conditions must hold, so the terms multiply: the signal must rise
    above the noise floor (SNR) *and* be periodic enough that independent
    sub-windows agree on the rate (stability).

    A missing stability value means a sub-window produced no peak at all, which
    is itself evidence of absence -- it scores zero rather than being ignored.
    """
    snr_term = snr_to_quality(snr_db)
    if hr_stability_bpm is None or not np.isfinite(hr_stability_bpm):
        return 0.0
    stab_term = float(np.exp(-((hr_stability_bpm / STABILITY_SCALE_BPM) ** 2)))
    return float(np.clip(snr_term * stab_term, 0.0, 1.0))


@dataclass
class Confidence:
    value: float
    terms: Dict[str, float]
    abstain: bool

    def as_dict(self) -> Dict[str, object]:
        return {"value": self.value, "abstain": self.abstain, "terms": dict(self.terms)}


def score(quality, model_score: Optional[float] = None) -> Confidence:
    """Combine window quality and pulse SNR into a single confidence.

    `quality` is a `neuroproxy.quality.metrics.WindowQuality`. `model_score`
    is the learned calibration term from design doc section 9.1.

    When no model exists (`model_score is None`, the current state of the
    project) the model term is omitted entirely rather than substituted with a
    constant: a constant cannot contribute ordering information, it only
    compresses the dynamic range.

    See the module docstring for why image and pulse quality multiply instead
    of being averaged together.
    """
    image_terms = {
        "face": float(np.clip(quality.face, 0.0, 1.0)),
        "lighting": float(np.clip(quality.lighting, 0.0, 1.0)),
        "motion": float(np.clip(quality.motion, 0.0, 1.0)),
        "compression": float(np.clip(quality.compression, 0.0, 1.0)),
    }
    image = sum(IMAGE_WEIGHTS[k] * v for k, v in image_terms.items())
    # Frame dropouts and an unstable capture rate scale image quality: they
    # invalidate the constant-rate assumption every filter above depends on.
    image *= float(np.clip(quality.valid_frame_ratio, 0.0, 1.0))
    image *= float(np.clip(0.5 + 0.5 * quality.fps_stability, 0.0, 1.0))

    pulse = pulse_quality(quality.pulse_snr_db, quality.hr_stability_bpm)

    value = float(np.clip(image * pulse, 0.0, 1.0))
    terms = dict(image_terms)
    terms["image"] = float(np.clip(image, 0.0, 1.0))
    terms["pulse"] = pulse
    if model_score is not None:
        terms["model"] = float(np.clip(model_score, 0.0, 1.0))
        value = float(np.clip(value * (0.5 + 0.5 * terms["model"]), 0.0, 1.0))
    return Confidence(value=value, terms=terms, abstain=value < ABSTAIN_BELOW)
