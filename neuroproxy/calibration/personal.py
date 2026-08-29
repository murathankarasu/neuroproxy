"""Personal baseline: express physiology relative to the person, not absolutely.

Resting HR of 65 in one person and 90 in another are both normal, so an
absolute threshold measures identity as much as state. Both source documents
make this a first-class requirement (design doc section 3 step 6; explainer
sections 8 and 17), and it is what makes the output contract honest: the engine
reports deviation from the subject's own calm baseline, not a number on a
universal scale it has no way to define.

Robust statistics throughout. The calibration period is short (30-60 s) and a
single swallow, cough or head turn is enough to wreck a mean/standard-deviation
baseline that then distorts the entire session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Features normalised against the person. Deliberately physiology only.
#
# Quality and confidence values are NOT baselined: they measure how well the
# camera is working, not who the subject is. Normalising them against a
# personal reference would rescale "this signal is bad" into "this signal is
# normal for them", which is precisely backwards.
BASELINE_FEATURES = (
    "hr_bpm",
    "ibi_mean_s",
    "pulse_amplitude",
)

# Consistency factor making MAD comparable to a standard deviation for
# normally distributed data.
MAD_TO_SIGMA = 1.4826

# Minimum usable calibration windows. Below this the baseline is refused
# rather than fitted on almost nothing.
MIN_CALIBRATION_WINDOWS = 3

# Scale floor, as a fraction of the location estimate. A feature that barely
# moved during calibration would otherwise produce enormous z-scores for
# ordinary later variation.
MIN_SCALE_FRACTION = 0.02

# Drift beyond this many baseline sigmas means the reference is stale.
DRIFT_SIGMA = 3.0

# See PersonalBaseline.transform for the measurement behind this default.
DEFAULT_MODE = "delta"


@dataclass
class FeatureBaseline:
    location: float          # median during calibration
    scale: float             # MAD-derived sigma, floored
    n: int
    scale_was_floored: bool = False


@dataclass
class PersonalBaseline:
    """Per-subject reference statistics and the transform they define."""

    features: Dict[str, FeatureBaseline] = field(default_factory=dict)
    n_windows: int = 0
    calibration_seconds: float = 0.0
    mean_confidence: float = 0.0
    ready: bool = False
    reason: Optional[str] = None

    def transform(
        self, values: Dict[str, Optional[float]], mode: str = DEFAULT_MODE
    ) -> Dict[str, Optional[float]]:
        """Express features relative to this subject's baseline.

        mode="delta" (default) subtracts the baseline location and keeps native
        units. mode="z" additionally divides by the baseline scale.

        DEFAULT IS "delta" ON MEASURED GROUNDS. Dividing by a scale estimated
        from a 45 s window made subject-independent task-vs-rest separability
        *worse* at every effect size tested:

            task response   raw     z-score   delta   within-subject ceiling
            12 bpm          0.700   0.873     0.907   1.000
             6 bpm          0.609   0.700     0.754   0.873
             3 bpm          0.562   0.606     0.647   0.752
             2 bpm          0.546   0.576     0.607   0.701

        The cause is visible in the baselines themselves: across subjects whose
        true HR variability was identical by construction, the MAD-derived
        scale ranged from 1.25 to 4.38 bpm. A 45 s window cannot estimate HR
        variability to better than about 3.5x, so dividing by that estimate
        injects between-subject noise instead of removing it.

        Caveat: the synthetic cohort gives every subject the same true
        variability, so scale normalisation there could only ever hurt. Real
        people differ in HR variability, and scale normalisation may pay for
        itself given a longer calibration. Re-run this ablation on real data
        before treating "delta" as settled.

        Features without a baseline, or with a missing value, map to None
        rather than 0.0 -- "no information" and "exactly at baseline" are
        different statements and must not be conflated downstream.
        """
        if mode not in ("delta", "z"):
            raise ValueError("mode must be 'delta' or 'z', got {!r}".format(mode))
        out: Dict[str, Optional[float]] = {}
        for name in BASELINE_FEATURES:
            base = self.features.get(name)
            val = values.get(name)
            if base is None or val is None or not np.isfinite(val):
                out[name] = None
                continue
            delta = float(val - base.location)
            out[name] = delta / base.scale if mode == "z" else delta
        return out

    def drifted(self, recent: Sequence[Dict[str, Optional[float]]]) -> List[str]:
        """Features whose recent median has moved off the baseline.

        Drift does not mean the subject changed state -- a state change is the
        signal we are looking for. It means the *reference* may no longer
        describe their calm condition, e.g. the lighting or posture changed
        after calibration. Callers should surface it, not silently recalibrate.
        """
        flagged = []
        for name in BASELINE_FEATURES:
            base = self.features.get(name)
            if base is None:
                continue
            vals = [
                r.get(name) for r in recent
                if r.get(name) is not None and np.isfinite(r.get(name))
            ]
            if len(vals) < MIN_CALIBRATION_WINDOWS:
                continue
            z = (float(np.median(vals)) - base.location) / base.scale
            if abs(z) > DRIFT_SIGMA:
                flagged.append(name)
        return flagged


def fit(
    windows: Sequence,
    calibration_seconds: float = 45.0,
    min_confidence: float = 0.45,
) -> PersonalBaseline:
    """Fit a baseline from the calm opening of a session.

    Only windows the engine was willing to answer are used. Calibrating on
    low-confidence windows would anchor the whole session to noise, and every
    later z-score would inherit that error -- a failure mode that is invisible
    at inference time because the numbers still look reasonable.
    """
    usable = [
        w for w in windows
        if w.end_s <= calibration_seconds
        and getattr(w, "valid", False)
        and getattr(w, "confidence", 0.0) >= min_confidence
    ]

    bl = PersonalBaseline(
        n_windows=len(usable), calibration_seconds=calibration_seconds
    )
    if len(usable) < MIN_CALIBRATION_WINDOWS:
        bl.reason = (
            "only {} usable calibration windows in the first {:.0f}s "
            "(need {})".format(len(usable), calibration_seconds, MIN_CALIBRATION_WINDOWS)
        )
        return bl

    bl.mean_confidence = float(np.mean([w.confidence for w in usable]))

    for name in BASELINE_FEATURES:
        vals = []
        for w in usable:
            v = getattr(w.features, name, None)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        if len(vals) < MIN_CALIBRATION_WINDOWS:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        location = float(np.median(arr))
        mad = float(np.median(np.abs(arr - location)))
        scale = MAD_TO_SIGMA * mad
        floor = MIN_SCALE_FRACTION * abs(location)
        floored = scale < floor
        scale = max(scale, floor, 1e-9)
        bl.features[name] = FeatureBaseline(
            location=location, scale=scale, n=len(arr), scale_was_floored=floored
        )

    bl.ready = bool(bl.features)
    if not bl.ready:
        bl.reason = "no feature had enough finite values during calibration"
    return bl


def feature_dict(features) -> Dict[str, Optional[float]]:
    """Pull the baselined features out of a CardiacFeatures record."""
    return {name: getattr(features, name, None) for name in BASELINE_FEATURES}
