"""Pre-session readiness check: predict failure before the session runs.

THE PROBLEM THIS SOLVES
-----------------------
On real recordings the engine produces no usable pulse for roughly half of
sessions (docs/limitations.md section 15). For a research customer sending a
link to 100 participants, that is 48 people who complete a study and contribute
nothing -- and nobody finds out until afterwards.

Almost all of that is predictable in the first second. Of every quality
dimension measured, exactly one separates sessions that work from sessions that
do not:

    metric        usable   unusable   Mann-Whitney p
    skin fraction  0.845     0.643     0.004   <-- the only one
    face conf      0.850     0.850     0.32
    lighting       0.997     0.994     0.37
    sharpness      0.970     0.971     0.90
    motion         0.824     0.828     0.67
    compression    0.181     0.174     0.26

and it predicts the outcome continuously, not just categorically:
`spearman(skin_fraction, HR error) = -0.404 (p = 0.004)`,
`spearman(skin_fraction, coverage) = +0.365 (p = 0.009)`.

    skin fraction   n    usable    median HR error
    0.27 - 0.71     17     18%     15.65 bpm
    0.71 - 0.86     16     69%      0.81 bpm
    0.86 - 0.98     17     71%      0.82 bpm

The cliff sits near 0.70. This check measures the same number from a few
seconds of preview and reports it before the study begins.

WHAT IT DOES NOT DO
-------------------
It does not explain *why* a subject's skin fraction is low. Facial hair,
hairline and bright specular regions all reduce it, and the chroma-distance
mask that produces it is still brightness-dependent in ways that are not fully
characterised. The check is predictive, not diagnostic, and its advice is
correspondingly general.

Thresholds are fitted to 25 MCD-rPPG subjects. Re-derive them on any new
population before relying on the wording shown to participants.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from .quality import metrics as qm
from .vision.detector import FaceDetector, adaptive_skin_mask
from .vision.roi import extract as roi_extract

# Below this, only 18% of real sessions produced a usable pulse.
SKIN_FRACTION_POOR = 0.70
# Above this, 71% did. Between the two is the uncertain band.
SKIN_FRACTION_GOOD = 0.86

# Frames to sample for the check. Two seconds at 30 fps is enough for a stable
# median and short enough to run while the participant reads the consent text.
DEFAULT_SAMPLE_FRAMES = 60

MIN_FACE_FRAMES = 0.5   # fraction of sampled frames that must contain a face


@dataclass
class Readiness:
    """Verdict plus the numbers behind it."""

    ready: bool
    verdict: str                       # "good" | "marginal" | "poor" | "no_face"
    skin_fraction: Optional[float] = None
    face_fraction: float = 0.0
    lighting: Optional[float] = None
    compression: Optional[float] = None
    expected_usable: Optional[float] = None   # from the fitted bands
    messages: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def assess(
    frames: Iterable[np.ndarray],
    detector: Optional[FaceDetector] = None,
    max_frames: int = DEFAULT_SAMPLE_FRAMES,
) -> Readiness:
    """Judge whether a session is worth running, from a short preview."""
    det = detector or FaceDetector("auto")
    skins, lights, comps = [], [], []
    n_seen = n_face = 0

    for i, frame in enumerate(frames):
        if i >= max_frames:
            break
        n_seen += 1
        face = det.detect(frame)
        if face is None:
            continue
        n_face += 1
        mask = adaptive_skin_mask(frame, face)
        sample = roi_extract(frame, face, mask=mask)
        if sample is not None:
            skins.append(sample.skin_fraction)
        light, _ = qm.lighting_score(frame, face)
        lights.append(light)
        if i % 15 == 0:
            comps.append(qm.compression_score(frame))

    face_fraction = (n_face / n_seen) if n_seen else 0.0
    r = Readiness(ready=False, verdict="no_face", face_fraction=face_fraction)

    if n_seen == 0:
        r.messages.append("No frames received from the camera.")
        return r
    if face_fraction < MIN_FACE_FRAMES or not skins:
        r.messages.append(
            "No face detected in most frames. Move into the centre of the "
            "frame and face the camera."
        )
        return r

    r.skin_fraction = float(np.median(skins))
    r.lighting = float(np.median(lights)) if lights else None
    r.compression = float(np.median(comps)) if comps else None

    if r.skin_fraction >= SKIN_FRACTION_GOOD:
        r.verdict, r.ready, r.expected_usable = "good", True, 0.71
    elif r.skin_fraction >= SKIN_FRACTION_POOR:
        r.verdict, r.ready, r.expected_usable = "marginal", True, 0.69
    else:
        r.verdict, r.ready, r.expected_usable = "poor", False, 0.18
        r.messages.append(
            "Too little clear skin is visible on the forehead and cheeks. Try "
            "moving hair off the forehead, removing glasses, and facing a soft "
            "even light rather than a bright one to the side."
        )

    if r.lighting is not None and r.lighting < 0.6:
        r.messages.append("Lighting is poor. Face a window or a lamp.")
    if r.compression is not None and r.compression < 0.15:
        r.messages.append(
            "The video stream is heavily compressed, which weakens the pulse "
            "signal. Close other video calls and prefer a wired connection."
        )
    if not r.messages:
        r.messages.append("Ready to start.")
    return r
