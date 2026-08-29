"""The browser extractor must agree with the Python pipeline.

Client-side extraction is only safe because the two produce the same numbers:
every benchmark, threshold and ablation in docs/limitations.md was measured on
the Python path, and a browser session inherits none of it if the extractors
disagree.

The JavaScript cannot run in pytest, so this pins the **Python** reference
values that the browser was checked against. If any of these move, the
equivalence has to be re-verified rather than assumed -- open
`/static/equivalence.html` against a running API and compare.

Verified in a browser on 2026-08-29, four skin tones from light to dim:

    fixture   Python RGB          JavaScript RGB      difference
    light     [198, 158, 140]     [198, 158, 140]     [0, 0, 0]
    medium    [157, 117, 102]     [157, 117, 102]     [0, 0, 0]
    dark      [ 76,  54,  43]     [ 76,  54,  43]     [0, 0, 0]
    dim       [ 46,  34,  26]     [ 46,  34,  26]     [0, 0, 0]

Mean RGB -- the quantity that drives heart rate -- is exact. `lighting` differs
by up to 0.014 (dark: 0.867 vs 0.881), which reaches the output only through
the confidence weighting.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuroproxy.quality import metrics as qm
from neuroproxy.vision.detector import FaceDetector, adaptive_skin_mask
from neuroproxy.vision.roi import extract as roi_extract
from training.datasets.synthetic import SyntheticConfig, generate

# Skin tone -> the mean RGB the browser extractor produced, to the decimal.
BROWSER_VERIFIED = {
    "light": ((0.78, 0.62, 0.55), (198.0, 158.0, 140.0)),
    "medium": ((0.62, 0.46, 0.40), (157.0, 117.0, 102.0)),
    "dark": ((0.30, 0.21, 0.17), (76.0, 54.0, 43.0)),
    "dim": ((0.18, 0.13, 0.10), (46.0, 34.0, 26.0)),
}


def _extract(tone):
    rec = generate(SyntheticConfig(
        duration_s=1.0, skin_tone=np.array(tone), noise_sigma=0.0, seed=1))
    frame = next(rec.frames())
    face = FaceDetector("skin").detect(frame)
    assert face is not None
    sample = roi_extract(frame, face,
                         mask=adaptive_skin_mask(frame, face),
                         anchor_forehead=False)
    return frame, face, sample


@pytest.mark.parametrize("name", sorted(BROWSER_VERIFIED))
def test_python_matches_the_browser_verified_values(name):
    tone, expected = BROWSER_VERIFIED[name]
    _, _, sample = _extract(tone)
    assert sample is not None
    assert sample.rgb == pytest.approx(np.array(expected), abs=1e-3)


def test_dark_tones_are_not_silently_dropped():
    """The failure this whole extraction path had to be careful about.

    A fixed chroma locus excluded a dark-skinned subject entirely
    (limitations 12). Both extractors derive the mask from the face's own
    colour, so all four tones must survive -- including the dimmest.
    """
    for name, (tone, _) in BROWSER_VERIFIED.items():
        _, _, sample = _extract(tone)
        assert sample is not None, "tone {} produced no ROI".format(name)
        assert sample.skin_fraction > 0.5


def test_feature_packet_round_trips_through_the_api_shape():
    """What the browser sends must be what the engine expects."""
    from api.main import _packet

    packet = _packet({
        "rgb": [157.0, 117.0, 102.0], "valid": True, "face": 0.6,
        "lighting": 1.0, "sharpness": 0.99, "motion": 1.0,
        "skin_fraction": 1.0, "compression": 1.0,
    })
    assert packet.valid
    assert packet.rgb == [157.0, 117.0, 102.0]
    assert packet.to_quality().face == pytest.approx(0.6)


def test_malformed_packets_are_treated_as_invalid_not_crashing():
    from api.main import _packet

    for raw in ({}, {"rgb": None}, {"rgb": [1, 2]}, {"rgb": "nope", "valid": True}):
        packet = _packet(raw)
        assert not packet.valid
        assert packet.rgb is None


def test_feature_path_and_frame_path_agree():
    """Same recording through both ingest paths must give identical state."""
    from neuroproxy.inference import FramePacket, StateEngine

    rec = generate(SyntheticConfig(duration_s=40.0, hr_bpm=72.0))
    frames = list(rec.frames())

    by_frame = StateEngine(fps=rec.fps, calibration_s=0.0)
    frame_out = [s for s in by_frame.run(iter(frames))]

    # Extract with the Python pipeline, then feed the feature path.
    detector = FaceDetector("auto")
    by_feature = StateEngine(fps=rec.fps, calibration_s=0.0)
    feature_out = []
    prev_center = None
    for frame in frames:
        face = detector.detect(frame)
        if face is None:
            out = by_feature.push_features(FramePacket())
        else:
            mask = adaptive_skin_mask(frame, face)
            sample = roi_extract(frame, face, mask=mask)
            light, _ = qm.lighting_score(frame, face)
            centre = np.array([face.x + face.w / 2.0, face.y + face.h / 2.0])
            disp = 0.0 if prev_center is None else float(np.linalg.norm(centre - prev_center))
            prev_center = centre
            out = by_feature.push_features(FramePacket(
                rgb=list(sample.rgb) if sample else None,
                valid=sample is not None,
                face=face.confidence,
                lighting=light,
                sharpness=qm.sharpness_score(frame, face),
                motion=qm.motion_score(disp, face.w),
                skin_fraction=sample.skin_fraction if sample else 0.0,
            ))
        if out is not None:
            feature_out.append(out)

    assert len(frame_out) == len(feature_out) > 0
    for a, b in zip(frame_out, feature_out):
        assert a.physiology.get("heart_rate_bpm") == pytest.approx(
            b.physiology.get("heart_rate_bpm"), abs=1e-9)
