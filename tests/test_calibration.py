"""Personal baseline: refuse bad calibration, and never fabricate a reference."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from neuroproxy.calibration import BASELINE_FEATURES, feature_dict
from neuroproxy.calibration import fit as fit_baseline
from neuroproxy.calibration.personal import MIN_SCALE_FRACTION
from neuroproxy.features.cardiac import CardiacFeatures


@dataclass
class FakeWindow:
    start_s: float
    end_s: float
    features: CardiacFeatures
    valid: bool = True
    confidence: float = 0.8


def _windows(hrs, confidence=0.8, valid=True, start=0.0, step=2.0):
    return [
        FakeWindow(
            start_s=start + i * step,
            end_s=start + i * step + 20.0,
            features=CardiacFeatures(hr_bpm=hr, ibi_mean_s=60.0 / hr,
                                     pulse_amplitude=1.0),
            valid=valid,
            confidence=confidence,
        )
        for i, hr in enumerate(hrs)
    ]


def test_baseline_refuses_when_too_few_windows():
    bl = fit_baseline(_windows([70.0]), calibration_seconds=45.0)
    assert not bl.ready
    assert "usable calibration windows" in bl.reason


def test_baseline_ignores_low_confidence_windows():
    """Anchoring a session to noisy windows is invisible later but poisons everything."""
    bl = fit_baseline(_windows([70.0] * 6, confidence=0.1), calibration_seconds=45.0)
    assert not bl.ready


def test_baseline_ignores_windows_the_engine_refused():
    bl = fit_baseline(_windows([70.0] * 6, valid=False), calibration_seconds=45.0)
    assert not bl.ready


def test_baseline_uses_only_the_calibration_period():
    """Windows after the calibration cutoff must not enter the reference."""
    early = _windows([70.0] * 6, start=0.0)
    late = _windows([120.0] * 6, start=60.0)
    bl = fit_baseline(early + late, calibration_seconds=45.0)
    assert bl.ready
    assert bl.features["hr_bpm"].location == pytest.approx(70.0, abs=1.0)


def test_missing_value_maps_to_none_not_zero():
    """'No information' and 'exactly at baseline' must stay distinguishable."""
    bl = fit_baseline(_windows([68.0, 70.0, 72.0, 71.0, 69.0]), calibration_seconds=45.0)
    assert bl.ready
    out = bl.transform({"hr_bpm": None, "ibi_mean_s": None, "pulse_amplitude": None})
    assert all(v is None for v in out.values())


def test_constant_calibration_does_not_explode_scores():
    """A feature that never moved during calibration has MAD 0."""
    bl = fit_baseline(_windows([70.0] * 8), calibration_seconds=45.0)
    assert bl.ready
    base = bl.features["hr_bpm"]
    assert base.scale_was_floored
    assert base.scale >= MIN_SCALE_FRACTION * 70.0
    z = bl.transform({"hr_bpm": 76.0}, mode="z")["hr_bpm"]
    assert abs(z) < 20.0  # finite and sane, not 6/0


def test_delta_mode_keeps_native_units():
    bl = fit_baseline(_windows([68.0, 70.0, 72.0, 71.0, 69.0]), calibration_seconds=45.0)
    delta = bl.transform({"hr_bpm": 82.0}, mode="delta")["hr_bpm"]
    assert delta == pytest.approx(82.0 - bl.features["hr_bpm"].location)


def test_quality_is_never_baselined():
    """Normalising signal quality against the person would invert its meaning."""
    for banned in ("pulse_snr_db", "confidence", "signal_quality", "compression"):
        assert banned not in BASELINE_FEATURES


def test_drift_is_flagged_when_the_reference_goes_stale():
    bl = fit_baseline(_windows([68.0, 70.0, 72.0, 71.0, 69.0]), calibration_seconds=45.0)
    steady = [feature_dict(CardiacFeatures(hr_bpm=hr)) for hr in (69.0, 70.0, 71.0, 70.0)]
    shifted = [feature_dict(CardiacFeatures(hr_bpm=hr)) for hr in (95.0, 96.0, 97.0, 96.0)]
    assert "hr_bpm" not in bl.drifted(steady)
    assert "hr_bpm" in bl.drifted(shifted)


def test_invalid_mode_is_rejected():
    bl = fit_baseline(_windows([68.0, 70.0, 72.0, 71.0, 69.0]), calibration_seconds=45.0)
    with pytest.raises(ValueError):
        bl.transform({"hr_bpm": 70.0}, mode="zscore")
