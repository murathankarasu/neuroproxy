"""Signal-layer properties. These guard the numbers every later claim rests on."""
from __future__ import annotations

import numpy as np
import pytest

from neuroproxy.rppg.signal import (
    bandpass,
    detrend,
    find_peaks_ibi,
    hr_from_psd,
    snr_db,
    welch_psd,
)

FS = 30.0


def _sine(hr_bpm: float, seconds: float = 20.0, fs: float = FS, noise: float = 0.0):
    t = np.arange(int(seconds * fs)) / fs
    rng = np.random.default_rng(0)
    return np.sin(2 * np.pi * (hr_bpm / 60.0) * t) + noise * rng.normal(size=t.size)


@pytest.mark.parametrize("hr", [48.0, 62.0, 75.0, 101.0, 140.0])
def test_hr_recovered_within_half_bpm(hr):
    """Peak interpolation must beat the raw PSD bin spacing (~5 bpm)."""
    freqs, psd = welch_psd(_sine(hr), FS)
    assert hr_from_psd(freqs, psd) == pytest.approx(hr, abs=0.5)


def test_hr_returns_none_on_flat_signal():
    """A dead signal must produce no answer rather than a fabricated one."""
    freqs, psd = welch_psd(np.zeros(600), FS)
    assert hr_from_psd(freqs, psd) is None


def test_detrend_removes_illumination_drift():
    """A large sub-band ramp must not survive into the HR band."""
    t = np.arange(600) / FS
    drift = 50.0 * np.sin(2 * np.pi * 0.05 * t)
    clean = _sine(72.0)
    out = bandpass(detrend(clean + drift, FS), FS)
    freqs, psd = welch_psd(out, FS)
    assert hr_from_psd(freqs, psd) == pytest.approx(72.0, abs=1.0)


def test_snr_orders_clean_above_noisy():
    clean = snr_db(bandpass(_sine(72.0, noise=0.0), FS), FS)
    noisy = snr_db(bandpass(_sine(72.0, noise=3.0), FS), FS)
    assert clean is not None and noisy is not None
    assert clean > noisy


def test_ibi_quantisation_is_visible():
    """Documents the 33 ms IBI floor at 30 fps that makes RMSSD a proxy only.

    If this ever fails because the grid got finer, the PRV caveat in
    docs/limitations.md should be revisited -- not silently dropped.
    """
    ibis = find_peaks_ibi(bandpass(_sine(72.0), FS), FS)
    assert ibis.size > 5
    steps = np.unique(np.round(ibis * FS).astype(int))
    assert steps.size <= 3  # IBIs land on a coarse integer-frame grid
