"""Cardiac features derived from an estimated BVP window.

Naming is deliberate: everything derived from inter-beat intervals at video
frame rate is a *proxy*, never clinical HRV. At 30 fps the IBI grid is
quantised to 33 ms while resting RMSSD is 20-50 ms, so the quantisation step is
the same order as the quantity being measured (docs/limitations.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from ..rppg.signal import (
    find_peaks_ibi,
    hr_from_psd,
    hr_stability,
    snr_db,
    welch_psd,
)


@dataclass
class CardiacFeatures:
    hr_bpm: Optional[float] = None
    pulse_snr_db: Optional[float] = None
    # Spread of HR across sub-windows. Low = genuinely periodic. This, not
    # SNR, is what distinguishes a pulse from filtered noise.
    hr_stability_bpm: Optional[float] = None
    pulse_amplitude: Optional[float] = None
    ibi_mean_s: Optional[float] = None
    # PROXY ONLY -- see module docstring. Not clinical RMSSD.
    rmssd_proxy_ms: Optional[float] = None
    n_beats: int = 0

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def extract(bvp: np.ndarray, fs: float) -> CardiacFeatures:
    """Compute cardiac features from a band-passed BVP window."""
    bvp = np.asarray(bvp, dtype=np.float64)
    if bvp.size < int(fs * 4):
        return CardiacFeatures()

    freqs, psd = welch_psd(bvp, fs)
    hr = hr_from_psd(freqs, psd)
    snr = snr_db(bvp, fs, hr_bpm=hr)

    ibis = find_peaks_ibi(bvp, fs)
    ibi_mean = float(ibis.mean()) if ibis.size else None
    rmssd = None
    if ibis.size >= 3:
        diffs = np.diff(ibis)
        rmssd = float(np.sqrt(np.mean(diffs ** 2)) * 1000.0)

    # Robust peak-to-trough amplitude of the normalised waveform.
    amp = float(np.percentile(bvp, 95) - np.percentile(bvp, 5))

    return CardiacFeatures(
        hr_bpm=hr,
        pulse_snr_db=snr,
        hr_stability_bpm=hr_stability(bvp, fs),
        pulse_amplitude=amp,
        ibi_mean_s=ibi_mean,
        rmssd_proxy_ms=rmssd,
        n_beats=int(ibis.size + 1) if ibis.size else 0,
    )
