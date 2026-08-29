"""Signal processing primitives shared by every rPPG method.

All functions operate on 1-D float arrays sampled at a known, constant rate.
Resampling to a constant rate is the caller's job (see `pipeline.offline`),
because dropped webcam frames are a quality signal we do not want to hide.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import signal as sps

# Physiological plausibility band for adult heart rate, in Hz.
# 0.7 Hz = 42 bpm, 3.0 Hz = 180 bpm. Anything outside is rejected, not clamped.
HR_BAND_HZ: Tuple[float, float] = (0.7, 3.0)

# Makes a median absolute deviation comparable to a standard deviation for
# normally distributed data.
MAD_TO_SIGMA = 1.4826


def detrend(x: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """Remove slow illumination drift with a high-pass Butterworth filter.

    Drift from auto-exposure and ambient light lives well below the HR band and
    otherwise dominates the variance of any raw RGB trace.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        return x - x.mean()
    nyq = fs / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = sps.butter(2, wn, btype="highpass")
    return sps.filtfilt(b, a, x, method="gust")


def bandpass(
    x: np.ndarray,
    fs: float,
    band: Tuple[float, float] = HR_BAND_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase band-pass restricted to the plausible HR band."""
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.99)
    if x.size < 3 * order or lo >= hi:
        return x - x.mean()
    b, a = sps.butter(order, [lo, hi], btype="bandpass")
    return sps.filtfilt(b, a, x, method="gust")


def normalize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance. Scale of a BVP estimate carries no meaning."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def welch_psd(
    x: np.ndarray,
    fs: float,
    nfft_seconds: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD with a segment long enough to resolve the HR band.

    A 12 s segment gives ~0.083 Hz raw bin spacing (5 bpm) before zero-padding;
    `hr_from_psd` recovers sub-bin resolution by interpolating the peak.
    """
    x = np.asarray(x, dtype=np.float64)
    nperseg = int(min(len(x), max(fs * nfft_seconds, 64)))
    if nperseg < 16:
        return np.zeros(0), np.zeros(0)
    # Zero-pad by 8x so the peak interpolation has a well-sampled lobe to work on.
    nfft = int(2 ** np.ceil(np.log2(nperseg * 8)))
    freqs, psd = sps.welch(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft, detrend="constant"
    )
    return freqs, psd


def hr_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: Tuple[float, float] = HR_BAND_HZ,
) -> Optional[float]:
    """Peak-frequency HR in bpm, with parabolic interpolation of the peak bin.

    Returns None when the band is empty or the spectrum is degenerate, so that
    callers emit an invalid window rather than a fabricated number.
    """
    if freqs.size == 0:
        return None
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return None
    band_freqs, band_psd = freqs[mask], psd[mask]
    if not np.isfinite(band_psd).all() or band_psd.max() <= 0:
        return None
    k = int(np.argmax(band_psd))
    f_peak = float(band_freqs[k])
    # Parabolic interpolation on the log spectrum around the peak bin.
    if 0 < k < len(band_psd) - 1:
        y0, y1, y2 = np.log(band_psd[k - 1 : k + 2] + 1e-30)
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                df = float(band_freqs[1] - band_freqs[0])
                f_peak += delta * df
    return f_peak * 60.0


def snr_db(
    x: np.ndarray,
    fs: float,
    hr_bpm: Optional[float] = None,
    half_width_hz: float = 0.2,
    band: Tuple[float, float] = HR_BAND_HZ,
) -> Optional[float]:
    """Pulse SNR in dB: power near the HR peak and its 2nd harmonic vs the rest.

    This is the de Haan-style rPPG SNR. It is the single most useful runtime
    quality number we have, because it needs no ground truth.
    """
    freqs, psd = welch_psd(x, fs)
    if freqs.size == 0:
        return None
    if hr_bpm is None:
        hr_bpm = hr_from_psd(freqs, psd, band)
    if hr_bpm is None:
        return None
    f0 = hr_bpm / 60.0
    # Evaluate over the full analysis band including room for the 2nd harmonic.
    region = (freqs >= band[0]) & (freqs <= min(band[1] * 2.0, freqs[-1]))
    if not region.any():
        return None
    signal_mask = np.zeros_like(freqs, dtype=bool)
    for harmonic in (f0, 2.0 * f0):
        signal_mask |= np.abs(freqs - harmonic) <= half_width_hz
    signal_mask &= region
    noise_mask = region & ~signal_mask
    sig_p = float(psd[signal_mask].sum())
    noi_p = float(psd[noise_mask].sum())
    if sig_p <= 0 or noi_p <= 0:
        return None
    return 10.0 * np.log10(sig_p / noi_p)


def find_peaks_ibi(
    bvp: np.ndarray, fs: float, band: Tuple[float, float] = HR_BAND_HZ
) -> np.ndarray:
    """Inter-beat intervals in seconds from systolic peaks.

    NOTE: at 30 fps the IBI grid is quantised to 33 ms. Resting RMSSD is
    20-50 ms, i.e. the same order as the quantisation step, so PRV metrics
    derived from this are reported as proxies only and are not clinical HRV.
    See docs/limitations.md.
    """
    bvp = np.asarray(bvp, dtype=np.float64)
    if bvp.size < int(fs * 2):
        return np.zeros(0)
    min_distance = int(fs / band[1])  # fastest plausible beat
    peaks, _ = sps.find_peaks(bvp, distance=max(min_distance, 1))
    if peaks.size < 2:
        return np.zeros(0)
    return np.diff(peaks) / fs


def hr_stability(
    x: np.ndarray,
    fs: float,
    n_sub: int = 5,
    sub_seconds: Optional[float] = None,
    band: Tuple[float, float] = HR_BAND_HZ,
) -> Optional[float]:
    """Spread (bpm) of HR estimated independently on overlapping sub-windows.

    A real cardiac pulse is periodic, so sub-windows agree. Band-pass-filtered
    noise still produces a spectral peak -- that is what a band-pass does -- but
    the peak wanders, so sub-windows disagree.

    This exists because SNR cannot make that distinction. Measured on 20 s
    windows at 30 fps: a clean pulse gives SNR 35.7 dB and spread 0.00 bpm,
    while pure noise gives SNR -0.2 to -1.1 dB and spread 10.5 to 39.5 bpm --
    and a genuine pulse buried in 3x noise gives SNR -0.3 dB, indistinguishable
    from noise by SNR alone. The spread separates all of them.

    `sub_seconds` defaults to a fraction of the window rather than a fixed
    length. A FIXED LENGTH IS A TRAP: with the default 10 s sub-window and a
    10 s analysis window, all three sub-windows were identical, the spread was
    always exactly 0.0, and the gate silently passed everything. That went
    unnoticed because the synthetic cohort uses 20 s windows; it surfaced on
    SCAMPS, whose clips are only 20 s long and so are analysed at 10 s.

    Returns None when the window is too short to yield distinct sub-windows, or
    when a sub-window has no usable peak. Both mean "no evidence of a pulse",
    which the confidence gate treats as absence rather than as unknown.
    """
    x = np.asarray(x, dtype=np.float64)
    if sub_seconds is None:
        sub_seconds = max(4.0, 0.6 * (x.size / fs))
    win = int(sub_seconds * fs)
    if n_sub < 2 or x.size < win or win < int(fs * 4):
        return None
    step = (x.size - win) // (n_sub - 1)
    if step <= 0:
        # Sub-windows would be identical, so the spread would be a meaningless
        # zero. Refuse rather than report perfect stability.
        return None
    hrs = []
    for i in range(n_sub):
        seg = x[i * step : i * step + win]
        freqs, psd = welch_psd(seg, fs)
        hr = hr_from_psd(freqs, psd, band)
        if hr is None:
            return None
        hrs.append(hr)

    # Robust spread, not the standard deviation. A short sub-window sometimes
    # locks onto the SECOND HARMONIC instead of the fundamental -- at a low
    # heart rate the harmonic sits well inside the analysis band. One such
    # sub-window destroys a standard deviation while the majority agree
    # perfectly. Measured on SCAMPS: a subject whose sub-window estimates were
    # [55.1, 55.1, 103.2] reported a spread of 22.7 bpm and was refused on 91%
    # of its windows, while its full-window estimates were accurate to 0.5 bpm
    # throughout.
    #
    # Folding harmonics back onto the fundamental was considered and rejected:
    # it also rescues noise, whose estimates land at arbitrary ratios that
    # frequently include 2x. A median-based spread separates both cases without
    # that risk -- it reads 0 for [55.1, 55.1, 103.2] and 23 for the noise
    # triple [129.5, 106.3, 52.7].
    arr = np.asarray(hrs, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return float(MAD_TO_SIGMA * mad)
