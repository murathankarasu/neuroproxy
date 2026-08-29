"""CHROM: chrominance-based rPPG (de Haan & Jeanne, 2013).

Projects normalised RGB onto two chrominance axes chosen so that the specular
(motion) component largely cancels, then combines them with a ratio that makes
the result robust to skin tone.
"""
from __future__ import annotations

import numpy as np

from .base import RPPGMethod, _validate_rgb, register
from .signal import bandpass, normalize

WINDOW_SECONDS = 1.6


@register
class Chrom(RPPGMethod):
    name = "chrom"

    def __call__(self, rgb: np.ndarray, fs: float) -> np.ndarray:
        rgb = _validate_rgb(rgb)
        n = rgb.shape[0]
        win = int(fs * WINDOW_SECONDS)
        if n < 16 or win < 4:
            return np.zeros(n)
        stride = max(win // 2, 1)
        out = np.zeros(n)
        weights = np.zeros(n)
        hann = np.hanning(win)
        for start in range(0, n - win + 1, stride):
            seg = rgb[start : start + win]
            mu = seg.mean(axis=0)
            if np.any(mu <= 1e-9):
                continue
            cn = seg / mu  # per-window temporal normalisation
            x = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
            y = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
            xf = bandpass(x, fs)
            yf = bandpass(y, fs)
            sy = yf.std()
            alpha = (xf.std() / sy) if sy > 1e-12 else 0.0
            s = xf - alpha * yf
            out[start : start + win] += hann * (s - s.mean())
            weights[start : start + win] += hann
        valid = weights > 1e-9
        out[valid] /= weights[valid]
        return normalize(out)
