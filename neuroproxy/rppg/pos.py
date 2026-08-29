"""POS: plane-orthogonal-to-skin rPPG (Wang et al., 2017).

Projects temporally normalised RGB onto a plane orthogonal to the skin-tone
direction, then tunes the combination of the two remaining axes so the
intensity/motion component cancels. Overlap-added over short windows.

This is the reference method for the whole project: it has no learned
parameters, so a failure here is a capture or ROI failure, never a model
failure. Design doc section 2.1.
"""
from __future__ import annotations

import numpy as np

from .base import RPPGMethod, _validate_rgb, register
from .signal import bandpass, normalize

WINDOW_SECONDS = 1.6

# Projection onto the plane orthogonal to the (1,1,1) skin-tone direction.
_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


@register
class POS(RPPGMethod):
    name = "pos"

    def __call__(self, rgb: np.ndarray, fs: float) -> np.ndarray:
        rgb = _validate_rgb(rgb)
        n = rgb.shape[0]
        win = int(fs * WINDOW_SECONDS)
        if n < 16 or win < 4:
            return np.zeros(n)
        out = np.zeros(n)
        for start in range(0, n - win + 1):
            seg = rgb[start : start + win]
            mu = seg.mean(axis=0)
            if np.any(mu <= 1e-9):
                continue
            cn = (seg / mu).T                       # (3, win)
            s = _PROJECTION @ cn                    # (2, win)
            s1_std = s[1].std()
            alpha = (s[0].std() / s1_std) if s1_std > 1e-12 else 0.0
            h = s[0] + alpha * s[1]
            out[start : start + win] += h - h.mean()  # overlap-add
        return normalize(bandpass(out, fs))
