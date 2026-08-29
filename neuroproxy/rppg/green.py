"""Green-channel baseline.

This is the weakest defensible method and exists as a floor: any method that
does not beat plain green on a given dataset is not earning its complexity.
"""
from __future__ import annotations

import numpy as np

from .base import RPPGMethod, _validate_rgb, register
from .signal import bandpass, detrend, normalize


@register
class Green(RPPGMethod):
    name = "green"

    def __call__(self, rgb: np.ndarray, fs: float) -> np.ndarray:
        rgb = _validate_rgb(rgb)
        g = rgb[:, 1]
        return normalize(bandpass(detrend(g, fs), fs))
