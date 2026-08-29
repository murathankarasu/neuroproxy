"""Common interface for rPPG extractors.

Every method maps an RGB trace -- one mean colour triple per frame, taken over
a skin ROI -- to a 1-D BVP estimate at the same sampling rate. Keeping this
contract narrow is what lets us swap POS for a neural model later without
touching the pipeline or the harness (design doc section 10).
"""
from __future__ import annotations

from typing import Callable, Dict, Type

import numpy as np


class RPPGMethod:
    """Base class for an rPPG extractor.

    Two kinds exist, distinguished by `needs_frames`:

    * **trace methods** (POS, CHROM, green) consume one mean RGB triple per
      frame. Cheap, no learned parameters.
    * **frame methods** (EfficientPhys, TS-CAN) consume the face crops
      themselves, because they learn their own spatial weighting.

    The distinction lives here rather than in the pipeline so that adding a
    neural model does not require the pipeline to know which models are neural.
    """

    name: str = "base"
    # When True, __call__ receives (T, S, S, 3) uint8 face crops instead of an
    # (T, 3) colour trace.
    needs_frames: bool = False

    def __call__(self, rgb: np.ndarray, fs: float) -> np.ndarray:
        """Map an (T, 3) RGB trace to a (T,) BVP estimate.

        The returned signal is zero-mean and unit-variance; absolute amplitude
        of an rPPG estimate is not physically meaningful.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<{} {}>".format(type(self).__name__, self.name)


_REGISTRY: Dict[str, Type[RPPGMethod]] = {}


def register(cls: Type[RPPGMethod]) -> Type[RPPGMethod]:
    """Class decorator adding a method to the CLI-visible registry."""
    _REGISTRY[cls.name] = cls
    return cls


def get_method(name: str) -> RPPGMethod:
    if name not in _REGISTRY:
        raise KeyError(
            "unknown rPPG method {!r}; available: {}".format(
                name, ", ".join(sorted(_REGISTRY))
            )
        )
    return _REGISTRY[name]()


def available_methods() -> list:
    return sorted(_REGISTRY)


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("expected an (T, 3) RGB trace, got shape {}".format(rgb.shape))
    return rgb
