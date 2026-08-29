"""Feature extractors that turn signals into model inputs."""
from __future__ import annotations

from .cardiac import CardiacFeatures
from .cardiac import extract as cardiac_features
from .ocular import OcularFeatures, detect_blinks
from .ocular import extract as ocular_features

__all__ = [
    "CardiacFeatures",
    "cardiac_features",
    "OcularFeatures",
    "ocular_features",
    "detect_blinks",
]
