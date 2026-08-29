"""rPPG extraction methods and the signal primitives they share."""
from __future__ import annotations

from .base import RPPGMethod, available_methods, get_method, register
from .chrom import Chrom
from .green import Green
from .pos import POS

# Neural methods need torch, which is an optional extra. Their absence must not
# break the classical path.
try:  # pragma: no cover - depends on optional dependency
    from .neural import EfficientPhysPURE, EfficientPhysUBFC
except Exception:  # torch missing, or checkpoints absent
    EfficientPhysPURE = EfficientPhysUBFC = None

__all__ = [
    "RPPGMethod",
    "available_methods",
    "get_method",
    "register",
    "POS",
    "Chrom",
    "Green",
    "EfficientPhysPURE",
    "EfficientPhysUBFC",
]
