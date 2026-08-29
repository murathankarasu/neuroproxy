"""Pretrained neural rPPG models, vendored and adapted.

Optional: importing this package requires torch. `neuroproxy.rppg` imports it
defensively so the classical methods keep working without it.
"""
from __future__ import annotations

from .adapter import EfficientPhysPURE, EfficientPhysSCAMPS, EfficientPhysUBFC

__all__ = ["EfficientPhysPURE", "EfficientPhysSCAMPS", "EfficientPhysUBFC"]
