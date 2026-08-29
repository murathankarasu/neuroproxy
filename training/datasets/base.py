"""Canonical recording/dataset interface.

Every dataset -- synthetic, UBFC-rPPG, UBFC-Phys, PURE -- is adapted to the
same `Recording` so the benchmark harness never learns dataset-specific
layout. This mirrors the canonical_window contract in design doc section 6.1.

Datasets are declared, not required: a dataset whose root is missing reports
`is_available() == False` and the harness skips it with a note. That keeps the
repo runnable before any download has been arranged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np


@dataclass
class Recording:
    """One subject-session: a frame source plus optional ground truth."""

    subject_id: str
    fps: float
    n_frames: int
    frame_source: Callable[[], Iterator[np.ndarray]]
    # Contact-sensor BVP, sampled at gt_bvp_fps (often != video fps).
    gt_bvp: Optional[np.ndarray] = None
    gt_bvp_fps: Optional[float] = None
    # Session-level labels, e.g. {"task": "arithmetic", "anxiety": 4.0}.
    labels: Dict[str, object] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def frames(self) -> Iterator[np.ndarray]:
        """Yield RGB uint8 frames of shape (H, W, 3)."""
        return self.frame_source()

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0

    @property
    def has_gt(self) -> bool:
        return self.gt_bvp is not None and self.gt_bvp_fps is not None


class Dataset:
    """A named collection of recordings rooted at a directory."""

    name: str = "base"

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else None

    def is_available(self) -> bool:
        return self.root is not None and self.root.exists()

    def unavailable_reason(self) -> str:
        if self.root is None:
            return "no root configured for dataset {!r}".format(self.name)
        return "dataset root not found: {}".format(self.root)

    def recordings(self, limit: Optional[int] = None) -> List[Recording]:
        raise NotImplementedError


_REGISTRY: Dict[str, type] = {}


def register(cls: type) -> type:
    _REGISTRY[cls.name] = cls
    return cls


def get_dataset(name: str, root: Optional[Path] = None, **kwargs) -> Dataset:
    if name not in _REGISTRY:
        raise KeyError(
            "unknown dataset {!r}; available: {}".format(
                name, ", ".join(sorted(_REGISTRY))
            )
        )
    return _REGISTRY[name](root=root, **kwargs)


def available_datasets() -> List[str]:
    return sorted(_REGISTRY)


def resample_to(x: np.ndarray, src_fps: float, dst_fps: float, n_out: int) -> np.ndarray:
    """Linear resample a ground-truth trace onto the video's time base.

    Ground-truth BVP is typically 64 Hz (UBFC) or 1 kHz while video is 30 fps;
    aligning by index instead of timestamp is a classic source of silent error.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or n_out <= 0:
        return np.zeros(max(n_out, 0))
    src_t = np.arange(x.size) / src_fps
    dst_t = np.arange(n_out) / dst_fps
    return np.interp(dst_t, src_t, x)
