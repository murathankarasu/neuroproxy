"""Adapters exposing pretrained neural rPPG models through `RPPGMethod`.

The point of running these against POS in the same harness is design doc
section 8.3's "POS vs EfficientPhys" ablation: does the neural model earn its
dependency, on *our* data rather than on its training distribution?

Checkpoints are chosen deliberately cross-dataset. Testing a PURE-trained model
on PURE would measure memorisation; testing it on MCD-rPPG measures
generalisation, which is the only property that matters for a product whose
users are not in any training set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..base import RPPGMethod, register
from ..signal import bandpass, normalize

# The released EfficientPhys checkpoints are 72x72 with a temporal shift depth
# of 20; both are baked into the weights, not free choices.
IMG_SIZE = 72
FRAME_DEPTH = 20

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance over the whole clip, as the toolbox does."""
    x = np.asarray(x, dtype=np.float32)
    sd = float(x.std())
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - float(x.mean())) / sd


class _EfficientPhysBase(RPPGMethod):
    """Shared machinery; subclasses only pick a checkpoint."""

    needs_frames = True
    checkpoint: str = ""

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            import torch

            from .efficientphys import load_pretrained

            path = MODEL_DIR / self.checkpoint
            if not path.exists():
                raise FileNotFoundError(
                    "missing checkpoint {}. Download it from the rPPG-Toolbox "
                    "release directory into models/.".format(path)
                )
            torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
            self._model = load_pretrained(path, img_size=IMG_SIZE,
                                          frame_depth=FRAME_DEPTH)
        return self._model

    def __call__(self, frames: np.ndarray, fs: float) -> np.ndarray:
        import torch

        frames = np.asarray(frames)
        if frames.ndim != 4:
            raise ValueError(
                "expected (T, H, W, 3) face crops, got shape {}".format(frames.shape)
            )
        n = frames.shape[0]
        if n < FRAME_DEPTH + 1:
            return np.zeros(n)

        model = self._load()
        x = _standardize(frames)                       # (T, S, S, 3)
        x = np.transpose(x, (0, 3, 1, 2))              # (T, 3, S, S)

        # The model differences internally, so T frames in yield T-1 out, and
        # the temporal shift needs the length to be a multiple of FRAME_DEPTH.
        usable = ((n - 1) // FRAME_DEPTH) * FRAME_DEPTH
        if usable < FRAME_DEPTH:
            return np.zeros(n)
        x = x[: usable + 1]

        with torch.no_grad():
            out = model(torch.from_numpy(np.ascontiguousarray(x))).squeeze(-1)
        bvp = out.numpy().astype(np.float64)

        # Pad back to the window length so callers can treat every method alike.
        if bvp.size < n:
            bvp = np.concatenate([bvp, np.full(n - bvp.size, bvp[-1])])
        return normalize(bandpass(bvp[:n], fs))


@register
class EfficientPhysSCAMPS(_EfficientPhysBase):
    """Trained on SCAMPS. In-domain on SCAMPS -- used to verify the adapter,
    not to claim performance."""

    name = "efficientphys_scamps"
    checkpoint = "SCAMPS_EfficientPhys.pth"


@register
class EfficientPhysPURE(_EfficientPhysBase):
    """Trained on PURE. Cross-dataset for everything else here."""

    name = "efficientphys_pure"
    checkpoint = "PURE_EfficientPhys.pth"


@register
class EfficientPhysUBFC(_EfficientPhysBase):
    """Trained on UBFC-rPPG. Cross-dataset for everything else here."""

    name = "efficientphys_ubfc"
    checkpoint = "UBFC-rPPG_EfficientPhys.pth"
