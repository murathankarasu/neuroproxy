"""SCAMPS loader -- photorealistic synthetic avatars with aligned physiology.

SCAMPS (Microsoft Research, NeurIPS 2022) renders synthetic humans with a
cardiac signal driven through the skin shader, plus respiration, head pose and
facial action units. Obtained by direct download; no request form.

WHY THIS DATASET IS HERE
------------------------
It is the first data in this project that contains an actual face. The built-in
generator in `synthetic.py` paints a pulse into a flat ellipse, which validates
the harness arithmetic and nothing else. SCAMPS exercises face detection, ROI
selection on real facial geometry, and skin rendering with subsurface
scattering.

It is still rendered, not recorded. It does not settle real skin tone response,
real camera pipelines, real ambient light or real motion artefacts. The design
doc's HR MAE bar remains a question for UBFC-rPPG and PURE.

LICENCE -- READ BEFORE USING FOR ANYTHING BUT VALIDATION
--------------------------------------------------------
Research Use of Data Agreement (R-UDA), research-only. The terms forbid using
the data "or any Results in any commercial offering, including as part of a
product or service (or to improve any product or service)". Use it to check
that published algorithms behave correctly; do not train a proprietary model on
it. See docs/datasets.md.

LAYOUT
------
    <root>/P??????.mat        one clip each, HDF5-format MATLAB v7.3

Each file holds, with dimensions reversed by h5py relative to MATLAB:
    RawFrames   (3, 320, 240, 600)  float64 in [0, 1], as (C, W, H, T)
    Xsub        (3, 240, 240, 600)  face-cropped variant, unused here
    d_ppg       (600, 1)            cardiac waveform, 30 fps
    d_ekg, d_br                     EKG and breathing waveforms
    d_pitch, d_yaw, d_roll          head pose per frame
    au*                             facial action unit intensities

`RawFrames` is used rather than `Xsub` on purpose: cropping to the face by hand
would skip the detector, which is one of the things this dataset is here to
test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from .base import Dataset, Recording, register

FPS = 30.0
# Frames per read. The time axis is contiguous in the file, so a frame is the
# most strided possible access: reading one at a time costs 186 s per clip
# versus 4.7 s in chunks of 100.
CHUNK = 100


def _iter_frames(path: Path, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames of shape (H, W, 3), reading in time chunks."""
    import h5py

    with h5py.File(str(path), "r") as f:
        raw = f["RawFrames"]
        n = raw.shape[3] if max_frames is None else min(max_frames, raw.shape[3])
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            block = np.asarray(raw[:, :, :, start:stop])   # (C, W, H, t)
            block = np.transpose(block, (3, 2, 1, 0))      # (t, H, W, C)
            block = np.clip(block * 255.0, 0, 255).astype(np.uint8)
            for i in range(block.shape[0]):
                yield block[i]


@register
class SCAMPS(Dataset):
    name = "scamps"

    def __init__(self, root: Optional[Path] = None, max_frames: Optional[int] = None):
        super().__init__(root)
        self.max_frames = max_frames

    def recordings(self, limit: Optional[int] = None) -> List[Recording]:
        if not self.is_available():
            return []
        import h5py

        files = sorted(self.root.glob("P*.mat"))
        out: List[Recording] = []
        for path in files:
            with h5py.File(str(path), "r") as f:
                if "RawFrames" not in f or "d_ppg" not in f:
                    continue
                n_frames = int(f["RawFrames"].shape[3])
                ppg = np.asarray(f["d_ppg"]).ravel()
                pose = {
                    k: float(np.ptp(np.asarray(f[k]))) if k in f else 0.0
                    for k in ("d_pitch", "d_yaw", "d_roll")
                }
                breathing = np.asarray(f["d_br"]).ravel() if "d_br" in f else None
            if self.max_frames is not None:
                n_frames = min(n_frames, self.max_frames)

            out.append(
                Recording(
                    subject_id=path.stem,
                    fps=FPS,
                    n_frames=n_frames,
                    frame_source=lambda p=path: _iter_frames(p, self.max_frames),
                    gt_bvp=ppg,
                    gt_bvp_fps=FPS,
                    labels={
                        # Total head-pose excursion in the clip, for splitting
                        # results by motion once enough clips are available.
                        "pose_range_deg": max(pose.values()) if pose else 0.0,
                    },
                    metadata={
                        "path": str(path),
                        "detector": "auto",
                        "synthetic": True,
                        "rendered": True,
                        "licence": "R-UDA research-only, no commercial use",
                        "breathing": breathing,
                        "breathing_fps": FPS,
                    },
                )
            )
            if limit is not None and len(out) >= limit:
                break
        return out
