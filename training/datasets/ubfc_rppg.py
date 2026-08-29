"""UBFC-rPPG loader -- the pulse-accuracy benchmark.

Layout (DATASET_2, the common release):
    <root>/subject<N>/vid.avi
    <root>/subject<N>/ground_truth.txt   # 3 rows: BVP, HR, timestamps

DATASET_1 uses `gtdump.xmp` (comma-separated, BVP in column 3); both are
handled. This dataset has contact PPG, so it is the only place an HR MAE
number means anything -- design doc section 6 lists it under rPPG robustness.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from neuroproxy.capture.video import probe, read_frames

from .base import Dataset, Recording, register

DEFAULT_GT_FPS = 30.0  # UBFC ground truth is sampled with the video


def _read_ground_truth(path: Path) -> "tuple":
    """Return (bvp, fps). Falls back to the video rate when timestamps are absent."""
    if path.name.endswith(".xmp"):
        rows = np.genfromtxt(path, delimiter=",")
        bvp = rows[:, 3] if rows.ndim == 2 and rows.shape[1] > 3 else rows.ravel()
        return np.asarray(bvp, float), DEFAULT_GT_FPS

    rows = [r for r in path.read_text().strip().splitlines() if r.strip()]
    values = [np.fromstring(r, sep=" ") for r in rows]
    bvp = values[0]
    fps = DEFAULT_GT_FPS
    if len(values) >= 3 and values[2].size > 1:
        t = values[2]
        dt = np.diff(t)
        dt = dt[dt > 0]
        if dt.size:
            fps = float(1.0 / np.median(dt))
    return np.asarray(bvp, float), fps


@register
class UBFCrPPG(Dataset):
    name = "ubfc_rppg"

    def __init__(self, root: Optional[Path] = None, max_frames: Optional[int] = None):
        super().__init__(root)
        self.max_frames = max_frames

    def recordings(self, limit: Optional[int] = None) -> List[Recording]:
        if not self.is_available():
            return []
        out: List[Recording] = []
        subjects = sorted(
            p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("subject")
        )
        for sub in subjects:
            video = sub / "vid.avi"
            gt_path = next(
                (sub / n for n in ("ground_truth.txt", "gtdump.xmp") if (sub / n).exists()),
                None,
            )
            if not video.exists() or gt_path is None:
                continue
            fps, n_frames = probe(video)
            if fps <= 0:
                fps = DEFAULT_GT_FPS
            if self.max_frames is not None:
                n_frames = min(n_frames, self.max_frames)
            bvp, gt_fps = _read_ground_truth(gt_path)
            out.append(
                Recording(
                    subject_id=sub.name,
                    fps=fps,
                    n_frames=n_frames,
                    frame_source=lambda v=video: read_frames(v, self.max_frames),
                    gt_bvp=bvp,
                    gt_bvp_fps=gt_fps,
                    metadata={"path": str(sub), "detector": "auto"},
                )
            )
            if limit is not None and len(out) >= limit:
                break
        return out
