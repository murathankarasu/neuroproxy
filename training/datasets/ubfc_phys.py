"""UBFC-Phys loader -- the first *state* supervision dataset.

Layout:
    <root>/s<ID>/vid_s<ID>_T<k>.avi
    <root>/s<ID>/bvp_s<ID>_T<k>.csv     # 64 Hz contact BVP
    <root>/s<ID>/eda_s<ID>_T<k>.csv     # 4 Hz EDA

Tasks: T1 = rest, T2 = speech, T3 = arithmetic.

WARNING -- the confound that decides whether this project is real:
T2 involves speaking, so it differs from T1 in mouth motion, head motion and
blink behaviour, not only in physiology. A classifier can separate the tasks
from motion alone and appear to "detect stress". Subject-independent splits do
NOT fix this. Before believing any state result on this dataset, run the
motion-only and SNR-only ablations in training/evaluation (see
docs/limitations.md). Task labels are exposed here as `labels["task"]` so those
ablations can be written against the same loader.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from neuroproxy.capture.video import probe, read_frames

from .base import Dataset, Recording, register

BVP_FPS = 64.0
EDA_FPS = 4.0
TASK_NAMES = {1: "rest", 2: "speech", 3: "arithmetic"}


def _read_csv_column(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros(0)
    return np.genfromtxt(path, delimiter=",").ravel()


@register
class UBFCPhys(Dataset):
    name = "ubfc_phys"

    def __init__(
        self,
        root: Optional[Path] = None,
        tasks: Optional[List[int]] = None,
        max_frames: Optional[int] = None,
    ):
        super().__init__(root)
        self.tasks = tasks or [1, 2, 3]
        self.max_frames = max_frames

    def recordings(self, limit: Optional[int] = None) -> List[Recording]:
        if not self.is_available():
            return []
        out: List[Recording] = []
        subjects = sorted(
            p for p in self.root.iterdir() if p.is_dir() and re.fullmatch(r"s\d+", p.name)
        )
        for sub in subjects:
            sid = sub.name
            for task in self.tasks:
                video = sub / "vid_{}_T{}.avi".format(sid, task)
                if not video.exists():
                    continue
                fps, n_frames = probe(video)
                if fps <= 0:
                    fps = 35.0  # UBFC-Phys ships 35 fps video
                if self.max_frames is not None:
                    n_frames = min(n_frames, self.max_frames)
                bvp = _read_csv_column(sub / "bvp_{}_T{}.csv".format(sid, task))
                eda = _read_csv_column(sub / "eda_{}_T{}.csv".format(sid, task))
                out.append(
                    Recording(
                        # subject_id excludes the task so GroupKFold/LOSO groups
                        # every task of one person together, as it must.
                        subject_id=sid,
                        fps=fps,
                        n_frames=n_frames,
                        frame_source=lambda v=video: read_frames(v, self.max_frames),
                        gt_bvp=bvp if bvp.size else None,
                        gt_bvp_fps=BVP_FPS if bvp.size else None,
                        labels={
                            "task": TASK_NAMES.get(task, str(task)),
                            "task_id": task,
                            "is_stress_task": task in (2, 3),
                        },
                        metadata={
                            "path": str(sub),
                            "detector": "auto",
                            "eda": eda if eda.size else None,
                            "eda_fps": EDA_FPS,
                            "session": "{}_T{}".format(sid, task),
                        },
                    )
                )
            if limit is not None and len({r.subject_id for r in out}) >= limit:
                break
        return out
