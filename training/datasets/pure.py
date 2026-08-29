"""PURE loader -- controlled head-motion benchmark.

Layout:
    <root>/<NN-MM>/<NN-MM>/*.png      # image sequence, 30 fps
    <root>/<NN-MM>/<NN-MM>.json       # {"/FullPackage": [{"Value": {"waveform": ...}}]}

The six sessions per subject are steady / talking / slow translation / fast
translation / small rotation / medium rotation. That makes PURE the right place
to measure the motion-robustness curve the design doc asks for in section 8.2,
rather than asserting robustness from a still-subject dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from neuroproxy.capture.video import read_image_sequence

from .base import Dataset, Recording, register

VIDEO_FPS = 30.0
GT_FPS = 60.0  # PURE pulse oximeter runs at 60 Hz


@register
class PURE(Dataset):
    name = "pure"

    def __init__(self, root: Optional[Path] = None, max_frames: Optional[int] = None):
        super().__init__(root)
        self.max_frames = max_frames

    def recordings(self, limit: Optional[int] = None) -> List[Recording]:
        if not self.is_available():
            return []
        out: List[Recording] = []
        sessions = sorted(p for p in self.root.iterdir() if p.is_dir())
        for sess in sessions:
            json_path = sess / "{}.json".format(sess.name)
            image_dir = sess / sess.name
            if not json_path.exists() or not image_dir.is_dir():
                continue
            payload = json.loads(json_path.read_text())
            waveform = np.array(
                [e["Value"]["waveform"] for e in payload.get("/FullPackage", [])],
                dtype=float,
            )
            n_frames = len(list(image_dir.glob("*.png")))
            if self.max_frames is not None:
                n_frames = min(n_frames, self.max_frames)
            subject = sess.name.split("-")[0]
            out.append(
                Recording(
                    subject_id="pure{}".format(subject),
                    fps=VIDEO_FPS,
                    n_frames=n_frames,
                    frame_source=lambda d=image_dir: read_image_sequence(
                        d, max_frames=self.max_frames
                    ),
                    gt_bvp=waveform if waveform.size else None,
                    gt_bvp_fps=GT_FPS if waveform.size else None,
                    labels={"session": sess.name},
                    metadata={"path": str(sess), "detector": "auto"},
                )
            )
            if limit is not None and len({r.subject_id for r in out}) >= limit:
                break
        return out
