"""Video file reading, normalised to RGB uint8 frames.

Kept separate from the datasets so that swapping in a different decoder (or a
live camera) does not touch dataset code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


def probe(path: Path) -> Tuple[float, int]:
    """Return (fps, n_frames) without decoding the whole file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError("cannot open video: {}".format(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, n
    finally:
        cap.release()


def read_frames(
    path: Path, max_frames: Optional[int] = None
) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames from a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError("cannot open video: {}".format(path))
    try:
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def read_image_sequence(
    directory: Path, pattern: str = "*.png", max_frames: Optional[int] = None
) -> Iterator[np.ndarray]:
    """Yield RGB frames from a sorted directory of images (PURE-style)."""
    files = sorted(Path(directory).glob(pattern))
    if max_frames is not None:
        files = files[:max_frames]
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
