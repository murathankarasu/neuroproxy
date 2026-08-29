"""Live camera frame source.

Yields RGB frames exactly like `capture.video.read_frames`, so the streaming
engine cannot tell a webcam from a file. That is the point: every offline
benchmark in this repo was produced on the same code path a live session runs.

CAPTURE SETTINGS ARE NOT COSMETIC. Auto-exposure and auto-white-balance
continuously renormalise the frame, and the pulse is a ~1% colour modulation --
the very thing those controls are built to remove. This module asks the driver
to disable them and, crucially, *reports whether the request was honoured*
rather than assuming it. Many webcams silently ignore it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import cv2
import numpy as np


@dataclass
class CaptureSettings:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    # Requested, not guaranteed. `CameraSource.applied` records what stuck.
    disable_auto_exposure: bool = True
    disable_auto_white_balance: bool = True


@dataclass
class CaptureReport:
    """What the camera actually did, for the session record."""

    requested: Dict[str, object] = field(default_factory=dict)
    applied: Dict[str, object] = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def auto_controls_disabled(self) -> bool:
        return not self.warnings


class CameraSource:
    """Context-managed webcam yielding RGB frames."""

    def __init__(self, index: int = 0, settings: Optional[CaptureSettings] = None):
        self.index = index
        self.settings = settings or CaptureSettings()
        self.report = CaptureReport()
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> "CameraSource":
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise IOError("cannot open camera {}".format(self.index))
        s = self.settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
        cap.set(cv2.CAP_PROP_FPS, s.fps)
        if s.disable_auto_exposure:
            # 0.25 is the widely used "manual" value for V4L2-style backends.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        if s.disable_auto_white_balance:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)

        self.report.requested = {
            "width": s.width, "height": s.height, "fps": s.fps,
            "auto_exposure_disabled": s.disable_auto_exposure,
            "auto_wb_disabled": s.disable_auto_white_balance,
        }
        self.report.applied = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "auto_exposure": float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            "auto_wb": float(cap.get(cv2.CAP_PROP_AUTO_WB)),
        }
        if s.disable_auto_exposure and self.report.applied["auto_exposure"] not in (0.25, 1.0, 0.0):
            self.report.warnings.append(
                "auto-exposure could not be disabled (driver reports {}); the "
                "pulse signal will be partly normalised away".format(
                    self.report.applied["auto_exposure"]))
        if s.disable_auto_white_balance and self.report.applied["auto_wb"] not in (0.0,):
            self.report.warnings.append(
                "auto white balance could not be disabled (driver reports {})".format(
                    self.report.applied["auto_wb"]))
        self._cap = cap
        return self

    def frames(self, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        if self._cap is None:
            self.open()
        assert self._cap is not None
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


def list_cameras(max_index: int = 4) -> list:
    """Indices that open successfully. Opens and immediately releases each."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found
