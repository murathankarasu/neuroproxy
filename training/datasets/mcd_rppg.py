"""MCD-rPPG loader -- the first dataset here with real human subjects.

MCD-rPPG (Egorov et al.) records 600 people at rest and immediately after
exercise, from three consumer cameras, with synchronised contact PPG at 100 Hz
and ECG. Obtained by direct download from Hugging Face: no request form, no
EULA, no account.

WHY IT MATTERS MORE THAN ITS SIZE SUGGESTS
------------------------------------------
1. **Real people.** Everything before this was either a painted ellipse or a
   rendered avatar.
2. **CC-BY-4.0.** The only rPPG dataset found that permits commercial use.
   Every other public option is research-only (docs/datasets.md), which meant
   nothing could legitimately train a shipping model.
3. **Compressed video.** The recordings are MPEG-4 (FMP4) from consumer
   webcams, so they exercise the codec damage that measurements here identify
   as the single largest threat to POS (docs/limitations.md section 3). An
   uncompressed dataset cannot test that at all.
4. **Rest and post-exercise.** Reference pulse spans 49-153 bpm, and the
   before/after structure is the rest/task protocol the explainer document
   asks for, without needing to run a study.
5. **Demographics.** `db.csv` carries age, sex, BMI and clinical measurements
   per subject, which is what makes the subgroup fairness reporting in
   limitations section 7 possible rather than aspirational.

LAYOUT
------
    <root>/db.csv                                     per-recording metadata
    <root>/video/<id>_<camera>_<step>.avi             640x480 MPEG-4
    <root>/ppg_sync/<id>_<camera>_<step>.txt          PPG aligned per frame
    <root>/meta/<id>_<camera>_<step>.txt              frame index + timestamp

`ppg_sync` is already on the video's time base, one sample per frame, which
removes the usual resampling step and its usual off-by-one errors.

Frame rate is computed from the `meta` timestamps rather than trusted from the
container: the FullHD webcam reports 29.90 and measures 29.991, and every
filter downstream assumes a constant, correct rate.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from neuroproxy.capture.video import read_frames

from .base import Dataset, Recording, register

DEFAULT_CAMERA = "FullHDwebcam"
FALLBACK_FPS = 30.0


def _read_timestamps(path: Path) -> List[dt.datetime]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) != 2 or not parts[0].strip().isdigit():
            continue
        try:
            out.append(dt.datetime.fromisoformat(parts[1].strip()))
        except ValueError:
            continue
    return out


def measured_fps(timestamps: List[dt.datetime]) -> Optional[float]:
    """Frame rate from actual timestamps, not from the container header."""
    if len(timestamps) < 10:
        return None
    span = (timestamps[-1] - timestamps[0]).total_seconds()
    if span <= 0:
        return None
    return float((len(timestamps) - 1) / span)


def _read_ppg_sync(path: Path) -> np.ndarray:
    """Per-frame PPG. First column is the value; the second is a time delta."""
    if not path.exists():
        return np.zeros(0)
    data = np.loadtxt(path)
    if data.ndim == 2:
        return np.asarray(data[:, 0], dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


@register
class MCDrPPG(Dataset):
    name = "mcd_rppg"

    def __init__(
        self,
        root: Optional[Path] = None,
        camera: str = DEFAULT_CAMERA,
        steps: Optional[List[str]] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        self.camera = camera
        self.steps = steps or ["before", "after"]
        self.max_frames = max_frames

    def _metadata(self) -> Dict:
        """Index db.csv by (patient_id, camera, step)."""
        path = self.root / "db.csv" if self.root else None
        if not path or not path.exists():
            return {}
        out = {}
        with path.open() as fh:
            for row in csv.DictReader(fh):
                out[(row["patient_id"], row["camera"], row["step"])] = row
        return out

    def recordings(self, limit: Optional[int] = None) -> List[Recording]:
        if not self.is_available():
            return []
        meta_index = self._metadata()
        video_dir = self.root / "video"
        if not video_dir.is_dir():
            return []

        out: List[Recording] = []
        seen_subjects = set()
        for video in sorted(video_dir.glob("*_{}_*.avi".format(self.camera))):
            stem = video.stem
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            subject, step = parts[0], parts[-1]
            if step not in self.steps:
                continue

            ppg = _read_ppg_sync(self.root / "ppg_sync" / (stem + ".txt"))
            timestamps = _read_timestamps(self.root / "meta" / (stem + ".txt"))
            fps = measured_fps(timestamps) or FALLBACK_FPS
            if ppg.size == 0:
                continue

            n_frames = ppg.size
            if self.max_frames is not None:
                n_frames = min(n_frames, self.max_frames)

            row = meta_index.get((subject, self.camera, step), {})
            labels = {
                "step": step,                       # "before" = rest
                "is_post_exercise": step == "after",
                "reference_pulse_bpm": _as_float(row.get("pulse")),
                "age": _as_float(row.get("age")),
                "sex": row.get("sex"),
                "bmi": _as_float(row.get("bmi")),
            }

            out.append(
                Recording(
                    # Both conditions of one person share a subject_id, so
                    # subject-independent splits group them together.
                    subject_id=subject,
                    fps=fps,
                    n_frames=n_frames,
                    frame_source=lambda v=video: read_frames(v, self.max_frames),
                    gt_bvp=ppg,
                    gt_bvp_fps=fps,          # already on the video time base
                    labels=labels,
                    metadata={
                        "path": str(video),
                        "detector": "auto",
                        "camera": self.camera,
                        "session": stem,
                        "licence": "CC-BY-4.0, commercial use permitted",
                        "container_compressed": True,
                    },
                )
            )
            seen_subjects.add(subject)
            if limit is not None and len(seen_subjects) >= limit:
                break
        return out


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
