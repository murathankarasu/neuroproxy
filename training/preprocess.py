"""Cache face crops and aligned labels for neural training.

Extraction dominates the cost of any training experiment -- roughly 15 s per
recording against 1.8 s to train on a 6 s chunk -- so it is done once and
cached. Caching also freezes the preprocessing across experiments, which is
what makes two training runs comparable at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from neuroproxy.pipeline.offline import DEFAULT_CROP_SIZE, extract_traces

CACHE_VERSION = 1


@dataclass
class CachedClip:
    """One recording's crops, labels and the metadata needed to split on it."""

    subject_id: str
    session: str
    crops: np.ndarray          # (T, S, S, 3) uint8
    bvp: np.ndarray            # (T,) contact PPG on the video time base
    fps: float
    labels: Dict[str, object]

    def __len__(self) -> int:
        return int(self.crops.shape[0])


def cache_path(root: Path, session: str, crop_size: int, max_frames: int) -> Path:
    return root / "v{}_{}_{}_{}.npz".format(CACHE_VERSION, session, crop_size, max_frames)


def build_cache(
    dataset,
    out_dir: Path,
    crop_size: int = DEFAULT_CROP_SIZE,
    max_frames: int = 900,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> List[Path]:
    """Extract crops for every recording and write one npz per session."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rec in dataset.recordings(limit=limit):
        session = str(rec.metadata.get("session", rec.subject_id))
        path = cache_path(out_dir, session, crop_size, max_frames)
        if path.exists():
            written.append(path)
            continue
        traces = extract_traces(rec, crop_size=crop_size)
        if traces.crops is None:
            continue
        gt = rec.gt_bvp
        if gt is None:
            continue
        n = min(len(traces.crops), max_frames, len(gt))
        if n < 2 * crop_size:
            continue
        np.savez_compressed(
            path,
            crops=traces.crops[:n],
            bvp=np.asarray(rec.gt_bvp, dtype=np.float32)[:n],
            fps=np.float32(rec.fps),
            subject_id=rec.subject_id,
            session=session,
            valid=traces.valid[:n],
            labels=np.array([repr(dict(rec.labels))], dtype=object),
        )
        written.append(path)
        if verbose:
            print("  cached {} ({} frames)".format(session, n))
    return written


def load_cache(
    out_dir: Path, crop_size: int = DEFAULT_CROP_SIZE, max_frames: int = 900
) -> List[CachedClip]:
    """Read every cached clip matching this preprocessing configuration."""
    clips = []
    pattern = "v{}_*_{}_{}.npz".format(CACHE_VERSION, crop_size, max_frames)
    for path in sorted(Path(out_dir).glob(pattern)):
        with np.load(path, allow_pickle=True) as z:
            labels = {}
            if "labels" in z:
                try:
                    labels = eval(str(z["labels"][0]), {"__builtins__": {}}, {})
                except Exception:
                    labels = {}
            clips.append(
                CachedClip(
                    subject_id=str(z["subject_id"]),
                    session=str(z["session"]),
                    crops=z["crops"],
                    bvp=z["bvp"],
                    fps=float(z["fps"]),
                    labels=labels,
                )
            )
    return clips


def estimate_label_lag(clip, max_lag_s: float = 3.0, window_s: float = 20.0) -> float:
    """Seconds by which the contact PPG lags the video, for this recording.

    MCD-rPPG's `ppg_sync` files are aligned in *rate* but not reliably in
    *phase*. Measured across 49 cached recordings: only 12 sit within +-0.2 s
    of zero, 22 are off by more than a full second, and the spread runs from
    -2.68 s to +2.14 s.

    That does not affect heart rate -- a lag leaves the spectrum untouched --
    which is why every HR benchmark in this repo was unaffected and the problem
    went unnoticed. It makes *waveform* training impossible: the model is asked
    to predict a signal that is out of phase with its input, and no amount of
    capacity fixes that. Correcting it lifts median waveform correlation from
    0.344 to 0.565, and the count of usable recordings from 17/49 to 39/49.

    CAVEAT: the lag is found by cross-correlating a POS estimate against the
    label, so alignment carries a mild dependency on POS. It is a
    synchronisation step, not a modelling one -- the alternative is training on
    labels known to be misaligned -- but a model trained on these labels has
    seen a target that POS helped position, and that should be stated whenever
    such a model is compared against POS.
    """
    import numpy as _np

    from neuroproxy.rppg.base import get_method
    from neuroproxy.rppg.signal import bandpass
    from neuroproxy.vision.detector import FaceDetector, adaptive_skin_mask
    from neuroproxy.vision.roi import extract as roi_extract

    fs = clip.fps
    win = int(window_s * fs)
    n = min(len(clip.crops), len(clip.bvp))
    if n < win + 1:
        return 0.0

    det = FaceDetector("auto")
    trace = []
    for frame in clip.crops[: win + 1]:
        box = det.detect(frame)
        sample = (
            roi_extract(frame, box, mask=adaptive_skin_mask(frame, box))
            if box is not None else None
        )
        trace.append(sample.rgb if sample else [_np.nan] * 3)

    t = _np.asarray(trace, dtype=float)
    ok = _np.isfinite(t[:, 0])
    if ok.sum() < win:
        return 0.0
    idx = _np.arange(len(t))
    t = _np.stack([_np.interp(idx, idx[ok], t[ok, c]) for c in range(3)], 1)

    a = get_method("pos")(t[:win], fs)
    b = bandpass(_np.asarray(clip.bvp[:win], dtype=float), fs)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)

    max_lag = int(max_lag_s * fs)
    best_r, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        x = a[max_lag + lag : len(a) - max_lag + lag]
        y = b[max_lag : len(b) - max_lag]
        if x.size == 0:
            continue
        r = float(_np.dot(x, y) / x.size)
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_lag / fs


def subject_split(
    clips: List[CachedClip], test_fraction: float = 0.33, seed: int = 0
) -> "tuple":
    """Split by subject, never by clip.

    Design doc section 8.1 is explicit: the same face must not appear in both
    halves. Splitting clips would put a subject's rest recording in train and
    their post-exercise recording in test, which measures memorisation of that
    face rather than generalisation to a new one.
    """
    subjects = sorted({c.subject_id for c in clips})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    n_test = max(1, int(round(test_fraction * len(subjects))))
    test = {subjects[i] for i in order[:n_test]}
    train_clips = [c for c in clips if c.subject_id not in test]
    test_clips = [c for c in clips if c.subject_id in test]
    return train_clips, test_clips, sorted(test)
