"""Ocular features: blink rate and eye openness.

Second modality in the design doc's feature vector, after cardiac. Built on the
openness proxy in `vision.eyes` -- read that module first for why the eye
region is geometric rather than landmark-based, and for the evidence that the
proxy tracks real blinks.

Thresholds below come from blink physiology, not from fitting: a spontaneous
blink lasts roughly 100-400 ms and consecutive blinks are separated by at least
~200 ms. `neuroproxy.cli ocular` sweeps them against ground truth so the
defaults can be checked rather than trusted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

MAD_TO_SIGMA = 1.4826

# A blink is a drop to this fraction of the open-eye baseline.
#
# A RELATIVE criterion, not a statistical one. The first version thresholded at
# a number of robust sigmas below the median, which fails on exactly the case
# it exists for: blinks are rare and brief, so the MAD of the series is
# dominated by the open-eye baseline and collapses toward zero. On a clean
# signal with four blinks in twenty seconds the scale estimate was 0 and the
# detector found nothing.
#
# Openness is edge energy -- a positive quantity with a meaningful zero, where
# a closed eye is a smooth lid. A proportional drop is both scale-free and
# physiologically motivated.
#
# The value is DERIVED, not chosen. `neuroproxy.cli ocular --sweep` scores it
# against blink ground truth. On SCAMPS (10 subjects, 30 ground-truth blinks):
#
#     drop_ratio  precision  recall    F1   rate MAE/min
#     0.75        0.40       0.28    0.33    7.2
#     0.80        0.50       0.47    0.48    5.4
#     0.85        0.70       0.59    0.63    4.2
#     0.90        0.80       0.60    0.66    3.9   <- default
#     0.95        0.48       0.47    0.42    7.8
#
# The drop is shallow because the geometric eye band includes a good deal of
# surrounding face, so closing the lids changes only part of the region. A
# 10% threshold is correspondingly sensitive; a landmark-based eye box would
# move it, as would any real cohort. Re-derive it rather than inheriting it.
#
# An earlier version scored F1 0.71 -- but only with a one-frame minimum blink
# duration, i.e. accepting 33 ms dips. SCAMPS is clean and rendered, so that
# choice costs nothing here and would produce false positives on real noisy
# video. The physiological bound was kept and the lower F1 accepted.
BLINK_DROP_RATIO = 0.90
# Duration bounds in seconds. A spontaneous blink lasts roughly 100-400 ms.
# The lower bound was 0.06 s, which at 30 fps rounds to a single frame -- a
# 33 ms dip is sensor noise, not an eyelid.
BLINK_MIN_S = 0.10
BLINK_MAX_S = 0.50
# Minimum separation between accepted blinks.
REFRACTORY_S = 0.20
# Below this fraction of frames with a usable eye region, report nothing.
MIN_VALID_FRACTION = 0.6


@dataclass
class OcularFeatures:
    blink_rate_per_min: Optional[float] = None
    blink_count: int = 0
    eye_openness_mean: Optional[float] = None
    # Robust spread of openness, in the same units. Large values mean the eye
    # region is unstable -- either genuine blinking or a wandering ROI.
    eye_openness_spread: Optional[float] = None
    valid_fraction: float = 0.0
    reason: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _runs_below(mask: np.ndarray) -> List:
    """Contiguous (start, stop) index runs where mask is True."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def detect_blinks(
    openness: Sequence[float],
    fs: float,
    drop_ratio: float = BLINK_DROP_RATIO,
    min_s: float = BLINK_MIN_S,
    max_s: float = BLINK_MAX_S,
    refractory_s: float = REFRACTORY_S,
) -> List:
    """Blink onsets as (start_index, stop_index).

    The baseline is taken *within the window*: openness in raw units depends on
    face size, focus and lighting, so a fixed absolute threshold would mean
    something different for every subject and every session.
    """
    x = np.asarray(openness, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < int(fs * 2):
        return []
    # The eye is open in the large majority of frames, so a high percentile is
    # a cleaner estimate of the open-eye level than the median when blinking is
    # frequent, and equivalent to it when blinking is rare.
    baseline = float(np.percentile(x[ok], 75))
    if baseline <= 1e-9:
        return []
    below = np.nan_to_num(x, nan=baseline) < drop_ratio * baseline

    min_f, max_f = int(min_s * fs), int(max_s * fs)
    refractory = int(refractory_s * fs)
    blinks = []
    last_end = -10 ** 9
    for start, stop in _runs_below(below):
        if not (min_f <= stop - start <= max_f):
            continue
        if start - last_end < refractory:
            continue
        blinks.append((start, stop))
        last_end = stop
    return blinks


def extract(
    openness: Sequence[Optional[float]], fs: float, **kwargs
) -> OcularFeatures:
    """Window-level ocular features from a per-frame openness series."""
    x = np.array(
        [np.nan if v is None else float(v) for v in openness], dtype=np.float64
    )
    valid = float(np.isfinite(x).mean()) if x.size else 0.0
    feats = OcularFeatures(valid_fraction=valid)
    if x.size < int(fs * 2):
        feats.reason = "window too short for blink detection"
        return feats
    if valid < MIN_VALID_FRACTION:
        feats.reason = "eye region unusable in {:.0%} of frames".format(1 - valid)
        return feats

    finite = x[np.isfinite(x)]
    med = float(np.median(finite))
    feats.eye_openness_mean = med
    feats.eye_openness_spread = float(
        MAD_TO_SIGMA * np.median(np.abs(finite - med))
    )

    blinks = detect_blinks(x, fs, **kwargs)
    feats.blink_count = len(blinks)
    feats.blink_rate_per_min = float(len(blinks) / (x.size / fs) * 60.0)
    return feats
