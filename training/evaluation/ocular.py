"""Validate blink detection against ground truth.

Rate error alone is not enough: a detector that fires at random with roughly
the right frequency scores well on rate and is useless. Blinks are events, so
they are matched as events -- precision, recall and F1 against ground-truth
onsets within a tolerance -- and the rate comparison is reported alongside.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# A detected blink counts as matched if its onset falls within this many
# seconds of a ground-truth onset. Wider than a blink because the geometric eye
# region responds slightly after the eyelid starts moving.
MATCH_TOLERANCE_S = 0.30


@dataclass
class OcularEval:
    subject_id: str
    n_gt: int
    n_detected: int
    true_positives: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    gt_rate_per_min: Optional[float] = None
    est_rate_per_min: Optional[float] = None
    rate_error_per_min: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def gt_blink_onsets(au45: Sequence[float], threshold: float = 0.5) -> np.ndarray:
    """Ground-truth blink onsets: upward crossings of the blink action unit."""
    a = np.asarray(au45, dtype=np.float64)
    if a.size < 2:
        return np.zeros(0, dtype=int)
    above = a > threshold
    return np.flatnonzero(above[1:] & ~above[:-1]) + 1


def match_events(
    detected: Sequence[int], truth: Sequence[int], fs: float,
    tolerance_s: float = MATCH_TOLERANCE_S,
) -> int:
    """Greedy one-to-one matching of detected onsets to ground-truth onsets."""
    tol = tolerance_s * fs
    unused = list(truth)
    tp = 0
    for d in detected:
        if not unused:
            break
        distances = [abs(d - t) for t in unused]
        k = int(np.argmin(distances))
        if distances[k] <= tol:
            unused.pop(k)
            tp += 1
    return tp


def evaluate(
    subject_id: str,
    detected_onsets: Sequence[int],
    au45: Sequence[float],
    fs: float,
    duration_s: Optional[float] = None,
) -> OcularEval:
    truth = gt_blink_onsets(au45)
    n_gt, n_det = int(truth.size), len(detected_onsets)
    ev = OcularEval(subject_id=subject_id, n_gt=n_gt, n_detected=n_det)

    tp = match_events(sorted(detected_onsets), truth.tolist(), fs)
    ev.true_positives = tp
    if n_det:
        ev.precision = tp / n_det
    if n_gt:
        ev.recall = tp / n_gt
    # A detector that fires never has undefined precision but its F1 is 0, not
    # missing. Leaving it as None drops the subject out of the mean and makes a
    # detector that finds nothing look excellent on the two subjects it fired
    # for -- which is exactly what the first run of this reported.
    if n_gt and n_det == 0:
        ev.f1 = 0.0
    elif ev.precision is not None and ev.recall is not None:
        denom = ev.precision + ev.recall
        ev.f1 = (2 * ev.precision * ev.recall / denom) if denom > 0 else 0.0

    dur = duration_s if duration_s else (len(au45) / fs)
    if dur > 0:
        ev.gt_rate_per_min = n_gt / dur * 60.0
        ev.est_rate_per_min = n_det / dur * 60.0
        ev.rate_error_per_min = abs(ev.est_rate_per_min - ev.gt_rate_per_min)
    return ev


def format_table(rows: List[OcularEval]) -> str:
    header = "{:<9} {:>5} {:>5} {:>5} {:>7} {:>7} {:>6} {:>9} {:>9}".format(
        "subject", "GT", "det", "TP", "prec", "recall", "F1", "GT_rate", "est_rate"
    )
    lines = [header, "-" * len(header)]
    f = lambda v: "  n/a" if v is None else "{:5.2f}".format(v)
    for r in rows:
        lines.append(
            "{:<9} {:>5d} {:>5d} {:>5d} {:>7} {:>7} {:>6} {:>9} {:>9}".format(
                r.subject_id, r.n_gt, r.n_detected, r.true_positives,
                f(r.precision), f(r.recall), f(r.f1),
                "  n/a" if r.gt_rate_per_min is None else "{:7.1f}".format(r.gt_rate_per_min),
                "  n/a" if r.est_rate_per_min is None else "{:7.1f}".format(r.est_rate_per_min),
            )
        )
    # Treat "fired never" as precision 0 for the mean, for the same reason.
    ps = [(r.precision if r.precision is not None else 0.0) for r in rows]
    rs = [r.recall for r in rows if r.recall is not None]
    fs_ = [r.f1 for r in rows if r.f1 is not None]
    es = [r.rate_error_per_min for r in rows if r.rate_error_per_min is not None]
    if fs_:
        lines.append("")
        lines.append(
            "  mean precision {:.2f}   recall {:.2f}   F1 {:.2f}   "
            "blink-rate MAE {:.1f}/min".format(
                float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs_)),
                float(np.mean(es)) if es else float("nan"),
            )
        )
    return "\n".join(lines)
