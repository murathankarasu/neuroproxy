"""Ablations from design doc section 8.3.

Each ablation answers "is this component earning its place?" with a number
rather than an argument. The first one implemented is
"No baseline vs personal baseline".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Area under the ROC curve via the rank-sum statistic, ties averaged.

    Implemented directly rather than pulled from scikit-learn: the sensor layer
    has no learned components yet and should not acquire a heavyweight
    dependency for one formula.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    from scipy.stats import rankdata

    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class BaselineAblation:
    """Task-vs-rest separability with and without personal normalisation."""

    feature: str
    n_subjects: int
    n_rest: int
    n_task: int
    auroc_raw: Optional[float] = None
    auroc_baselined: Optional[float] = None
    auroc_raw_within_subject: Optional[float] = None
    delta: Optional[float] = None
    # Fraction of the distance from pooled-raw up to the within-subject ceiling
    # that personal normalisation actually closes. More informative than the
    # raw delta, which grows with effect size for trivial reasons.
    gap_recovered: Optional[float] = None
    subject_spread_bpm: Optional[float] = None
    task_delta_bpm: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def baseline_ablation(
    per_subject: Dict[str, List[Tuple[str, float, Optional[float]]]],
    feature: str = "hr_bpm",
    subject_spread_bpm: Optional[float] = None,
    task_delta_bpm: Optional[float] = None,
) -> BaselineAblation:
    """Compare pooled task-vs-rest separability, raw versus baselined.

    `per_subject` maps subject id to a list of
    `(condition, raw_value, baselined_value)` triples.

    Three numbers are produced, and the third is the one that keeps the result
    honest:

      auroc_raw                  pooled across subjects, absolute values
      auroc_baselined            pooled across subjects, personal z-scores
      auroc_raw_within_subject   raw values, scored per subject then averaged

    The within-subject figure is the ceiling that personal normalisation is
    trying to reach. If baselining lands near it, it has removed the
    between-subject offset and nothing else. If it exceeds it, something is
    wrong -- most likely the baseline period overlaps the task.
    """
    raw_scores, base_scores, labels = [], [], []
    within = []
    n_subjects = 0

    for subject, rows in per_subject.items():
        usable = [r for r in rows if r[0] in ("rest", "task")]
        if not usable:
            continue
        n_subjects += 1
        s_raw = [r[1] for r in usable]
        s_lab = [1 if r[0] == "task" else 0 for r in usable]
        raw_scores.extend(s_raw)
        labels.extend(s_lab)
        base_scores.extend([r[2] if r[2] is not None else np.nan for r in usable])
        a = auroc(s_raw, s_lab)
        if a is not None:
            within.append(a)

    result = BaselineAblation(
        feature=feature,
        n_subjects=n_subjects,
        n_rest=int(sum(1 for l in labels if l == 0)),
        n_task=int(sum(1 for l in labels if l == 1)),
        subject_spread_bpm=subject_spread_bpm,
        task_delta_bpm=task_delta_bpm,
    )
    if not labels:
        result.notes.append("no rest/task windows survived the quality gate")
        return result

    result.auroc_raw = auroc(raw_scores, labels)
    result.auroc_raw_within_subject = float(np.mean(within)) if within else None

    base_arr = np.asarray(base_scores, dtype=float)
    lab_arr = np.asarray(labels, dtype=int)
    finite = np.isfinite(base_arr)
    if finite.sum() < 10:
        result.notes.append(
            "only {} windows had a personal baseline; most subjects failed "
            "calibration".format(int(finite.sum()))
        )
    else:
        result.auroc_baselined = auroc(base_arr[finite], lab_arr[finite])
        if finite.sum() < len(base_arr):
            result.notes.append(
                "{} of {} windows lacked a baseline and were excluded".format(
                    len(base_arr) - int(finite.sum()), len(base_arr)
                )
            )

    if result.auroc_raw is not None and result.auroc_baselined is not None:
        result.delta = result.auroc_baselined - result.auroc_raw
        ceiling = result.auroc_raw_within_subject
        if ceiling is not None and ceiling - result.auroc_raw > 1e-9:
            result.gap_recovered = float(
                result.delta / (ceiling - result.auroc_raw)
            )
    return result


def format_baseline_ablation(r: BaselineAblation) -> str:
    lines = [
        "personal baseline ablation  (feature: {})".format(r.feature),
        "  {} subjects, {} rest windows, {} task windows".format(
            r.n_subjects, r.n_rest, r.n_task
        ),
    ]
    if r.subject_spread_bpm is not None and r.task_delta_bpm is not None:
        lines.append(
            "  between-subject resting spread {:.0f} bpm vs task response "
            "{:.0f} bpm".format(r.subject_spread_bpm, r.task_delta_bpm)
        )
    lines.append("")

    def fmt(v):
        return "n/a" if v is None else "{:.3f}".format(v)

    lines.append("  task-vs-rest AUROC")
    lines.append(
        "    raw, pooled across subjects        {}".format(fmt(r.auroc_raw))
    )
    lines.append(
        "    personal baseline, pooled          {}".format(fmt(r.auroc_baselined))
    )
    lines.append(
        "    raw, scored within each subject    {}   <- ceiling".format(
            fmt(r.auroc_raw_within_subject)
        )
    )
    if r.gap_recovered is not None:
        lines.append("")
        lines.append(
            "    gap to ceiling closed             {:.0%}".format(r.gap_recovered)
        )
    if r.delta is not None:
        lines.append("")
        verdict = (
            "personal baseline earns its place"
            if r.delta > 0.05
            else "no measurable benefit on this cohort"
            if abs(r.delta) <= 0.05
            else "personal baseline HURTS separability"
        )
        lines.append("    delta {:+.3f}  ->  {}".format(r.delta, verdict))
    for n in r.notes:
        lines.append("  note: " + n)
    return "\n".join(lines)
