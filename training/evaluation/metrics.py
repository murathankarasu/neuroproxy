"""Benchmark metrics for the rPPG sensor layer (design doc section 8.2).

Aggregation rule: metrics are computed *per subject first*, then summarised
across subjects. Pooling every window from every subject lets one long or one
easy recording dominate, which is how rPPG papers accidentally report numbers
that do not survive a new cohort.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SubjectMetrics:
    subject_id: str
    n_windows: int
    n_valid: int
    coverage: float                    # valid windows / total windows
    # Error over windows the engine chose to answer. Selection-biased by
    # design: abstention removes hard windows, so this improves whenever
    # coverage falls. Never read it without reading coverage.
    mae_bpm: Optional[float] = None
    # Error over every window with a ground-truth pair, including refused ones,
    # using the prediction they would have made. Independent of the abstain
    # policy, so this is the number that measures signal degradation.
    mae_all_bpm: Optional[float] = None
    rmse_bpm: Optional[float] = None
    pearson_r: Optional[float] = None
    mean_snr_db: Optional[float] = None
    mean_quality: Optional[float] = None
    # False when the signal is not recoverable from this recording at all, so
    # refusing every window was the correct behaviour rather than a failure.
    usable: bool = True
    # Recording labels (age, sex, condition, ...) carried through so results
    # can be broken down by subgroup rather than only pooled.
    labels: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class BenchmarkMetrics:
    """Cohort-level summary. `median_mae_bpm` is the go/no-go number."""

    method: str
    dataset: str
    n_subjects: int
    n_windows: int
    coverage: float
    median_mae_bpm: Optional[float] = None
    mean_mae_bpm: Optional[float] = None
    worst_subject_mae_bpm: Optional[float] = None
    # Abstain-independent counterpart of median_mae_bpm. Compare the two: a
    # large gap means the reported accuracy is being bought with coverage.
    median_mae_all_bpm: Optional[float] = None
    # Pooled coverage conflates two different failures: sessions where nothing
    # was measurable, and windows dropped inside otherwise fine sessions. The
    # first is a capture/onboarding problem, the second a signal problem, and
    # they have different fixes. Reported separately.
    usable_session_rate: Optional[float] = None
    coverage_within_usable: Optional[float] = None
    rmse_bpm: Optional[float] = None
    pearson_r: Optional[float] = None
    mean_snr_db: Optional[float] = None
    subjects: List[SubjectMetrics] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["subjects"] = [s.as_dict() for s in self.subjects]
        return d


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _counterfactual_error(window) -> Optional[float]:
    """Absolute error a window would have had, whether or not it answered.

    Cardiac features are computed before the abstain decision, so a refused
    window still carries a hypothetical prediction.
    """
    feats = getattr(window, "features", None)
    hr = getattr(feats, "hr_bpm", None) if feats is not None else None
    if hr is None or window.hr_gt_bpm is None:
        return None
    return abs(hr - window.hr_gt_bpm)


def subject_metrics(
    subject_id: str,
    windows: List,
    usable_error_bpm: float = 5.0,
    labels: Optional[Dict[str, object]] = None,
) -> SubjectMetrics:
    """Summarise one subject's windows.

    `usable_error_bpm` decides whether the recording is counted as usable at
    all: a session whose signal is not recoverable even in principle should not
    be scored as a coverage failure of the engine. This is an evaluation-time
    judgement -- it needs ground truth -- and it is the "usable-session rate"
    the explainer document lists among the success metrics.
    """
    paired = [
        (w.hr_pred_bpm, w.hr_gt_bpm)
        for w in windows
        if w.valid and w.hr_pred_bpm is not None and w.hr_gt_bpm is not None
    ]
    n_valid = sum(1 for w in windows if w.valid)
    coverage = n_valid / len(windows) if windows else 0.0
    snrs = [
        w.features.pulse_snr_db
        for w in windows
        if w.features is not None and w.features.pulse_snr_db is not None
    ]
    quals = [w.quality.overall for w in windows] if windows else []

    m = SubjectMetrics(
        subject_id=subject_id,
        labels=dict(labels or {}),
        n_windows=len(windows),
        n_valid=n_valid,
        coverage=coverage,
        mean_snr_db=float(np.mean(snrs)) if snrs else None,
        mean_quality=float(np.mean(quals)) if quals else None,
    )
    all_errors = [e for e in (_counterfactual_error(w) for w in windows) if e is not None]
    if all_errors:
        m.mae_all_bpm = float(np.mean(all_errors))
        m.usable = bool(m.mae_all_bpm <= usable_error_bpm)

    if paired:
        pred = np.array([p for p, _ in paired], float)
        gt = np.array([g for _, g in paired], float)
        err = pred - gt
        m.mae_bpm = float(np.mean(np.abs(err)))
        m.rmse_bpm = float(np.sqrt(np.mean(err ** 2)))
        m.pearson_r = _pearson(pred, gt)
    return m


def benchmark_metrics(
    method: str, dataset: str, per_subject: List[SubjectMetrics], notes=None
) -> BenchmarkMetrics:
    """Aggregate subject metrics into the cohort summary."""
    maes = [s.mae_bpm for s in per_subject if s.mae_bpm is not None]
    maes_all = [s.mae_all_bpm for s in per_subject if s.mae_all_bpm is not None]
    rmses = [s.rmse_bpm for s in per_subject if s.rmse_bpm is not None]
    rs = [s.pearson_r for s in per_subject if s.pearson_r is not None]
    snrs = [s.mean_snr_db for s in per_subject if s.mean_snr_db is not None]
    n_windows = sum(s.n_windows for s in per_subject)
    n_valid = sum(s.n_valid for s in per_subject)
    usable = [s for s in per_subject if s.usable]
    u_windows = sum(s.n_windows for s in usable)
    u_valid = sum(s.n_valid for s in usable)

    return BenchmarkMetrics(
        method=method,
        dataset=dataset,
        n_subjects=len(per_subject),
        n_windows=n_windows,
        coverage=(n_valid / n_windows) if n_windows else 0.0,
        median_mae_bpm=float(np.median(maes)) if maes else None,
        mean_mae_bpm=float(np.mean(maes)) if maes else None,
        # The worst subject is the honest number for a product that must work
        # for whoever opens the link, not for the median participant.
        worst_subject_mae_bpm=float(np.max(maes)) if maes else None,
        median_mae_all_bpm=float(np.median(maes_all)) if maes_all else None,
        usable_session_rate=(len(usable) / len(per_subject)) if per_subject else None,
        coverage_within_usable=(u_valid / u_windows) if u_windows else None,
        rmse_bpm=float(np.mean(rmses)) if rmses else None,
        pearson_r=float(np.mean(rs)) if rs else None,
        mean_snr_db=float(np.mean(snrs)) if snrs else None,
        subjects=per_subject,
        notes=list(notes or []),
    )


# --- Subgroup reporting ---------------------------------------------------

# Age bands for subgroup breakdown. Deliberately coarse: finer bands on a small
# cohort produce groups of two people and differences that mean nothing.
AGE_BANDS = ((18, 29, "18-29"), (30, 49, "30-49"), (50, 120, "50+"))

# Below this many subjects a subgroup result is reported but flagged, because
# a median over three people is not a finding.
MIN_SUBGROUP_N = 4


def age_band(age: Optional[float]) -> Optional[str]:
    if age is None:
        return None
    for lo, hi, name in AGE_BANDS:
        if lo <= age <= hi:
            return name
    return None


@dataclass
class SubgroupMetrics:
    """One subgroup's accuracy and coverage, with an explicit small-n flag."""

    key: str
    value: str
    n_subjects: int
    n_windows: int
    median_mae_all_bpm: Optional[float] = None
    worst_mae_all_bpm: Optional[float] = None
    coverage: Optional[float] = None
    usable_session_rate: Optional[float] = None
    underpowered: bool = False

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def subgroup_metrics(
    per_subject: List[SubjectMetrics], key: str
) -> List[SubgroupMetrics]:
    """Break results down by a recording label.

    Design doc section 14 lists skin tone and device bias as a project risk,
    and limitations section 7 recorded that fairness was asserted but never
    measured. Pooled metrics cannot show a subgroup failure -- the one found on
    SCAMPS (a dark-skinned subject producing no output at all) was invisible in
    every pooled number and only surfaced from a per-subject listing.
    """
    groups: Dict[str, List[SubjectMetrics]] = {}
    for s in per_subject:
        raw = s.labels.get("age") if key == "age_band" else s.labels.get(key)
        value = age_band(raw) if key == "age_band" else raw
        if value is None:
            continue
        groups.setdefault(str(value), []).append(s)

    out = []
    for value, members in sorted(groups.items()):
        maes = [m.mae_all_bpm for m in members if m.mae_all_bpm is not None]
        n_win = sum(m.n_windows for m in members)
        n_valid = sum(m.n_valid for m in members)
        out.append(
            SubgroupMetrics(
                key=key,
                value=value,
                n_subjects=len(members),
                n_windows=n_win,
                median_mae_all_bpm=float(np.median(maes)) if maes else None,
                worst_mae_all_bpm=float(np.max(maes)) if maes else None,
                coverage=(n_valid / n_win) if n_win else None,
                usable_session_rate=(
                    sum(1 for m in members if m.usable) / len(members)
                ),
                underpowered=len(members) < MIN_SUBGROUP_N,
            )
        )
    return out


def format_subgroups(rows: List[SubgroupMetrics]) -> str:
    if not rows:
        return "  (no subgroup labels available)"
    key = rows[0].key
    lines = [
        "  by {}:".format(key),
        "    {:<10} {:>5} {:>8} {:>10} {:>10} {:>8} {:>8}".format(
            "group", "subj", "windows", "MAE_med", "MAE_worst", "cover", "usable"
        ),
    ]
    for r in rows:
        f = lambda v, sp="{:>10.2f}": "       n/a" if v is None else sp.format(v)
        lines.append(
            "    {:<10} {:>5d} {:>8d} {} {} {:>7.0f}% {:>7.0f}%{}".format(
                r.value, r.n_subjects, r.n_windows,
                f(r.median_mae_all_bpm), f(r.worst_mae_all_bpm),
                100 * (r.coverage or 0.0), 100 * (r.usable_session_rate or 0.0),
                "  (small n)" if r.underpowered else "",
            )
        )
    return "\n".join(lines)
