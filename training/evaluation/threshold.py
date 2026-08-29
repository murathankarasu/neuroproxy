"""Derive the abstain threshold from data instead of choosing a constant.

The threshold decides what fraction of sessions a customer gets an answer for,
so picking it by intuition is picking the product's coverage by intuition. This
sweeps it and reports what each level costs and buys.

The right threshold sits just above the highest confidence ever observed on a
window whose error exceeds the acceptable bound. Above that point, raising it
further buys no accuracy and only destroys coverage -- which is exactly what
the initial hand-picked value of 0.45 was doing on SCAMPS: identical MAE to
0.20, with 24 percentage points less coverage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18,
                0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


@dataclass
class ThresholdPoint:
    threshold: float
    coverage: float
    mae: float
    p90: float
    max_error: float


@dataclass
class ThresholdReport:
    n_windows: int
    acceptable_error_bpm: float
    points: List[ThresholdPoint] = field(default_factory=list)
    # Highest confidence seen on a window whose error exceeded the bound.
    worst_bad_confidence: Optional[float] = None
    n_bad: int = 0
    recommended: Optional[float] = None
    coverage_at_recommended: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["points"] = [asdict(p) for p in self.points]
        return d


def sweep(
    confidences: Sequence[float],
    errors: Sequence[float],
    acceptable_error_bpm: float = 5.0,
    margin: float = 0.04,
    grid: Optional[Sequence[float]] = None,
) -> ThresholdReport:
    """Sweep the abstain threshold over (confidence, counterfactual error) pairs.

    `errors` must be counterfactual -- the error a window *would* have had --
    including windows the engine refused. Sweeping over answered windows only
    would show the effect of a threshold that has already been applied.
    """
    conf = np.asarray(confidences, dtype=float)
    err = np.asarray(errors, dtype=float)
    keep = np.isfinite(conf) & np.isfinite(err)
    conf, err = conf[keep], err[keep]

    report = ThresholdReport(
        n_windows=int(conf.size), acceptable_error_bpm=acceptable_error_bpm
    )
    if conf.size < 20:
        report.notes.append(
            "only {} windows; a threshold fitted on this little data is a "
            "guess with extra steps".format(conf.size)
        )
        return report

    for th in (grid or DEFAULT_GRID):
        sel = conf >= th
        if not sel.any():
            continue
        report.points.append(
            ThresholdPoint(
                threshold=float(th),
                coverage=float(sel.mean()),
                mae=float(err[sel].mean()),
                p90=float(np.percentile(err[sel], 90)),
                max_error=float(err[sel].max()),
            )
        )

    bad = err > acceptable_error_bpm
    report.n_bad = int(bad.sum())
    if report.n_bad == 0:
        report.notes.append(
            "no window exceeded {:.0f} bpm error, so this cohort cannot "
            "locate the threshold".format(acceptable_error_bpm)
        )
        return report

    report.worst_bad_confidence = float(conf[bad].max())
    recommended = report.worst_bad_confidence + margin
    report.recommended = float(recommended)
    sel = conf >= recommended
    report.coverage_at_recommended = float(sel.mean()) if sel.size else 0.0

    if (conf[~bad] >= recommended).mean() < 0.5:
        report.notes.append(
            "the recommendation refuses more than half of the acceptable "
            "windows too; confidence separates poorly on this cohort"
        )
    return report


def format_report(r: ThresholdReport, current: Optional[float] = None) -> str:
    lines = [
        "abstain threshold sweep ({} windows, acceptable error <= {:.0f} bpm)".format(
            r.n_windows, r.acceptable_error_bpm
        ),
        "",
        "{:>10} {:>10} {:>9} {:>9} {:>9}".format(
            "threshold", "coverage", "MAE", "p90", "max"
        ),
        "-" * 50,
    ]
    for p in r.points:
        mark = ""
        if current is not None and abs(p.threshold - current) < 1e-9:
            mark = "  <- current"
        if r.recommended is not None and abs(p.threshold - r.recommended) < 0.02:
            mark += "  <- near recommended"
        lines.append(
            "{:>10.2f} {:>9.0f}% {:>9.2f} {:>9.2f} {:>9.2f}{}".format(
                p.threshold, 100 * p.coverage, p.mae, p.p90, p.max_error, mark
            )
        )
    if r.recommended is not None:
        lines.append("")
        lines.append(
            "  {} of {} windows exceeded the error bound; the highest confidence "
            "any of them reached was {:.3f}".format(
                r.n_bad, r.n_windows, r.worst_bad_confidence
            )
        )
        lines.append(
            "  recommended threshold {:.2f}  ->  coverage {:.0%}".format(
                r.recommended, r.coverage_at_recommended or 0.0
            )
        )
        lines.append(
            "  NOTE: this is fitted to this cohort. Re-derive it on every new "
            "dataset before shipping it."
        )
    for n in r.notes:
        lines.append("  note: " + n)
    return "\n".join(lines)
