"""Benchmark harness: methods x datasets -> a comparable metrics table.

Design doc go/no-go criteria (section 13.2) are encoded here so a run either
passes or fails explicitly, rather than producing numbers someone has to
interpret from memory.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from neuroproxy.pipeline.offline import (
    DEFAULT_CROP_SIZE,
    analyze,
    extract_traces,
)
from neuroproxy.rppg.base import get_method

from ..datasets.base import Dataset
from .calibration import CalibrationReport, evaluate as evaluate_calibration
from .metrics import BenchmarkMetrics, benchmark_metrics, subject_metrics

# Design doc section 13.2 engineering thresholds. These are targets, not claims.
GO_NO_GO = {
    "median_mae_bpm": 5.0,     # median subject HR MAE must be at or below this
    "coverage": 0.80,          # valid-window rate in controlled conditions
    # Confidence must rank errors better than chance, or "abstain when the
    # signal is bad" is a slogan rather than a feature.
    "capture_ratio": 0.25,
}


def counterfactual_error(window) -> Optional[float]:
    """Absolute HR error a window *would* have had, including if it abstained.

    The pipeline computes cardiac features before deciding whether to answer,
    so a refused window still has a hypothetical prediction. Comparing those
    against the answered ones is the only way to check that abstention removed
    bad windows rather than arbitrary ones.
    """
    hr = window.features.hr_bpm if window.features is not None else None
    if hr is None or window.hr_gt_bpm is None:
        return None
    return abs(hr - window.hr_gt_bpm)


def calibration_for(windows: List) -> CalibrationReport:
    """Risk-coverage report over every window that has a ground-truth pair."""
    kept_conf, kept_err, abstained_err = [], [], []
    for w in windows:
        err = counterfactual_error(w)
        if err is None:
            continue
        if w.valid:
            kept_conf.append(w.confidence)
            kept_err.append(err)
        else:
            abstained_err.append(err)
    return evaluate_calibration(kept_conf, kept_err, abstained_errors=abstained_err)


@dataclass
class BenchmarkRun:
    dataset: str
    window_s: float
    stride_s: float
    results: List[BenchmarkMetrics] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "window_s": self.window_s,
            "stride_s": self.stride_s,
            "elapsed_s": self.elapsed_s,
            "notes": self.notes,
            "results": [r.as_dict() for r in self.results],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, default=str))
        return path


def run_benchmark(
    dataset: Dataset,
    methods: Sequence[str],
    window_s: float = 20.0,
    stride_s: float = 1.0,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> BenchmarkRun:
    """Evaluate each method over the same recordings and windows.

    Traces are extracted once per recording and reused across methods, so the
    comparison isolates the rPPG algebra rather than detector jitter.
    """
    run = BenchmarkRun(dataset=dataset.name, window_s=window_s, stride_s=stride_s)
    t0 = time.time()

    if not dataset.is_available():
        run.notes.append(dataset.unavailable_reason())
        run.elapsed_s = time.time() - t0
        return run

    recordings = dataset.recordings(limit=limit)
    if not recordings:
        run.notes.append("dataset {!r} produced no recordings".format(dataset.name))
        run.elapsed_s = time.time() - t0
        return run

    if not any(r.has_gt for r in recordings):
        run.notes.append(
            "no contact ground truth in this dataset; HR error metrics unavailable"
        )

    per_method: Dict[str, List] = {m: [] for m in methods}
    for rec in recordings:
        if verbose:
            print("  [{}] {:.0f}s @ {:.0f}fps".format(rec.subject_id, rec.duration_s, rec.fps))
        needs_crops = any(
            getattr(get_method(n), "needs_frames", False) for n in methods
        )
        traces = extract_traces(
            rec, crop_size=DEFAULT_CROP_SIZE if needs_crops else None
        )
        if traces.valid_ratio < 0.5:
            run.notes.append(
                "{}: face found in only {:.0%} of frames".format(
                    rec.subject_id, traces.valid_ratio
                )
            )
        for name in methods:
            method = get_method(name)
            windows = analyze(
                rec, method, traces=traces, window_s=window_s, stride_s=stride_s
            )
            per_method[name].append(
                subject_metrics(rec.subject_id, windows, labels=rec.labels)
            )

    for name in methods:
        run.results.append(benchmark_metrics(name, dataset.name, per_method[name]))

    # Rank on the abstain-independent error: sorting by answered-only MAE would
    # reward a method for refusing its hard windows.
    run.results.sort(
        key=lambda r: (r.median_mae_all_bpm if r.median_mae_all_bpm is not None else 1e9)
    )
    run.elapsed_s = time.time() - t0
    return run


def check_go_no_go(m: BenchmarkMetrics) -> Dict[str, object]:
    """Evaluate one method against the design doc thresholds.

    The MAE bar is checked against the abstain-independent error. Checking the
    answered-only error would let a method pass by refusing its hard windows --
    the coverage bar limits that, but it does not remove the incentive, and a
    threshold that can be gamed is not a threshold.
    """
    checks = {}
    mae_all = m.median_mae_all_bpm
    checks["median_mae_all_bpm"] = {
        "value": mae_all,
        "threshold": GO_NO_GO["median_mae_bpm"],
        "pass": bool(mae_all is not None and mae_all <= GO_NO_GO["median_mae_bpm"]),
    }
    checks["coverage"] = {
        "value": m.coverage,
        "threshold": GO_NO_GO["coverage"],
        "pass": bool(m.coverage >= GO_NO_GO["coverage"]),
    }
    checks["overall_pass"] = all(c["pass"] for c in checks.values() if isinstance(c, dict))
    return checks


def format_table(run: BenchmarkRun) -> str:
    """Render a run as a fixed-width table for the terminal."""
    header = (
        "{:<8} {:>6} {:>8} {:>9} {:>9} {:>9} {:>7} {:>8} {:>6}".format(
            "method", "subj", "windows", "MAE_all", "MAE_ans", "MAE_worst",
            "r", "SNR_dB", "cover",
        )
    )
    lines = [header, "-" * len(header)]

    def fmt(v, spec="{:>9.2f}", width=9):
        return ("{:>" + str(width) + "}").format("n/a") if v is None else spec.format(v)

    for m in run.results:
        lines.append(
            "{:<8} {:>6d} {:>8d} {} {} {} {} {} {:>5.0f}%".format(
                m.method,
                m.n_subjects,
                m.n_windows,
                fmt(m.median_mae_all_bpm),
                fmt(m.median_mae_bpm),
                fmt(m.worst_subject_mae_bpm),
                fmt(m.pearson_r, "{:>7.3f}", 7),
                fmt(m.mean_snr_db, "{:>8.1f}", 8),
                m.coverage * 100.0,
            )
        )

    if run.results:
        lines.append("")
        lines.append(
            "MAE_all = every window with ground truth, including refused ones "
            "(abstain-independent)."
        )
        lines.append(
            "MAE_ans = only windows the engine answered; improves as coverage "
            "falls, so read it with cover."
        )
        lines.append("")
        best = run.results[0]
        if best.usable_session_rate is not None:
            lines.append(
                "coverage decomposes as: usable sessions {:.0%} x coverage within "
                "them {:.0%}".format(
                    best.usable_session_rate, best.coverage_within_usable or 0.0
                )
            )
            lines.append(
                "  a session counts as unusable when its signal is not recoverable "
                "at all, so refusing it was correct"
            )
            lines.append("")
        from .metrics import format_subgroups, subgroup_metrics

        for key in ("sex", "age_band", "step"):
            rows = subgroup_metrics(best.subjects, key)
            if rows:
                lines.append(format_subgroups(rows))
                lines.append("")
        lines.append("go/no-go (design doc 13.2), best method '{}':".format(best.method))
        checks = check_go_no_go(run.results[0])
        for key in ("median_mae_all_bpm", "coverage"):
            c = checks[key]
            val = "n/a" if c["value"] is None else "{:.2f}".format(c["value"])
            lines.append(
                "  {:<20} {:>8}  target {:<6}  {}".format(
                    key, val, c["threshold"], "PASS" if c["pass"] else "FAIL"
                )
            )
    for note in run.notes:
        lines.append("note: {}".format(note))
    return "\n".join(lines)
