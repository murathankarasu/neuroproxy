"""Does confidence actually predict error?

The explainer document lists "confidence-first design: abstain when the signal
is bad" as a core differentiator, and the design doc's success metrics include
the relationship between low quality and error. Both are claims about
*ordering*: a useful confidence score must rank a window we get wrong below a
window we get right.

This module tests that claim with a risk-coverage analysis, the standard tool
for selective prediction:

  - Sort windows by confidence, highest first.
  - At each coverage level, keep that fraction and measure MAE on the kept set.
  - A useful score produces MAE that rises monotonically with coverage.

The number that matters is not the curve's absolute height but how it compares
to two references computed on the same windows:

  random  -- shuffle the ordering. Any score must beat this.
  oracle  -- sort by the true absolute error. Nothing can beat this.

`capture_ratio` reports where confidence sits between them, in [0, 1]. A score
near 0 is worthless regardless of how good the curve looks on its own, because
the curve's shape is mostly determined by the error distribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# Coverage levels at which risk is reported.
COVERAGE_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


@dataclass
class CalibrationReport:
    n_windows: int
    spearman_rho: Optional[float] = None      # confidence vs |error|; want < 0
    spearman_p: Optional[float] = None
    aurc: Optional[float] = None              # area under risk-coverage curve
    aurc_random: Optional[float] = None
    aurc_oracle: Optional[float] = None
    capture_ratio: Optional[float] = None     # 0 = random, 1 = oracle
    risk_at_coverage: Dict[str, float] = field(default_factory=dict)
    decile_mae: List[Dict[str, float]] = field(default_factory=list)
    abstain_rate: float = 0.0
    mae_kept: Optional[float] = None
    mae_abstained: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _risk_coverage(order: np.ndarray, errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative mean error along a given ordering, as coverage grows."""
    sorted_err = errors[order]
    cum = np.cumsum(sorted_err) / np.arange(1, sorted_err.size + 1)
    coverage = np.arange(1, sorted_err.size + 1) / sorted_err.size
    return coverage, cum


def _aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    return float(np.trapz(risk, coverage))


def evaluate(
    confidences: Sequence[float],
    errors: Sequence[float],
    abstained_errors: Optional[Sequence[float]] = None,
    n_random: int = 200,
    seed: int = 0,
) -> CalibrationReport:
    """Risk-coverage analysis of a confidence score against realised errors.

    `errors` are absolute HR errors for windows the engine chose to answer.
    `abstained_errors` are the errors it *would* have made on windows it
    refused -- available only in offline evaluation, and the only way to check
    that abstention removed bad windows rather than arbitrary ones.
    """
    conf = np.asarray(confidences, dtype=float)
    err = np.asarray(errors, dtype=float)
    keep = np.isfinite(conf) & np.isfinite(err)
    conf, err = conf[keep], err[keep]

    report = CalibrationReport(n_windows=int(conf.size))
    if conf.size < 10:
        report.notes.append("fewer than 10 comparable windows; calibration not assessed")
        return report

    if np.allclose(conf, conf[0]):
        report.notes.append(
            "confidence is constant across windows; it carries no ordering "
            "information and cannot be calibrated"
        )
        return report

    rho, p = stats.spearmanr(conf, err)
    report.spearman_rho = float(rho)
    report.spearman_p = float(p)

    order = np.argsort(-conf, kind="stable")
    coverage, risk = _risk_coverage(order, err)
    report.aurc = _aurc(coverage, risk)

    oracle_order = np.argsort(err, kind="stable")
    _, oracle_risk = _risk_coverage(oracle_order, err)
    report.aurc_oracle = _aurc(coverage, oracle_risk)

    rng = np.random.default_rng(seed)
    randoms = []
    for _ in range(n_random):
        _, r = _risk_coverage(rng.permutation(err.size), err)
        randoms.append(_aurc(coverage, r))
    report.aurc_random = float(np.mean(randoms))

    span = report.aurc_random - report.aurc_oracle
    if span > 1e-12:
        report.capture_ratio = float(
            np.clip((report.aurc_random - report.aurc) / span, -1.0, 1.0)
        )

    for c in COVERAGE_GRID:
        idx = max(int(round(c * err.size)) - 1, 0)
        report.risk_at_coverage["{:.0%}".format(c)] = float(risk[idx])

    # Confidence deciles, ordered worst-confidence first, so a working score
    # shows MAE falling down the table.
    deciles = np.array_split(np.argsort(conf, kind="stable"), 10)
    for i, chunk in enumerate(deciles):
        if chunk.size == 0:
            continue
        report.decile_mae.append(
            {
                "decile": i + 1,
                "conf_mean": float(conf[chunk].mean()),
                "mae": float(err[chunk].mean()),
                "n": int(chunk.size),
            }
        )

    report.mae_kept = float(err.mean())
    if abstained_errors is not None:
        ab = np.asarray(abstained_errors, dtype=float)
        ab = ab[np.isfinite(ab)]
        total = err.size + ab.size
        report.abstain_rate = float(ab.size / total) if total else 0.0
        if ab.size:
            report.mae_abstained = float(ab.mean())
            if report.mae_abstained <= report.mae_kept:
                report.notes.append(
                    "abstained windows were no worse than kept windows "
                    "({:.2f} vs {:.2f} bpm): abstention is discarding usable data"
                    .format(report.mae_abstained, report.mae_kept)
                )
    return report


def format_report(r: CalibrationReport) -> str:
    """Render a calibration report for the terminal."""
    lines = ["confidence calibration ({} comparable windows)".format(r.n_windows)]
    if r.n_windows < 10 or r.spearman_rho is None:
        lines.extend("  note: " + n for n in r.notes)
        return "\n".join(lines)

    lines.append(
        "  spearman(confidence, |error|)  {:+.3f}  (p={:.2g})   want clearly negative".format(
            r.spearman_rho, r.spearman_p
        )
    )
    lines.append(
        "  AURC  score {:.3f}   random {:.3f}   oracle {:.3f}".format(
            r.aurc, r.aurc_random, r.aurc_oracle
        )
    )
    if r.capture_ratio is not None:
        verdict = (
            "no better than random"
            if r.capture_ratio < 0.05
            else "weak" if r.capture_ratio < 0.25
            else "useful" if r.capture_ratio < 0.6
            else "strong"
        )
        lines.append(
            "  capture ratio  {:+.3f}  (0 = random, 1 = oracle)  -> {}".format(
                r.capture_ratio, verdict
            )
        )
    lines.append("")
    lines.append("  risk-coverage (MAE bpm of the most-confident N%):")
    row = "    " + "  ".join("{:>6}".format(k) for k in r.risk_at_coverage)
    val = "    " + "  ".join("{:>6.2f}".format(v) for v in r.risk_at_coverage.values())
    lines.extend([row, val])

    if r.decile_mae:
        lines.append("")
        lines.append("  confidence deciles (worst first):")
        lines.append("    {:>7} {:>10} {:>8} {:>5}".format("decile", "conf_mean", "MAE", "n"))
        for d in r.decile_mae:
            lines.append(
                "    {:>7} {:>10.3f} {:>8.2f} {:>5}".format(
                    d["decile"], d["conf_mean"], d["mae"], d["n"]
                )
            )
    if r.abstain_rate:
        lines.append("")
        lines.append(
            "  abstained on {:.1%} of windows; MAE kept {:.2f} vs abstained {}".format(
                r.abstain_rate,
                r.mae_kept if r.mae_kept is not None else float("nan"),
                "n/a" if r.mae_abstained is None else "{:.2f}".format(r.mae_abstained),
            )
        )
    for n in r.notes:
        lines.append("  note: " + n)
    return "\n".join(lines)
