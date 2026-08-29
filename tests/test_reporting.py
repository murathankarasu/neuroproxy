"""Reported accuracy must not be purchasable with coverage.

Abstention removes hard windows, so any metric computed only over answered
windows improves as the engine refuses more. A benchmark that ranks or gates on
that metric rewards refusing, which is the opposite of what it is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from neuroproxy.features.cardiac import CardiacFeatures
from neuroproxy.quality.metrics import WindowQuality
from training.evaluation.harness import check_go_no_go
from training.evaluation.metrics import benchmark_metrics, subject_metrics


@dataclass
class FakeWindow:
    hr_pred_bpm: Optional[float]
    hr_gt_bpm: Optional[float]
    valid: bool
    features: CardiacFeatures = field(default_factory=CardiacFeatures)
    quality: WindowQuality = field(default_factory=WindowQuality)
    confidence: float = 0.5

    @property
    def abs_error(self):
        if self.hr_pred_bpm is None or self.hr_gt_bpm is None:
            return None
        return abs(self.hr_pred_bpm - self.hr_gt_bpm)


def _windows(errors, answered):
    """Build windows with given errors; `answered` marks which ones replied."""
    out = []
    for err, ans in zip(errors, answered):
        pred = 70.0 + err
        out.append(
            FakeWindow(
                hr_pred_bpm=pred if ans else None,
                hr_gt_bpm=70.0,
                valid=ans,
                features=CardiacFeatures(hr_bpm=pred),
            )
        )
    return out


def test_mae_all_is_unchanged_by_abstaining_more():
    """The abstain-independent error must ignore the refusal policy."""
    errors = [0.5, 0.5, 20.0, 20.0]
    lenient = subject_metrics("s", _windows(errors, [True] * 4))
    strict = subject_metrics("s", _windows(errors, [True, True, False, False]))
    assert lenient.mae_all_bpm == pytest.approx(strict.mae_all_bpm)
    # ...while the answered-only number improves dramatically, as expected.
    assert strict.mae_bpm < lenient.mae_bpm
    assert strict.coverage < lenient.coverage


def test_go_no_go_cannot_be_passed_by_refusing():
    """A method whose signal is bad must fail even if it only answers easy windows."""
    errors = [0.5, 0.5, 40.0, 40.0]
    strict = subject_metrics("s", _windows(errors, [True, True, False, False]))
    m = benchmark_metrics("pos", "fake", [strict])
    checks = check_go_no_go(m)
    assert m.median_mae_bpm < 5.0          # looks fine on answered windows
    assert m.median_mae_all_bpm > 5.0      # but the signal is not fine
    assert not checks["median_mae_all_bpm"]["pass"]
    assert not checks["overall_pass"]


def test_unmeasurable_session_is_not_counted_as_a_coverage_failure():
    """Pooled coverage conflates two different failures; they must separate.

    A session whose signal is not recoverable at all and a session where a few
    windows were dropped are different problems with different fixes. Measured
    on SCAMPS: pooled coverage 75% decomposed into an 80% usable-session rate
    and 93% coverage inside those sessions.
    """
    from training.evaluation.metrics import benchmark_metrics

    good = subject_metrics("good", _windows([0.4, 0.4, 0.5, 0.5], [True] * 4))
    broken = subject_metrics("broken", _windows([60.0] * 4, [False] * 4))
    assert good.usable
    assert not broken.usable

    m = benchmark_metrics("pos", "fake", [good, broken])
    assert m.coverage == pytest.approx(0.5)             # pooled: half refused
    assert m.usable_session_rate == pytest.approx(0.5)  # one session unusable
    assert m.coverage_within_usable == pytest.approx(1.0)  # the good one is complete
