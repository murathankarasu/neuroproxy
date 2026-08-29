"""Neural methods are optional and must not weaken the classical path."""
from __future__ import annotations

import numpy as np
import pytest

from neuroproxy.rppg.base import available_methods, get_method

torch = pytest.importorskip("torch", reason="neural methods need the torch extra")


def test_classical_methods_declare_no_frame_need():
    """The pipeline routes on this flag; a wrong default would break POS."""
    for name in ("pos", "chrom", "green"):
        assert get_method(name).needs_frames is False


def test_neural_methods_declare_a_frame_need():
    assert get_method("efficientphys_pure").needs_frames is True


def test_neural_method_rejects_a_colour_trace():
    """Passing an (T, 3) trace where frames are required must fail loudly."""
    m = get_method("efficientphys_pure")
    with pytest.raises(ValueError):
        m(np.zeros((600, 3)), 30.0)


def test_missing_checkpoint_names_the_file():
    """A missing weight file is a setup problem; say which one."""
    from neuroproxy.rppg.neural.adapter import _EfficientPhysBase

    class Missing(_EfficientPhysBase):
        name = "missing"
        checkpoint = "definitely_not_here.pth"

    with pytest.raises(FileNotFoundError, match="definitely_not_here"):
        Missing()(np.zeros((100, 72, 72, 3), np.uint8), 30.0)


def test_short_window_returns_silence_not_a_guess():
    """Fewer frames than the temporal shift depth cannot produce an estimate."""
    m = get_method("efficientphys_pure")
    out = m(np.zeros((5, 72, 72, 3), np.uint8), 30.0)
    assert out.shape == (5,)
    assert np.allclose(out, 0.0)


def test_pipeline_refuses_frame_method_without_crops():
    """A frame method with no crops must error, not silently use the trace."""
    from neuroproxy.pipeline.offline import Traces, analyze
    from training.datasets.synthetic import SyntheticConfig, generate

    rec = generate(SyntheticConfig(duration_s=25.0))
    with pytest.raises(ValueError, match="face crops"):
        analyze(
            rec,
            get_method("efficientphys_pure"),
            traces=Traces(
                rgb=np.zeros((750, 3)),
                timestamps=np.arange(750) / 30.0,
                quality=[],
                valid=np.ones(750, bool),
                fps=30.0,
                crops=None,
            ),
            window_s=20.0,
            stride_s=5.0,
        )
