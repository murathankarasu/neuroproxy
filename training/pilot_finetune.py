"""Three-arm pilot: POS vs pretrained EfficientPhys vs fine-tuned, held out.

Run as: python -m training.pilot_finetune
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from neuroproxy.rppg.base import get_method
from neuroproxy.rppg.signal import bandpass, hr_from_psd, normalize, welch_psd
from training.preprocess import estimate_label_lag, load_cache, subject_split
from training.train_efficientphys import (
    CHUNK,
    FRAME_DEPTH,
    TrainConfig,
    _standardize,
    finetune_spectral,
)

WINDOW_S = 20.0
STRIDE_S = 10.0


def _hr(signal: np.ndarray, fs: float):
    freqs, psd = welch_psd(signal, fs)
    return hr_from_psd(freqs, psd)


def _model_bvp(model, frames: np.ndarray, fs: float) -> np.ndarray:
    """Run a torch EfficientPhys over a window of crops."""
    import torch

    n = frames.shape[0]
    usable = ((n - 1) // FRAME_DEPTH) * FRAME_DEPTH
    if usable < FRAME_DEPTH:
        return np.zeros(n)
    x = _standardize(frames[: usable + 1])
    x = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)))
    with torch.no_grad():
        out = model(x).squeeze(-1).numpy().astype(np.float64)
    if out.size < n:
        out = np.concatenate([out, np.full(n - out.size, out[-1])])
    return normalize(bandpass(out[:n], fs))


def evaluate(clips, arms: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """Per-arm HR error over the same windows of the same held-out clips."""
    errors: Dict[str, List[float]] = {k: [] for k in arms}
    pos = get_method("pos")

    for clip in clips:
        fs = clip.fps
        win, stride = int(WINDOW_S * fs), int(STRIDE_S * fs)
        n = min(len(clip.crops), len(clip.bvp))
        for start in range(0, n - win + 1, stride):
            stop = start + win
            ref = _hr(bandpass(clip.bvp[start:stop].astype(np.float64), fs), fs)
            if ref is None:
                continue
            crops = clip.crops[start:stop]
            # POS needs a colour trace; take it from the same crops so every
            # arm sees exactly the same pixels.
            trace = crops.reshape(crops.shape[0], -1, 3).mean(axis=1)
            for name, arm in arms.items():
                bvp = pos(trace, fs) if arm is None else _model_bvp(arm, crops, fs)
                est = _hr(bvp, fs)
                if est is not None:
                    errors[name].append(abs(est - ref))

    out = {}
    for name, errs in errors.items():
        out[name] = {
            "n_windows": len(errs),
            "mae": float(np.mean(errs)) if errs else float("nan"),
            "median_ae": float(np.median(errs)) if errs else float("nan"),
            "p90_ae": float(np.percentile(errs, 90)) if errs else float("nan"),
            "within_5bpm": float(np.mean([e <= 5 for e in errs])) if errs else float("nan"),
        }
    return out


def load_lags(clips, path: Path = Path("data/cache/lags.json")) -> Dict[str, float]:
    """Per-recording video/PPG offsets, computed once and cached.

    Without these the training target is out of phase with the input on most
    recordings -- see docs/limitations.md section 21.
    """
    known = {}
    if path.exists():
        known = json.loads(path.read_text())
    missing = [c for c in clips if c.session not in known]
    if missing:
        print("estimating label lag for {} recordings...".format(len(missing)))
        for i, clip in enumerate(missing, 1):
            known[clip.session] = float(estimate_label_lag(clip))
            if i % 20 == 0:
                print("  {}/{}".format(i, len(missing)))
        path.write_text(json.dumps(known, indent=2))
    lags = np.array([known[c.session] for c in clips])
    print("lags: median {:+.2f}s  |lag|<0.2s in {}/{}  |lag|>1s in {}/{}".format(
        float(np.median(lags)), int((np.abs(lags) < 0.2).sum()), len(lags),
        int((np.abs(lags) > 1.0).sum()), len(lags)))
    return known


def main() -> int:
    clips = load_cache(Path("data/cache"), max_frames=900)
    if len(clips) < 10:
        print("only {} cached clips; run training.preprocess first".format(len(clips)))
        return 1

    # Three-way subject split. Test is scored once, at the end.
    trainval, test_clips, test_subjects = subject_split(clips, test_fraction=0.30, seed=7)
    train_clips, val_clips, val_subjects = subject_split(trainval, test_fraction=0.25, seed=3)

    print("clips {}  |  train subj {}  val {}  test {}".format(
        len(clips),
        len({c.subject_id for c in train_clips}),
        len(val_subjects), len(test_subjects)))
    print("test subjects (scored once):", " ".join(test_subjects))
    print()


    from neuroproxy.rppg.neural.adapter import MODEL_DIR
    from neuroproxy.rppg.neural.efficientphys import load_pretrained

    pretrained = load_pretrained(MODEL_DIR / "PURE_EfficientPhys.pth")
    print("fine-tuning, phase-invariant spectral objective (subject-independent):")
    tuned, result = finetune_spectral(
        train_clips, val_clips,
        cfg=TrainConfig(epochs=8),
        out_path=Path("models/MCD_EfficientPhys_spectral.pth"),
    )
    print()

    scores = evaluate(test_clips, {
        "pos": None,
        "efficientphys_pretrained": pretrained,
        "efficientphys_spectral": tuned,
    })

    print("held-out test subjects ({} of them), HR error in bpm".format(len(test_subjects)))
    hdr = "{:<28} {:>8} {:>9} {:>9} {:>9} {:>10}".format(
        "arm", "windows", "MAE", "median", "p90", "within5bpm")
    print(hdr); print("-" * len(hdr))
    for name in ("pos", "efficientphys_pretrained", "efficientphys_spectral"):
        s = scores[name]
        print("{:<28} {:>8d} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.0f}%".format(
            name, s["n_windows"], s["mae"], s["median_ae"], s["p90_ae"],
            100 * s["within_5bpm"]))

    payload = {"result": result.as_dict(), "scores": scores,
               "test_subjects": test_subjects}
    Path("results").mkdir(exist_ok=True)
    Path("results/pilot_spectral.json").write_text(json.dumps(payload, indent=2))
    print("\nwrote results/pilot_finetune.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
