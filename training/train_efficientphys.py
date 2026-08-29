"""Fine-tune EfficientPhys on first-party data, and check it earned it.

WHY THIS EXISTS
---------------
Pretrained EfficientPhys collapses across domains: 31.7 bpm error on MCD-rPPG
against POS's 2.87 (docs/limitations.md section 19). In-domain, the same
architecture beats POS on both error and coverage. Section 18 separately
measured ~4.5x of headroom in spatial ROI weighting that hand-built heuristics
could not reach. All three point the same way: the architecture can help, but
only trained on the distribution it will serve.

This is the pilot that decides whether to invest in that. It runs on the 25
subjects already downloaded rather than the full 600, because a fine-tune that
cannot move the needle on 25 will not be rescued by 600 -- and finding that out
costs an hour instead of a week.

THE COMPARISON THAT MATTERS
---------------------------
Three arms on the *same* held-out subjects:

    POS                      no learned parameters, the incumbent
    EfficientPhys pretrained the design doc's Phase A starting point
    EfficientPhys fine-tuned this

Reporting only the third against the first would confound "fine-tuning helped"
with "the architecture helps". The middle arm separates them.

SPLIT DISCIPLINE
----------------
Three-way, by subject, never by clip (design doc section 8.1). Validation
subjects drive early stopping; test subjects are scored once at the end. A
subject's rest and post-exercise recordings always travel together -- splitting
them would put the same face on both sides.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CHUNK = 180          # 6 s at 30 fps; a multiple of the model's frame_depth
FRAME_DEPTH = 20


def negative_pearson(pred, target):
    """1 - r, the standard rPPG objective.

    Scale-invariant on purpose: an rPPG waveform has no meaningful amplitude,
    so training on MSE would spend capacity fitting a quantity that carries no
    information.
    """
    import torch

    pred = pred - pred.mean()
    target = target - target.mean()
    denom = torch.sqrt((pred ** 2).sum() * (target ** 2).sum()) + 1e-8
    return 1.0 - (pred * target).sum() / denom


HR_BAND_HZ = (0.7, 3.0)
# Zero-padding does not add information, but it makes the spectral target
# smooth enough to give useful gradients at 6-10 s chunk lengths.
SPECTRUM_PAD = 4
# Width of the soft target around the true rate. Narrower than this and the
# loss becomes a near-one-hot target on a coarse frequency grid.
HR_SIGMA_BPM = 3.0


def hr_from_label(bvp: np.ndarray, fs: float) -> Optional[float]:
    """Reference heart rate from the contact PPG.

    Taken from the spectrum, which is exactly the part of this label that is
    trustworthy: MCD-rPPG's PPG is aligned in rate but not in phase
    (docs/limitations.md section 22).
    """
    from neuroproxy.rppg.signal import bandpass, hr_from_psd, welch_psd

    x = np.asarray(bvp, dtype=np.float64)
    if x.size < int(fs * 4):
        return None
    freqs, psd = welch_psd(bandpass(x, fs), fs)
    return hr_from_psd(freqs, psd)


def spectral_cross_entropy(pred, hr_bpm: float, fs: float):
    """Phase-invariant loss: put the predicted spectrum's mass at the true rate.

    WHY NOT CORRELATION. The waveform objective the architecture ships with
    cannot be trained on this data: the label's phase is inconsistent between
    chunks by whole pulse periods, so nearly identical inputs carry targets a
    full cycle apart, and training plateaus at r = 0.035 (limitations 22).

    A loss on the power spectrum is invariant to exactly that. It also happens
    to optimise the quantity the product actually reports -- heart rate --
    rather than a pulse phase nothing downstream consumes.

    The target is a Gaussian around the reference rate rather than a single
    bin: at these chunk lengths the frequency grid is coarse, and a one-hot
    target on a coarse grid mostly teaches the model to be confident.
    """
    import torch
    import torch.nn.functional as F

    n = pred.shape[0]
    p = pred - pred.mean()
    window = torch.hann_window(n, dtype=p.dtype, device=p.device)
    spec = torch.fft.rfft(p * window, n=n * SPECTRUM_PAD)
    psd = spec.real ** 2 + spec.imag ** 2

    freqs = torch.fft.rfftfreq(n * SPECTRUM_PAD, d=1.0 / fs).to(p.device)
    mask = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])
    band_psd = psd[mask]
    band_bpm = freqs[mask] * 60.0

    logits = torch.log(band_psd + 1e-12)
    target = torch.exp(-0.5 * ((band_bpm - hr_bpm) / HR_SIGMA_BPM) ** 2)
    target = target / (target.sum() + 1e-12)
    return -(target * F.log_softmax(logits, dim=0)).sum()


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sd = float(x.std())
    return np.zeros_like(x) if sd < 1e-8 else (x - float(x.mean())) / sd


def diff_normalized(bvp: np.ndarray) -> np.ndarray:
    """First difference of the label, standardised.

    THE MODEL PREDICTS A DERIVATIVE, NOT A WAVEFORM. EfficientPhys applies
    `torch.diff` to its input and is trained on `DiffNormalized` labels; its
    output is the frame-to-frame change in blood volume, not the pulse itself.

    Training it against the raw waveform is not a small mismatch, it is an
    orthogonal target: for a sinusoid, the correlation between the signal and
    its derivative is exactly zero. An earlier pilot did precisely that and
    reported that fine-tuning "failed"; the loop could not overfit four chunks,
    which is the signature of a bug rather than of data scarcity
    (docs/limitations.md section 21).

    Inference is unaffected -- differentiating a sinusoid preserves its
    frequency, so the PSD peak, and therefore the heart rate, is unchanged.
    Only the training target was wrong.
    """
    d = np.diff(np.asarray(bvp, dtype=np.float32))
    return _standardize(d)


def make_spectral_chunks(
    clips, chunk: int = 300, stride: Optional[int] = None
) -> List[Tuple[np.ndarray, float]]:
    """Chunks labelled with a reference heart rate rather than a waveform.

    No lag correction is needed or possible here -- that is the point. A rate
    is invariant to the offset the labels cannot pin down.
    """
    out = []
    for clip in clips:
        n = min(len(clip.crops), len(clip.bvp))
        for start in range(0, n - chunk - 1, stride or chunk):
            frames = clip.crops[start : start + chunk + 1]
            hr = hr_from_label(clip.bvp[start : start + chunk + 1], clip.fps)
            if frames.shape[0] != chunk + 1 or hr is None:
                continue
            out.append((frames, float(hr)))
    return out


def make_chunks(
    clips, chunk: int = CHUNK, lags: Optional[Dict[str, float]] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Cut cached clips into fixed-length training chunks.

    Each chunk carries chunk+1 frames because the model differences internally
    and so consumes one more frame than it emits.

    `lags` shifts each recording's label by its measured video/PPG offset.
    Without it the target is out of phase with the input on most recordings and
    the model cannot learn at all -- see `preprocess.estimate_label_lag`.
    """
    out = []
    for clip in clips:
        shift = int(round((lags or {}).get(clip.session, 0.0) * clip.fps))
        n = min(len(clip.crops), len(clip.bvp))
        for start in range(0, n - chunk - 1, chunk):
            frames = clip.crops[start : start + chunk + 1]
            # chunk+1 raw samples difference down to chunk, matching the
            # model's own internal diff over chunk+1 frames.
            lo = start + shift
            if lo < 0 or lo + chunk + 1 > len(clip.bvp):
                continue
            raw = clip.bvp[lo : lo + chunk + 1]
            if frames.shape[0] != chunk + 1 or raw.shape[0] != chunk + 1:
                continue
            out.append((frames, diff_normalized(raw)))
    return out


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_chunks: int = 2
    # Design doc section 5.3 proposes encoder 1e-5 / head 1e-3. THAT SETTING
    # DOES NOT TRAIN. A first pilot used it, reported that fine-tuning made
    # things worse, and was wrong: at 1e-5 the encoder barely moves, and six
    # epochs over 65 chunks is roughly 195 optimiser updates in total.
    #
    # The check that settled it: a single 180-frame chunk, dropout off. The
    # loop memorises it completely -- correlation 0.03 -> 0.94 by step 50 and
    # 0.998 by step 200 -- so the machinery was never broken, only starved of
    # step size and steps. See docs/limitations.md section 21.
    #
    # The encoder still moves more slowly than the head, which is the sound
    # part of the design doc's reasoning; it just needs to move at all.
    lr_encoder: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    checkpoint: str = "PURE_EfficientPhys.pth"


@dataclass
class TrainResult:
    epochs_run: int = 0
    best_epoch: int = 0
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_subjects: List[str] = field(default_factory=list)
    val_subjects: List[str] = field(default_factory=list)
    test_subjects: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _param_groups(model, cfg: TrainConfig):
    head, encoder = [], []
    for name, p in model.named_parameters():
        (head if name.startswith("final_dense") else encoder).append(p)
    return [
        {"params": encoder, "lr": cfg.lr_encoder},
        {"params": head, "lr": cfg.lr_head},
    ]


def _eval_loss(model, chunks, device) -> float:
    import torch

    if not chunks:
        return float("nan")
    model.eval()
    total = 0.0
    with torch.no_grad():
        for frames, label in chunks:
            x = _standardize(frames)
            x = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).to(device)
            y = torch.from_numpy(label).to(device)
            pred = model(x).squeeze(-1)
            total += float(negative_pearson(pred, y))
    return total / len(chunks)


def finetune(
    train_clips,
    val_clips,
    cfg: Optional[TrainConfig] = None,
    lags: Optional[Dict[str, float]] = None,
    out_path: Optional[Path] = None,
    verbose: bool = True,
):
    """Fine-tune from a released checkpoint; return (model, TrainResult)."""
    import torch

    from neuroproxy.rppg.neural.adapter import MODEL_DIR
    from neuroproxy.rppg.neural.efficientphys import load_pretrained

    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = "cpu"      # MPS measured slower than CPU for this model size

    model = load_pretrained(MODEL_DIR / cfg.checkpoint, img_size=72,
                            frame_depth=FRAME_DEPTH).to(device)
    train_chunks = make_chunks(train_clips, lags=lags)
    val_chunks = make_chunks(val_clips, lags=lags)
    if verbose:
        print("  chunks: {} train / {} val".format(len(train_chunks), len(val_chunks)))

    result = TrainResult(
        train_subjects=sorted({c.subject_id for c in train_clips}),
        val_subjects=sorted({c.subject_id for c in val_clips}),
    )
    if not train_chunks:
        result.notes.append("no training chunks; nothing to fine-tune on")
        return model, result

    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(train_chunks))
        running = 0.0
        for step, idx in enumerate(order):
            frames, label = train_chunks[idx]
            x = _standardize(frames)
            x = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).to(device)
            y = torch.from_numpy(label).to(device)
            loss = negative_pearson(model(x).squeeze(-1), y)
            loss = loss / cfg.batch_chunks
            loss.backward()
            if (step + 1) % cfg.batch_chunks == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            running += float(loss) * cfg.batch_chunks
        tr = running / len(order)
        va = _eval_loss(model, val_chunks, device)
        result.train_loss.append(tr)
        result.val_loss.append(va)
        result.epochs_run = epoch + 1
        if verbose:
            print("  epoch {:2d}  train {:.4f}  val {:.4f}{}".format(
                epoch + 1, tr, va, "  *" if va < best else ""))
        if va < best:
            best = va
            result.best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(out_path))
        result.notes.append("saved {}".format(out_path))
    return model, result


# --- Phase-invariant training ---------------------------------------------

def finetune_spectral(
    train_clips,
    val_clips,
    cfg: Optional[TrainConfig] = None,
    chunk: int = 300,
    out_path: Optional[Path] = None,
    verbose: bool = True,
):
    """Fine-tune against the spectral objective instead of the waveform.

    This exists because the waveform objective cannot be trained on MCD-rPPG:
    label phase is inconsistent between chunks by whole pulse periods
    (docs/limitations.md section 22). A spectral target is invariant to that,
    needs no lag correction, and optimises the quantity the product reports.

    The overfit check that justified building it: two chunks, dropout off,
    40 steps -- loss 6.59 to 2.15 (2.14 is the floor for a perfect sinusoid at
    the target rate) and HR error 29.4 to 0.20 bpm.
    """
    import torch

    from neuroproxy.rppg.neural.adapter import MODEL_DIR
    from neuroproxy.rppg.neural.efficientphys import load_pretrained

    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)

    model = load_pretrained(MODEL_DIR / cfg.checkpoint, img_size=72,
                            frame_depth=FRAME_DEPTH)
    train_chunks = make_spectral_chunks(train_clips, chunk=chunk)
    val_chunks = make_spectral_chunks(val_clips, chunk=chunk)

    result = TrainResult(
        train_subjects=sorted({c.subject_id for c in train_clips}),
        val_subjects=sorted({c.subject_id for c in val_clips}),
    )
    if not train_chunks:
        result.notes.append("no training chunks")
        return model, result
    if verbose:
        print("  chunks: {} train / {} val (spectral objective)".format(
            len(train_chunks), len(val_chunks)))

    fs = train_clips[0].fps
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def evaluate(chunks) -> float:
        if not chunks:
            return float("nan")
        model.eval()
        total = 0.0
        with torch.no_grad():
            for frames, hr in chunks:
                x = torch.from_numpy(np.transpose(_standardize(frames), (0, 3, 1, 2)))
                total += float(spectral_cross_entropy(model(x).squeeze(-1), hr, fs))
        return total / len(chunks)

    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(train_chunks))
        running = 0.0
        for step, idx in enumerate(order):
            frames, hr = train_chunks[idx]
            x = torch.from_numpy(np.transpose(_standardize(frames), (0, 3, 1, 2)))
            loss = spectral_cross_entropy(model(x).squeeze(-1), hr, fs) / cfg.batch_chunks
            loss.backward()
            if (step + 1) % cfg.batch_chunks == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            running += float(loss.detach()) * cfg.batch_chunks
        tr = running / len(order)
        va = evaluate(val_chunks)
        result.train_loss.append(tr)
        result.val_loss.append(va)
        result.epochs_run = epoch + 1
        if verbose:
            print("  epoch {:2d}  train {:.4f}  val {:.4f}{}".format(
                epoch + 1, tr, va, "  *" if va < best else ""))
        if va < best:
            best, result.best_epoch = va, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(out_path))
        result.notes.append("saved {}".format(out_path))
    return model, result
