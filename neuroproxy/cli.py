"""NeuroProxy command line: benchmark and stress-test the sensor layer.

    python -m neuroproxy.cli list
    python -m neuroproxy.cli bench --dataset synthetic --methods pos,chrom,green
    python -m neuroproxy.cli bench --dataset ubfc_rppg --root /data/UBFC
    python -m neuroproxy.cli sweep --axis jpeg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuroproxy.rppg.base import available_methods  # noqa: E402
from training.datasets.base import available_datasets, get_dataset  # noqa: E402
from training.evaluation.harness import format_table, run_benchmark  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "datasets.json"


def _load_roots(config: Optional[Path]) -> Dict[str, str]:
    path = config or DEFAULT_CONFIG
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: v for k, v in data.get("roots", {}).items() if v}


def cmd_list(args: argparse.Namespace) -> int:
    roots = _load_roots(args.config)
    print("rPPG methods: " + ", ".join(available_methods()))
    print("datasets:")
    for name in available_datasets():
        root = roots.get(name)
        ds = get_dataset(name, root=Path(root) if root else None)
        status = "available" if ds.is_available() else ds.unavailable_reason()
        print("  {:<12} {}".format(name, status))
    return 0


def _resolve_dataset(args: argparse.Namespace):
    roots = _load_roots(args.config)
    root = args.root or roots.get(args.dataset)
    kwargs = {}
    if args.dataset == "synthetic":
        kwargs["n_subjects"] = args.limit or 4
        kwargs["duration_s"] = args.duration
    if args.max_frames and args.dataset != "synthetic":
        kwargs["max_frames"] = args.max_frames
    return get_dataset(args.dataset, root=Path(root) if root else None, **kwargs)


def cmd_bench(args: argparse.Namespace) -> int:
    dataset = _resolve_dataset(args)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print("benchmark: dataset={} methods={} window={}s stride={}s".format(
        dataset.name, ",".join(methods), args.window, args.stride))
    run = run_benchmark(
        dataset,
        methods,
        window_s=args.window,
        stride_s=args.stride,
        limit=args.limit,
        verbose=not args.quiet,
    )
    print()
    print(format_table(run))
    if args.out:
        path = run.save(Path(args.out))
        print("\nwrote {}".format(path))
    # Exit non-zero when the leading method misses the go/no-go bar, so this
    # can gate CI once real datasets are wired in.
    if args.strict and run.results:
        from training.evaluation.harness import check_go_no_go

        if not check_go_no_go(run.results[0])["overall_pass"]:
            return 1
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Degrade synthetic recordings along one axis and watch accuracy fall.

    A method that does not degrade here is not being tested; a method that
    collapses immediately tells us which capture constraint the product must
    enforce in the browser.
    """
    from training.datasets.synthetic import SyntheticDataset
    from training.evaluation.harness import run_benchmark as _run

    axes: Dict[str, List] = {
        "noise": [0.5, 1.5, 3.0, 6.0, 12.0],
        "motion": [0.0, 1.0, 3.0, 6.0, 12.0],
        "jpeg": [None, 95, 85, 70, 50],
        "light": [0.0, 0.05, 0.15, 0.30, 0.60],
    }
    key = {
        "noise": "noise_sigma",
        "motion": "motion_px",
        "jpeg": "jpeg_quality",
        "light": "illum_drift",
    }[args.axis]

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print("sweep: axis={} methods={}".format(args.axis, ",".join(methods)))
    header = "{:<10} ".format(args.axis) + " ".join(
        "{:>16}".format(m) for m in methods
    )
    print()
    print(header)
    print("-" * len(header))

    rows = []
    for value in axes[args.axis]:
        ds = SyntheticDataset(
            n_subjects=args.limit or 3, duration_s=args.duration, **{key: value}
        )
        run = _run(ds, methods, window_s=args.window, stride_s=args.stride, verbose=False)
        by_method = {m.method: m for m in run.results}
        cells = []
        for m in methods:
            r = by_method.get(m)
            mae = r.median_mae_all_bpm if r is not None else None
            cov = r.coverage if r is not None else None
            cells.append("{:>16}".format(
                "n/a" if mae is None
                else "{:.2f} ({:.0f}%)".format(mae, 100.0 * (cov or 0.0))
            ))
        print("{:<10} ".format(str(value)) + " ".join(cells))
        rows.append({
            "value": value,
            "mae_all": {m: (by_method[m].median_mae_all_bpm if m in by_method else None)
                        for m in methods},
            "mae_answered": {m: (by_method[m].median_mae_bpm if m in by_method else None)
                             for m in methods},
            "coverage": {m: (by_method[m].coverage if m in by_method else None)
                         for m in methods},
        })

    print()
    print("median subject HR MAE in bpm over ALL windows with ground truth,")
    print("including ones the engine refused (abstain-independent), with the")
    print("share it actually answered in brackets. Measuring signal degradation")
    print("on answered windows only would let a method look better by refusing more.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"axis": args.axis, "rows": rows}, indent=2))
        print("wrote {}".format(args.out))
    return 0


def cmd_calib(args: argparse.Namespace) -> int:
    """Test whether confidence predicts error, on a mixed-quality cohort.

    This is the evidence behind the "abstain when the signal is bad" claim. A
    clean cohort cannot answer it -- every window is easy, so there is nothing
    for a confidence score to rank.
    """
    from neuroproxy.pipeline.offline import analyze, extract_traces
    from neuroproxy.rppg.base import get_method
    from training.datasets.base import get_dataset
    from training.datasets.synthetic import STRESS_LEVELS, SyntheticStressDataset
    from training.evaluation.calibration import format_report
    from training.evaluation.harness import calibration_for, counterfactual_error

    if args.dataset == "synthetic_stress":
        ds = SyntheticStressDataset(
            subjects_per_level=args.per_level, duration_s=args.duration
        )
    else:
        roots = _load_roots(args.config)
        root = roots.get(args.dataset)
        ds = get_dataset(args.dataset, root=Path(root) if root else None)
        if not ds.is_available():
            print("dataset unavailable: {}".format(ds.unavailable_reason()))
            return 1
    method = get_method(args.method)
    print("calibration: method={} dataset={}".format(args.method, ds.name))

    all_windows = []
    by_level = {}
    for rec in ds.recordings(limit=args.limit):
        traces = extract_traces(rec)
        windows = analyze(
            rec, method, traces=traces, window_s=args.window, stride_s=args.stride
        )
        all_windows.extend(windows)
        level = str(rec.labels.get("quality_level", rec.subject_id))
        by_level.setdefault(level, []).append((rec.subject_id, windows))

    levels = [l for l, *_ in STRESS_LEVELS if l in by_level]
    if not levels:
        levels = sorted(by_level)   # per-recording rows for real datasets

    print()
    print("per {}:".format("capture-quality level" if len(levels) < len(by_level) or
                           levels[0] in ("clean", "good") else "recording"))
    header = "{:<8} {:>8} {:>9} {:>10} {:>11} {:>9}".format(
        "level", "windows", "answered", "conf_mean", "MAE_answer", "MAE_all")
    print(header)
    print("-" * len(header))
    for level in levels:
        entries = by_level.get(level, [])
        ws = [w for _, group in entries for w in group]
        if not ws:
            continue
        answered = [w for w in ws if w.valid]
        errs_ans = [w.abs_error for w in answered if w.abs_error is not None]
        errs_all = [e for e in (counterfactual_error(w) for w in ws) if e is not None]
        print("{:<8} {:>8d} {:>8.0f}% {:>10.3f} {:>11} {:>9}".format(
            level,
            len(ws),
            100.0 * len(answered) / max(len(ws), 1),
            float(np.mean([w.confidence for w in ws])),
            "n/a" if not errs_ans else "{:.2f}".format(float(np.mean(errs_ans))),
            "n/a" if not errs_all else "{:.2f}".format(float(np.mean(errs_all))),
        ))

    print()
    report = calibration_for(all_windows)
    print(format_report(report))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report.as_dict(), indent=2))
        print("\nwrote {}".format(args.out))

    if args.strict:
        from training.evaluation.harness import GO_NO_GO

        cr = report.capture_ratio
        ok = cr is not None and cr >= GO_NO_GO["capture_ratio"]
        print("\ngo/no-go  capture_ratio {} target {}  {}".format(
            "n/a" if cr is None else "{:.3f}".format(cr),
            GO_NO_GO["capture_ratio"], "PASS" if ok else "FAIL"))
        return 0 if ok else 1
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """Design doc 8.3 ablations. Currently: no baseline vs personal baseline."""
    from neuroproxy.calibration import feature_dict
    from neuroproxy.calibration import fit as fit_baseline
    from neuroproxy.pipeline.offline import analyze, extract_traces
    from neuroproxy.rppg.base import get_method
    from training.datasets.synthetic import (
        SyntheticProtocolDataset,
        window_condition,
    )
    from training.evaluation.ablation import (
        baseline_ablation,
        format_baseline_ablation,
    )

    # Rest is longer than the calibration period on purpose: the baseline is
    # fitted on the first `calib` seconds and evaluated on rest windows that
    # start after it. Scoring the same windows the baseline was fitted on would
    # force their z-scores to ~0 and manufacture separability.
    ds = SyntheticProtocolDataset(
        n_subjects=args.limit or 6,
        rest_s=args.rest,
        task_s=args.task,
        recovery_s=args.recovery,
        task_delta_bpm=args.task_delta,
    )
    method = get_method(args.method)
    print("ablation: personal baseline, method={} subjects={}".format(
        args.method, args.limit or 6))
    print("  protocol: {:.0f}s rest / {:.0f}s task / {:.0f}s recovery, "
          "task +{:.0f} bpm".format(args.rest, args.task, args.recovery,
                                    args.task_delta))
    print("  baseline fitted on the first {:.0f}s; rest windows starting before "
          "{:.0f}s are excluded from scoring".format(args.calib, args.calib))
    print("  normalisation mode: {}".format(args.mode))
    print()

    per_subject = {}
    resting_hrs = []
    calib_failures = []
    for rec in ds.recordings():
        resting_hrs.append(float(rec.labels["hr_bpm_resting"]))
        traces = extract_traces(rec)
        windows = analyze(
            rec, method, traces=traces, window_s=args.window, stride_s=args.stride
        )
        baseline = fit_baseline(windows, calibration_seconds=args.calib)
        if not baseline.ready:
            calib_failures.append("{}: {}".format(rec.subject_id, baseline.reason))
            continue

        rows = []
        for w in windows:
            if not w.valid or w.features.hr_bpm is None:
                continue
            cond = window_condition(w, rec.labels)
            if cond is None or cond == "recovery":
                continue
            # Held-out: a rest window overlapping the calibration period was
            # used to build the baseline it would be scored against.
            if cond == "rest" and w.start_s < args.calib:
                continue
            norm = baseline.transform(
                feature_dict(w.features), mode=args.mode
            ).get("hr_bpm")
            rows.append((cond, float(w.features.hr_bpm), norm))
        if rows:
            per_subject[rec.subject_id] = rows

    spread = (max(resting_hrs) - min(resting_hrs)) if resting_hrs else None
    result = baseline_ablation(
        per_subject,
        feature="hr_bpm",
        subject_spread_bpm=spread,
        task_delta_bpm=args.task_delta,
    )
    for f in calib_failures:
        result.notes.append("calibration failed for " + f)
    print(format_baseline_ablation(result))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result.as_dict(), indent=2))
        print("\nwrote {}".format(args.out))
    return 0


def cmd_threshold(args: argparse.Namespace) -> int:
    """Derive the abstain threshold from measured error, not intuition."""
    from neuroproxy.confidence import ABSTAIN_BELOW
    from neuroproxy.pipeline.offline import analyze, extract_traces
    from neuroproxy.rppg.base import get_method
    from training.datasets.base import get_dataset
    from training.evaluation.harness import counterfactual_error
    from training.evaluation.threshold import format_report, sweep

    roots = _load_roots(args.config)
    root = args.root or roots.get(args.dataset)
    ds = get_dataset(args.dataset, root=Path(root) if root else None)
    if not ds.is_available():
        print("dataset unavailable: {}".format(ds.unavailable_reason()))
        return 1

    method = get_method(args.method)
    print("threshold sweep: dataset={} method={}".format(ds.name, args.method))
    conf, err = [], []
    for rec in ds.recordings(limit=args.limit):
        windows = analyze(
            rec, method, traces=extract_traces(rec),
            window_s=args.window, stride_s=args.stride,
        )
        for w in windows:
            e = counterfactual_error(w)
            if e is not None:
                conf.append(w.confidence)
                err.append(e)

    report = sweep(conf, err, acceptable_error_bpm=args.acceptable)
    print()
    print(format_report(report, current=ABSTAIN_BELOW))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report.as_dict(), indent=2))
        print("\nwrote {}".format(args.out))
    return 0


def cmd_ocular(args: argparse.Namespace) -> int:
    """Validate blink detection against a dataset's blink ground truth."""
    import h5py

    from neuroproxy.features.ocular import detect_blinks
    from neuroproxy.pipeline.offline import extract_traces
    from training.datasets.base import get_dataset
    from training.evaluation.ocular import evaluate, format_table

    roots = _load_roots(args.config)
    root = args.root or roots.get(args.dataset)
    ds = get_dataset(args.dataset, root=Path(root) if root else None)
    if not ds.is_available():
        print("dataset unavailable: {}".format(ds.unavailable_reason()))
        return 1

    from neuroproxy.features.ocular import BLINK_DROP_RATIO

    print("blink validation: dataset={}  drop_ratio={}".format(
        ds.name, "sweep" if args.sweep else args.drop_ratio))
    cache = []
    rows = []
    for rec in ds.recordings(limit=args.limit):
        traces = extract_traces(rec)
        if traces.openness is None:
            continue
        # Ground truth lives in the dataset file; only SCAMPS provides it today.
        path = rec.metadata.get("path")
        if not path or not str(path).endswith(".mat"):
            print("  {}: no blink ground truth available".format(rec.subject_id))
            continue
        with h5py.File(str(path), "r") as f:
            if "au45" not in f:
                continue
            au45 = np.asarray(f["au45"]).ravel()[: traces.openness.size]
        cache.append((rec.subject_id, traces.openness, au45, rec.fps))
        onsets = [s for s, _ in detect_blinks(
            traces.openness, rec.fps, drop_ratio=args.drop_ratio)]
        rows.append(evaluate(rec.subject_id, onsets, au45, rec.fps))

    if not rows:
        print("no recordings with blink ground truth")
        return 1

    if args.sweep:
        print()
        print("{:>10} {:>10} {:>8} {:>7} {:>14}".format(
            "drop_ratio", "precision", "recall", "F1", "rate_MAE/min"))
        print("-" * 52)
        for dr in (0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95):
            evs = [evaluate(sid, [s for s, _ in detect_blinks(op, f, drop_ratio=dr)],
                            au, f) for sid, op, au, f in cache]
            mark = "  <- current" if abs(dr - BLINK_DROP_RATIO) < 1e-9 else ""
            errs = [e.rate_error_per_min for e in evs if e.rate_error_per_min is not None]
            print("{:>10.2f} {:>10.2f} {:>8.2f} {:>7.2f} {:>14.1f}{}".format(
                dr,
                float(np.mean([e.precision if e.precision is not None else 0.0 for e in evs])),
                float(np.mean([e.recall or 0.0 for e in evs])),
                float(np.mean([e.f1 or 0.0 for e in evs])),
                float(np.mean(errs)) if errs else float("nan"), mark))
        print("\n  fitted on {} subjects; re-derive on any new cohort".format(len(cache)))
        return 0

    print()
    print(format_table(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps([r.as_dict() for r in rows], indent=2))
        print("\nwrote {}".format(args.out))
    return 0


def cmd_readiness(args: argparse.Namespace) -> int:
    """Score the pre-session readiness check against actual session outcomes."""
    from neuroproxy.pipeline.offline import analyze, extract_traces
    from neuroproxy.readiness import assess
    from neuroproxy.rppg.base import get_method
    from training.datasets.base import get_dataset
    from training.evaluation.metrics import subject_metrics

    roots = _load_roots(args.config)
    root = args.root or roots.get(args.dataset)
    ds = get_dataset(args.dataset, root=Path(root) if root else None,
                     **({"max_frames": args.max_frames} if args.max_frames else {}))
    if not ds.is_available():
        print("dataset unavailable: {}".format(ds.unavailable_reason()))
        return 1

    method = get_method(args.method)
    print("readiness check: dataset={}  preview={} frames".format(
        ds.name, args.preview))
    print("  NOTE: thresholds were fitted on this dataset, so this is an "
          "in-sample fit, not a held-out result.")
    rows = []
    for rec in ds.recordings(limit=args.limit):
        r = assess(rec.frames(), max_frames=args.preview)
        windows = analyze(rec, method, traces=extract_traces(rec),
                          window_s=args.window, stride_s=args.stride)
        m = subject_metrics(rec.subject_id, windows, labels=rec.labels)
        rows.append((rec.subject_id, rec.labels.get("step"), r, m))

    print()
    header = "{:<7} {:<7} {:<9} {:>7} {:>9} {:>8} {:>8}".format(
        "id", "step", "verdict", "skin", "predicted", "actual", "MAE")
    print(header)
    print("-" * len(header))
    for sid, step, r, m in rows:
        print("{:<7} {:<7} {:<9} {:>7} {:>8}% {:>8} {:>8}".format(
            sid, str(step), r.verdict,
            "n/a" if r.skin_fraction is None else "{:.2f}".format(r.skin_fraction),
            "n/a" if r.expected_usable is None else "{:.0f}".format(100 * r.expected_usable),
            "usable" if m.usable else "FAILED",
            "n/a" if m.mae_all_bpm is None else "{:.1f}".format(m.mae_all_bpm)))

    # Confusion between the pre-session verdict and what actually happened.
    tp = sum(1 for _, _, r, m in rows if r.ready and m.usable)
    fp = sum(1 for _, _, r, m in rows if r.ready and not m.usable)
    fn = sum(1 for _, _, r, m in rows if not r.ready and m.usable)
    tn = sum(1 for _, _, r, m in rows if not r.ready and not m.usable)
    total = len(rows)
    print()
    print("  predicted ready  -> usable {:>2}   failed {:>2}".format(tp, fp))
    print("  predicted poor   -> usable {:>2}   failed {:>2}".format(fn, tn))
    if total:
        print()
        print("  sessions correctly flagged before running: {}/{} = {:.0%}".format(
            tp + tn, total, (tp + tn) / total))
        if fp + tp:
            print("  of sessions it let through, {:.0%} produced a usable pulse".format(
                tp / (tp + fp)))
        if fn + tn:
            print("  of sessions it warned about, {:.0%} would have failed".format(
                tn / (fn + tn)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neuroproxy", description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="dataset roots JSON")
    sub = p.add_subparsers(dest="command", required=True)

    lst = sub.add_parser("list", help="show available methods and datasets")
    lst.set_defaults(func=cmd_list)

    def common(sp):
        sp.add_argument("--methods", default="pos,chrom,green")
        sp.add_argument("--window", type=float, default=20.0, help="window seconds")
        sp.add_argument("--stride", type=float, default=1.0, help="stride seconds")
        sp.add_argument("--limit", type=int, default=None, help="max subjects")
        sp.add_argument("--duration", type=float, default=60.0, help="synthetic seconds")
        sp.add_argument("--out", type=Path, default=None, help="write JSON results")

    b = sub.add_parser("bench", help="benchmark rPPG methods against ground truth")
    b.add_argument("--dataset", default="synthetic", choices=available_datasets())
    b.add_argument("--root", type=Path, default=None, help="dataset root override")
    b.add_argument("--max-frames", type=int, default=None)
    b.add_argument("--quiet", action="store_true")
    b.add_argument("--strict", action="store_true", help="exit 1 if go/no-go fails")
    common(b)
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("sweep", help="degrade synthetic data and plot the fall-off")
    s.add_argument("--axis", default="noise", choices=["noise", "motion", "jpeg", "light"])
    common(s)
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser(
        "calib", help="test whether confidence predicts error (risk-coverage)"
    )
    c.add_argument("--method", default="pos")
    c.add_argument("--dataset", default="synthetic_stress",
                   help="synthetic_stress, or any configured dataset")
    c.add_argument("--limit", type=int, default=None, help="max recordings")
    c.add_argument("--per-level", type=int, default=2,
                   help="subjects per quality level (synthetic_stress only)")
    c.add_argument("--window", type=float, default=20.0)
    c.add_argument("--stride", type=float, default=2.0)
    c.add_argument("--duration", type=float, default=60.0)
    c.add_argument("--out", type=Path, default=None)
    c.add_argument("--strict", action="store_true", help="exit 1 if calibration fails")
    c.set_defaults(func=cmd_calib)

    a = sub.add_parser("ablate", help="design doc 8.3 ablations")
    a.add_argument("--method", default="pos")
    a.add_argument("--limit", type=int, default=None, help="number of subjects")
    a.add_argument("--rest", type=float, default=90.0, help="rest seconds")
    a.add_argument("--task", type=float, default=60.0, help="task seconds")
    a.add_argument("--recovery", type=float, default=30.0)
    a.add_argument("--task-delta", type=float, default=12.0, help="task HR rise, bpm")
    a.add_argument("--calib", type=float, default=45.0, help="calibration seconds")
    a.add_argument("--mode", default="delta", choices=["delta", "z"],
                   help="baseline normalisation: subtract location, or also divide by scale")
    a.add_argument("--window", type=float, default=20.0)
    a.add_argument("--stride", type=float, default=2.0)
    a.add_argument("--out", type=Path, default=None)
    a.set_defaults(func=cmd_ablate)

    t = sub.add_parser(
        "threshold", help="derive the abstain threshold from measured error"
    )
    t.add_argument("--dataset", default="scamps")
    t.add_argument("--root", type=Path, default=None)
    t.add_argument("--method", default="pos")
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--window", type=float, default=10.0)
    t.add_argument("--stride", type=float, default=1.0)
    t.add_argument("--acceptable", type=float, default=5.0,
                   help="error bound in bpm above which a window is unacceptable")
    t.add_argument("--out", type=Path, default=None)
    t.set_defaults(func=cmd_threshold)

    o = sub.add_parser("ocular", help="validate blink detection against ground truth")
    o.add_argument("--dataset", default="scamps")
    o.add_argument("--root", type=Path, default=None)
    o.add_argument("--limit", type=int, default=None)
    o.add_argument("--sweep", action="store_true",
                   help="score a range of thresholds instead of just the default")
    # Default comes from the module so the two cannot drift apart -- they
    # already had, leaving the CLI scoring 0.85 while the library used 0.90.
    from neuroproxy.features.ocular import BLINK_DROP_RATIO as _BDR

    o.add_argument("--drop-ratio", type=float, default=_BDR,
                   help="blink threshold as a fraction of the open-eye baseline "
                        "(default {})".format(_BDR))
    o.add_argument("--out", type=Path, default=None)
    o.set_defaults(func=cmd_ocular)

    rd = sub.add_parser(
        "readiness", help="score the pre-session readiness check against outcomes"
    )
    rd.add_argument("--dataset", default="mcd_rppg")
    rd.add_argument("--root", type=Path, default=None)
    rd.add_argument("--method", default="pos")
    rd.add_argument("--limit", type=int, default=None)
    rd.add_argument("--preview", type=int, default=60, help="preview frames to sample")
    rd.add_argument("--window", type=float, default=20.0)
    rd.add_argument("--stride", type=float, default=5.0)
    rd.add_argument("--max-frames", type=int, default=1800)
    rd.set_defaults(func=cmd_readiness)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
