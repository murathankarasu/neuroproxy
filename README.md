# NeuroProxy

Camera-based physiological state engine. This repository currently implements
**the sensor layer and its benchmark harness** -- the offline half of the
pipeline described in `NeuroProxy_Technical_Pipeline_TR.pdf` (design doc v1.0).

The order is deliberate. Before any state model is trained, we prove that a
pulse can be recovered from video at all, and we measure how fast that breaks
as the camera degrades. A state model built on an unvalidated sensor layer
produces confident numbers with nothing underneath them.

## Status

| Layer | State |
|---|---|
| Signal primitives (band-pass, PSD, HR, SNR, IBI) | done, tested |
| rPPG methods: POS, CHROM, green baseline | done, tested |
| Face detection + skin ROI | done (mediapipe / Haar / skin / static backends) |
| Quality gate (face, light, blur, motion, FPS, compression) | done, tested |
| Confidence + abstention, with risk-coverage evidence | done, tested |
| Abstain-independent reporting + go/no-go | done, tested |
| Personal baseline + calibration gate | done, tested |
| Ocular: blink detection, validated against ground truth | done, tested |
| Respiration proxy | **not built** -- could not be validated, see below |
| Ablation: no baseline vs personal baseline | done |
| Offline windowed pipeline | done, tested |
| Benchmark harness + go/no-go check | done |
| MCD-rPPG (600 real subjects, CC-BY-4.0) | **downloaded and benchmarked** |
| Subgroup fairness reporting (age, sex, condition) | done |
| Pre-session readiness check | done, tested |
| SCAMPS (rendered faces) | downloaded and benchmarked |
| Dataset adapters: UBFC-rPPG, UBFC-Phys, PURE | written, **not yet run on real data** |
| Data-derived abstain threshold | done |
| Streaming engine (1 Hz state, exact offline parity) | done, tested |
| Session API: sessions, events, summary, WS stream | done, tested |
| Live camera source with capture-settings reporting | written, not run here |
| Single-session web demo (timeline, events, summary) | done, served at `/` |
| Neural rPPG (EfficientPhys, pretrained) | benchmarked, **does not transfer** |
| Fine-tuning, waveform objective | **blocked by label sync**, limitations 22 |
| Fine-tuning, phase-invariant objective | works; error halved, still behind POS |
| State model (XGBoost -> TCN) | not started |
| Realtime + API | not started |

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m neuroproxy.cli list
```

Verify the harness against synthetic data with known ground truth:

```bash
.venv/bin/python -m neuroproxy.cli bench --dataset synthetic --duration 60
```

Measure how each method degrades as the camera gets worse:

```bash
.venv/bin/python -m neuroproxy.cli sweep --axis jpeg
```

Test whether confidence actually predicts error, rather than assuming it does:

```bash
.venv/bin/python -m neuroproxy.cli calib
```

Run the design doc's personal-baseline ablation:

```bash
.venv/bin/python -m neuroproxy.cli ablate
```

Run the tests:

```bash
.venv/bin/python -m pytest tests/ -q
```

## Results on real people (MCD-rPPG)

25 real subjects, 50 recordings at rest and post-exercise, consumer webcam,
compressed MPEG-4:

| method | MAE_all | MAE_answered | worst | r | SNR dB | coverage |
|---|---|---|---|---|---|---|
| pos | **2.87** | 0.55 | 31.82 | 0.696 | 0.8 | 43% |
| chrom | 4.13 | 0.71 | 31.98 | 0.669 | 0.7 | 42% |
| green | 16.65 | 2.09 | 36.25 | 0.533 | -0.7 | 22% |

Accuracy passes the design doc bar (2.87 vs 5.0). **Coverage fails** (43% vs
80%), decomposing into 58% usable sessions x 69% coverage within them.

The reality gap against everything measured earlier:

| | ellipse | SCAMPS (rendered) | MCD-rPPG (real) |
|---|---|---|---|
| POS MAE_all | 0.02 | 0.53 | **2.87** |
| coverage | 100% | 75% | **43%** |

Subgroup reporting shows a large median difference by sex (F 1.22, M 6.09)
that **does not survive testing** (p = 0.083, n = 25) and whose obvious
explanation -- face size -- was tested and rejected. Limitations section 15 has
the full analysis. MCD-rPPG has 600 subjects, so this is answerable by
downloading more, not by speculating.

```bash
.venv/bin/python -m neuroproxy.cli bench --dataset mcd_rppg --window 20 --stride 5 --max-frames 1800
```

MCD-rPPG is **CC-BY-4.0 -- commercial use permitted**, direct download, no form.

### Why sessions fail, and catching it early

Of every quality dimension measured, exactly one separates usable sessions from
unusable ones: **skin fraction** (0.845 vs 0.643, p = 0.004). Below ~0.70 only
18% of sessions yield a usable pulse; above it, 70% do at 0.8 bpm median error.

This also resolves the apparent sex gap -- skin fraction differs sharply by sex
(F 0.845, M 0.660), and controlling for it removes the difference entirely
(p = 0.82 in the high-skin stratum, where men are slightly better). The cause is
how much clear skin the ROI sees, not sex.

A pre-session check measures it from a 2-second preview:

```bash
.venv/bin/python -m neuroproxy.cli readiness --dataset mcd_rppg
```

70% of sessions correctly flagged before running; sessions it passes are usable
68% of the time against a 52% base rate. Thresholds are fitted in-sample, so
expect less out of sample. Limitations section 16.

The mechanism behind low skin fraction is identified: the forehead ROI sits on
the hairline, because ROI boxes are fixed fractions of a detector box that
frames the head rather than the face.

The first fix -- re-seating every ROI on the detected skin region and hairline
-- looked visibly correct and measured worse on every axis (MAE_all 4.00 ->
4.28), because the cheeks slid onto the jaw and beard. It was reverted.

Moving **only the forehead**, as a band below the detected hairline, worked:

| | baseline | all-ROI shift | forehead only |
|---|---|---|---|
| MAE_all | 4.00 | 4.28 | **2.87** |
| usable sessions | 52% | 52% | **58%** |
| worst subject | 35.15 | 47.76 | **31.82** |

A 28% error reduction from moving one ROI instead of three. Limitations
section 17.

Pretrained **EfficientPhys** was benchmarked against POS. In-domain it is
clearly better (SCAMPS weights on SCAMPS: 0.21 bpm vs POS 1.05, and better
coverage). Across domains it collapses -- on MCD-rPPG it scores 31.73 (PURE
weights) and 43.96 (UBFC weights) against POS's 2.87, 11-15x worse than a
method with no learned parameters.

That invalidates the design doc's Phase A plan of freezing a pretrained encoder
and training only a state head on top. Limitations section 19.

Fine-tuning it took three attempts to get right. The waveform objective the
architecture ships with **cannot be trained on this dataset** -- the contact PPG
is aligned in rate but not in phase, inconsistently by whole pulse periods, so
training plateaus at r = 0.035 (limitations 21-22). Replacing it with a
phase-invariant spectral objective works:

| arm | MAE | median | within 5 bpm |
|---|---|---|---|
| POS | **11.30** | **1.52** | **67%** |
| EfficientPhys pretrained | 32.17 | 24.71 | 13% |
| EfficientPhys, waveform fine-tune | 27.88 | 21.14 | 12% |
| EfficientPhys, spectral fine-tune | 17.88 | 12.12 | 28% |

Error nearly halved against the pretrained model, and the objective is now
demonstrably trainable. **POS remains the reference method** by a wide margin.
Limitations section 23.

Whether the three ROIs should be pooled at all was tested next. There is about
**4.5x of headroom** in choosing between them (oracle 0.67 bpm vs 2.99 pooled),
and neither SNR-based selection nor cross-ROI consensus reaches any of it --
both are coin flips. Hand-built quality proxies test whether a signal is peaky,
not whether it is right. That is the concrete argument for the neural stage,
with a target to beat. Limitations section 18.

## Results on rendered faces (SCAMPS)

The first data here containing an actual face. POS, 10 subjects, 10 s windows:

| method | MAE_all | MAE_answered | MAE worst | r | SNR dB | coverage |
|---|---|---|---|---|---|---|
| pos | **0.53** | 0.37 | 0.66 | 0.753 | 2.6 | 75% |
| chrom | 0.59 | 0.45 | 51.56 | 0.789 | 2.9 | 70% |
| green | 72.10 | 80.20 | 98.63 | 0.342 | 1.5 | 57% |

Accuracy passes the design doc bar with wide margin. **Pooled coverage does
not** (75% vs 80%), and it decomposes into two different problems:

| | value |
|---|---|
| usable-session rate | 8/10 = 80% |
| coverage within usable sessions | 93% |

The two unusable sessions are overexposed (12% and 34% of pixels clipped) with
counterfactual errors of 31 and 67 bpm. Refusing them is correct; that is a
capture and onboarding problem, not a model one. The literal 80% pooled
criterion is left in place and still fails.

Reaching 80% pooled coverage on this cohort would require admitting a window
with 49 bpm of error, so coverage and accuracy are in genuine conflict here --
not merely mistuned.

Green collapsing here is the point of keeping it: on the flat-ellipse generator
it scored 0.06 and looked competitive. Real facial geometry is what separates
the chrominance methods from a luma baseline.

Running SCAMPS surfaced four defects invisible on synthetic data, three of them
running in the same direction -- a fixed skin-colour locus that excluded a
dark-skinned subject entirely, exposure judged on the background rather than the
face, a pulse-stability gate that was a silent no-op at 10 s windows, and an
abstain threshold discarding a quarter of all windows for zero accuracy gain.
A fifth surfaced on the follow-up: the pulse-stability metric used a
non-robust standard deviation, so a single sub-window locking onto the second
harmonic (`[55.1, 55.1, 103.2]`) refused a subject on 91% of its windows while
its estimates were accurate to 0.5 bpm. Limitations section 12 has the details
and the before/after numbers (MAE_all 4.36 -> 0.53, worst subject 52.69 -> 0.66,
coverage 37% -> 75%).

```bash
.venv/bin/python -m neuroproxy.cli bench --dataset scamps --window 10 --stride 1
```

### Blinks: validated. Respiration: not observable.

Both were tested against the same standard -- correlation with ground truth
compared against a permutation null over mismatched subject pairs.

Blinks pass clearly (vertical edge energy in the eye band: matched 0.691 vs
mismatched 0.113, p < 0.0001), and the detector scores **precision 0.80, recall
0.60, F1 0.66** at event level.

```bash
.venv/bin/python -m neuroproxy.cli ocular --dataset scamps --sweep
```

Respiration fails completely: matched correlation 0.233 against a mismatched
0.234, p = 0.553. Head pose is exactly zero in every clip, closing the design
doc's motion route too. **No respiration estimator was built**, because a rate
estimator on a narrowband signal always returns a rate -- the first naive
version returned 7.5 breaths/min for all ten subjects, which reads as a
systematic bias rather than the absence of signal it actually was. Limitations
sections 13 and 14.

SCAMPS is **research-licensed, no commercial use** -- see
[docs/datasets.md](docs/datasets.md) for the full access and licence matrix.

## Results on the built-in generator

Synthetic, 4 subjects x 60 s, 20 s windows / 1 s stride. Median subject HR MAE:

| method | MAE_all | MAE_answered | MAE worst subject | r | SNR dB | coverage |
|---|---|---|---|---|---|---|
| pos   | 0.03 | 0.03 | 0.03 | 1.000 | 16.7 | 100% |
| chrom | 0.05 | 0.05 | 0.16 | 1.000 | 15.7 | 100% |
| green | 0.06 | 0.06 | 0.18 | 1.000 | 16.5 | 100% |

`MAE_all` covers every window with ground truth including ones the engine
refused; `MAE_answered` covers only the ones it replied to. They are equal here
because nothing was refused. Where they diverge, the gap is coverage being
converted into apparent accuracy -- limitations section 9.

**This is a correctness check, not a performance result.** It says the harness
is wired correctly. It says nothing about real faces. See
[docs/limitations.md](docs/limitations.md) section 1.

### Findings that do transfer

**Compression is the dominant error source.** POS degrades ~25x from
uncompressed video to JPEG q=50 while the green baseline barely moves, because
chroma subsampling attacks exactly the chrominance signal POS reads. The
quality gate now detects this directly. This constrains how the browser SDK
must capture frames -- limitations section 3.

**Confidence had to be restructured before it meant anything.** Under the
design doc's single weighted sum, a well-lit recording of a face with *no pulse
at all* scored 0.637 and answered every window with a mean error of 25.2 bpm.
Image quality outvoted the absence of a signal. Image and pulse quality now
gate multiplicatively, and pulse presence is measured by sub-window HR
agreement rather than SNR -- SNR cannot tell a real pulse under noise from
filtered noise, and sub-window agreement can. Result on a mixed-quality cohort:

| metric | before | after |
|---|---|---|
| Spearman(confidence, abs error) | -0.476 | -0.610 |
| capture ratio (0 = random, 1 = oracle) | 0.482 | 0.673 |
| windows answered at worst capture level | 100% | 14% |

Details, and what this still does not prove, in limitations section 10.

**Personal baseline earns its place, but subtract only -- do not rescale.** On a
rest/task/recovery protocol where between-subject resting HR spread (42 bpm)
exceeds the task response, pooled task-vs-rest AUROC goes from 0.609 raw to
0.754 with a personal baseline, against a within-subject ceiling of 0.873 --
about 55% of the gap closed. Dividing by a scale estimated from a 45 s
calibration window made it *worse* at every effect size, because that window
cannot estimate HR variability to better than ~3.5x. Limitations section 11.

## Running against real datasets

Nothing is downloaded automatically; each dataset has its own licence and
access request. Point `configs/datasets.json` at local copies:

```json
{ "roots": { "ubfc_rppg": "/data/UBFC-rPPG", "pure": "/data/PURE" } }
```

```bash
.venv/bin/python -m neuroproxy.cli bench --dataset ubfc_rppg --strict
```

`--strict` exits non-zero when the design doc go/no-go bar (median subject HR
MAE <= 5 bpm, coverage >= 80%) is missed, so this can gate CI.

## Running the API

```bash
.venv/bin/uvicorn api.main:app --reload
```

Then open http://localhost:8000 for a full session: camera consent, 45 s
calibration, live timeline with event markers, and an end-of-session summary.

`POST /v1/sessions` to open one, push frames over
`WS /v1/sessions/{id}/stream`, read `GET /v1/sessions/{id}/summary`. State is
emitted at 1 Hz as **deviation from the subject's own baseline in bpm**, never
as an absolute score, and is `null` with a reason whenever confidence does not
support an answer.

The engine reproduces the offline pipeline exactly (max difference 0.0000 bpm
over 41 windows on a real recording), so every benchmark below describes what a
live session actually does.

**Do not push lossy frames.** Measured on the same recording: JPEG q95 costs
0.38 bpm and 3 points of coverage; JPEG q75 costs 0.85 bpm and the engine then
answers **0%** of windows -- confidence falls just enough to abstain on
everything. Limitations section 24.

## Layout

```
neuroproxy/
  calibration/ per-subject baseline and drift detection
  capture/    video, image-sequence and live-camera sources
  inference/  streaming engine: frames in, state out at 1 Hz
  vision/     face detection, skin mask, ROI averaging
  rppg/       POS, CHROM, green + shared signal primitives
  features/   cardiac features from an estimated BVP
  quality/    per-frame and per-window quality gate
  pipeline/   offline recording -> windowed results
  confidence.py image-quality gate x pulse-presence gate
  cli.py      bench / sweep / calib / ablate / list
api/          FastAPI session + WebSocket state stream
training/
  datasets/   synthetic, UBFC-rPPG, UBFC-Phys, PURE (one Recording contract)
  evaluation/ metrics, harness, risk-coverage calibration, ablations
docs/         limitations.md -- read before quoting any number
```

## Ground rules

These come from the design doc's scope-exclusion section and are enforced in
code where possible.

- The engine does not measure neurotransmitters and does not reconstruct EEG.
- No medical diagnosis, hiring decisions or employee monitoring.
- Camera-derived beat intervals are a **PRV proxy**, never HRV.
- When quality is insufficient the engine returns no state and a reason, rather
  than a low-confidence number that will be read as a measurement. This is
  tested (`tests/test_confidence.py`), not asserted.
- Metrics are computed per subject, then summarised. The worst subject is
  reported next to the median.
