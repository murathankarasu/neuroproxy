# Limitations and standing caveats

This file is the project's memory for things that are easy to forget and
expensive to get wrong. Every claim the engine makes should be checkable
against it. Items are removed only when evidence replaces them, never because
they became inconvenient.

## 1. Synthetic results are a correctness floor, not evidence

`training/datasets/synthetic.py` paints a known pulse into the pixels using the
same dichromatic reflection model POS and CHROM are derived from. Recovering
that pulse proves the harness is wired correctly -- resampling, window
alignment, PSD scaling, ground-truth pairing. It proves nothing about accuracy
on a real face.

**Never quote a synthetic MAE as a performance number.** The go/no-go bar in
the design doc (median subject HR MAE <= 5 bpm) is only meaningful on
UBFC-rPPG or PURE.

## 2. The motion axis of the synthetic sweep is a weak test

Measured behaviour: increasing `motion_px` barely moves HR error for any
method. That is a property of the generator, not a robustness result.

The generator models motion-linked shading, which is a common-mode intensity
change -- exactly what POS and CHROM are built to cancel. The mechanisms that
actually break rPPG under motion are **not modelled**:

- tracker error, so the ROI drifts across non-uniform skin and background
- non-rigid facial deformation (speech, expression)
- partial occlusion and the face leaving frame
- rolling-shutter interaction with motion

**We currently have no motion-robustness evidence.** PURE is the dataset that
answers this, because its six sessions per subject are steady / talking / slow
translation / fast translation / small rotation / medium rotation. Until PURE
is wired in, treat motion robustness as unknown.

## 3. Compression is the largest measured threat

Measured on synthetic, 40 s, 3 subjects. Median subject HR MAE (bpm) over every
window with ground truth, with the share the engine actually answered:

| JPEG quality | POS | CHROM | green |
|---|---|---|---|
| none | 0.05 (100%) | 0.25 (100%) | 0.18 (100%) |
| 95 | 0.66 (79%) | 0.55 (89%) | 0.20 (97%) |
| 85 | 0.41 (29%) | 0.73 (46%) | 0.10 (83%) |
| 70 | 0.52 (17%) | 0.75 (30%) | 0.24 (76%) |
| 50 | 1.27 (37%) | 1.01 (30%) | 0.24 (70%) |

POS degrades ~25x from uncompressed to q=50; green barely moves. The mechanism
is chroma subsampling: JPEG discards chrominance detail, and POS/CHROM read the
pulse out of chrominance while green rides on luma.

The coverage column carries the sharper statement: under compression POS is not
merely less accurate, it becomes largely **unusable**, refusing 60-80% of
windows. The engine is right to refuse -- the signal genuinely is not there --
but a product cannot be built on 17% coverage.

Product consequences:

- The browser SDK must not send or process a re-encoded stream where it can be
  avoided; prefer raw frame access and derive features on-device.
- Capture settings (codec, quality, chroma subsampling) belong in session
  metadata and must feed the quality score.
- A conference-call-grade video stream is not a valid input.

The quality gate now detects this directly (`neuroproxy.quality.compression_score`)
using two complementary indicators, measured on the synthetic generator:

| JPEG quality | chroma/luma HF ratio | blockiness | compression score |
|---|---|---|---|
| none | 0.071 | 0.000 | 1.00 |
| 95 | 0.0028 | 0.006 | 0.25 |
| 85 | 0.0014 | 0.002 | 0.08 |
| 70 | 0.0025 | 0.111 | 0.11 |
| 50 | 0.0029 | 0.304 | 0.08 |

The chroma ratio detects *whether* a codec touched the frame -- it collapses
25x at the first hint of compression, which is exactly where POS error jumps
from 0.05 to 0.66 bpm. Blockiness grades severity.

**Caveat for real data:** every webcam and every browser stream is already
compressed, so this term will be low across the board on real recordings. That
reflects genuine risk rather than a bug, but it means absolute confidence
values are not comparable between a compressed and an uncompressed source.
Ranking within one source still holds.

## 4. PRV is a proxy, not HRV

At 30 fps the inter-beat-interval grid is quantised to 33 ms. Resting RMSSD is
roughly 20-50 ms, i.e. the same order as the quantisation step, before any rPPG
peak jitter is added.

Consequence: `rmssd_proxy_ms` is retained for ablation purposes only. It must
not appear in any customer-facing output, and it must never be called HRV.
Revisit only at 60 fps or higher with peak interpolation, and only after
measuring against contact BVP. `tests/test_signal.py::test_ibi_quantisation_is_visible`
pins this caveat.

## 5. The UBFC-Phys task confound (highest-priority open item)

UBFC-Phys tasks are T1 rest, T2 speech, T3 arithmetic. T2 involves speaking, so
it differs from T1 in mouth motion, head motion and blink behaviour -- not only
in physiology. A classifier can separate the tasks from motion alone and appear
to detect stress.

**Subject-independent splits do not fix this.** LOSO prevents identity
memorisation; it does nothing about a confound present in every subject.

Required controls before any state result is believed:

1. **Motion-only ablation.** Train using only head-motion and blink features.
   Comparable accuracy means the physiology model is a motion detector.
2. **SNR-only ablation.** Train using only `pulse_snr_db` and quality features.
   rPPG quality itself degrades during speech, so signal quality can leak the
   label.
3. **Within-condition test.** Separate high- from low-anxiety subjects *inside*
   a single task, where the motion confound is constant.

The loader exposes `labels["task"]` and `labels["task_id"]` specifically so
these ablations can be written against it.

## 6. Label resolution does not support the output contract

UBFC-Phys provides task condition and self-reported anxiety per session.
EmpathicSchool annotates at roughly two-minute resolution. Neither supports a
per-20-second ground truth for `cognitive_load`.

Consequence for the API: a 0-1 absolute scale implies a measurement we cannot
make. Report state relative to the subject's own baseline, with an interval:

```json
"state": {
  "arousal": { "index": 1.4, "unit": "z_vs_baseline", "ci": [0.6, 2.2] }
}
```

and return `state: null` with a `reason` when quality is insufficient, rather
than a low-confidence number that will be read as a measurement anyway.

## 7. Fairness is not yet measured

The YCrCb skin locus in `neuroproxy/vision/detector.py` is deliberately wide,
and the synthetic cohort spans four skin tones, but neither is evidence.
rPPG amplitude genuinely falls with increasing melanin concentration, so
accuracy differences across skin tone are expected, not hypothetical.

Any benchmark on real data must report metrics **per subgroup**, not pooled.
`BenchmarkMetrics.worst_subject_mae_bpm` exists so the worst case stays visible
next to the median.

## 8. Detector backend affects results

With `mediapipe` absent the detector falls back to Haar, and on synthetic
recordings to skin-colour segmentation. These do not localise equivalently.
Benchmark runs across different backends are not comparable; the backend is
recorded in `FaceBox.backend` and should be reported alongside any number.

## 9. Two MAE definitions, and why the difference matters

Once abstention exists, "MAE" is ambiguous, and the two readings can disagree
sharply:

- **MAE_all** -- every window with a ground-truth pair, including refused ones,
  scored with the prediction they would have made. Independent of the abstain
  policy. This measures the signal.
- **MAE_answered** -- only windows the engine replied to. This is what a user
  experiences, but it is selection-biased: it improves whenever coverage falls,
  because abstention removes the hard windows first.

Measured example, POS at JPEG q=85: MAE_answered 0.21 bpm looks *better* than
q=95's 0.64, while MAE_all is 0.68 vs 0.74 -- essentially unchanged. The
apparent improvement was entirely coverage collapsing from 79% to 29%.

Consequences, enforced in code:

- The sweep and benchmark tables report MAE_all with coverage in brackets.
- Methods are ranked on MAE_all, so refusing hard windows cannot win.
- The go/no-go MAE bar is checked against MAE_all. Checking the answered-only
  figure would let a method pass by refusing; the coverage bar limits that but
  does not remove the incentive, and a threshold that can be gamed is not a
  threshold. `tests/test_reporting.py` pins this.

This was a real regression: the abstain gate was added before the reporting was
updated, and for one commit the sweep table silently credited methods for
refusing.

## 10. Confidence: what it does and does not establish

The explainer document lists "confidence-first design: abstain when the signal
is bad" as a differentiator. That is a claim about ordering, so it was tested
by risk-coverage analysis rather than asserted
(`training/evaluation/calibration.py`, `neuroproxy.cli calib`).

### The defect this testing found

Under the design doc's section 9.1 formula -- one weighted sum over face,
lighting, motion, pulse SNR and a model term -- a well-lit, still, uncompressed
recording of a face containing **no pulse at all** scored 0.637 and answered
100% of its windows, with a mean error of 25.2 bpm. Four of the five terms
describe the image, so image quality outvoted the complete absence of a signal.
The only condition that ever triggered abstention was total darkness, and that
was caught by the face detector rather than by confidence.

Two changes followed, both structural rather than reweighting:

1. **Image and pulse quality now gate multiplicatively**, not additively.
   Good lighting cannot compensate for an absent pulse, so image quality is
   necessary but not sufficient: `confidence = image_quality * pulse_quality`.

2. **Pulse presence is measured by sub-window HR agreement, not by SNR.**
   SNR cannot make the distinction. Measured on 20 s windows at 30 fps:

   | signal | SNR (dB) | HR spread (bpm) |
   |---|---|---|
   | clean pulse | 35.7 | 0.00 |
   | pulse + 3x noise | -0.3 | 13.07 |
   | pure noise (3 seeds) | -1.1 / -0.8 / -0.2 | 10.55 / 32.15 / 39.53 |

   A genuine pulse under 3x noise has essentially the same SNR as pure noise.
   The sub-window spread separates them, because a band-pass filter always
   produces a spectral peak but only a real pulse holds its rate.

### Measured result after the change

Mixed-quality synthetic cohort, 10 subjects across five capture-quality levels,
POS, 20 s windows:

| metric | before | after |
|---|---|---|
| Spearman(confidence, abs error) | -0.476 | **-0.610** |
| capture ratio (0 random, 1 oracle) | 0.482 | **0.673** |
| windows answered at "bad" level | 100% | **14%** |
| MAE kept vs abstained | abstention never fired | 0.16 vs 0.40 bpm |

Coverage now degrades with capture quality as intended: 100% / 98% / 64% / 38%
/ 14% from clean to bad.

### What this still does not establish

- **The abstain threshold is over-conservative on this cohort.** Abstained
  windows would have had a mean error of 0.40 bpm -- an order of magnitude
  below the 5 bpm go/no-go bar. On synthetic data the engine is discarding
  usable windows. The threshold (`ABSTAIN_BELOW`) must be set from a
  risk-coverage curve on real data, where errors are actually large, not tuned
  against synthetic.
- **Capture ratio 0.673 is close to this cohort's ceiling, not to a general
  result.** The synthetic errors do not order monotonically by degradation
  level (good 0.33, fair 0.31, poor 0.18, bad 0.43 bpm), so there is limited
  ordering for any score to recover. The one real cliff is compressed vs
  uncompressed, and confidence detects it.
- **Nothing here has seen a real face.** All of section 1 applies.

## 11. Personal baseline: what the ablation showed

Design doc section 8.3 lists "no baseline vs personal baseline" as a required
ablation. Run it with `neuroproxy.cli ablate`.

Protocol: 8 synthetic subjects, 90 s rest / 60 s task / 30 s recovery, baseline
fitted on the first 45 s, rest windows starting before 45 s excluded from
scoring so the baseline is never evaluated on the windows it was fitted on.
Between-subject resting HR spread (42 bpm) deliberately exceeds the task
response, which is the situation that makes a personal reference necessary.

Task-vs-rest AUROC, pooled across subjects:

| task response | raw | z-score | delta (location only) | within-subject ceiling |
|---|---|---|---|---|
| 12 bpm | 0.700 | 0.873 | **0.907** | 1.000 |
| 6 bpm | 0.609 | 0.700 | **0.754** | 0.873 |
| 3 bpm | 0.562 | 0.606 | **0.647** | 0.752 |
| 2 bpm | 0.546 | 0.576 | **0.607** | 0.701 |

Three conclusions:

1. **Personal normalisation earns its place**, closing about 55% of the gap
   between pooled-raw and the within-subject ceiling at a 6 bpm response.

2. **It does not close all of it.** Reporting the raw AUROC delta alone would
   overstate this; `gap_recovered` is the honest figure and is what the
   ablation prints.

3. **Dividing by the baseline scale makes things worse at every effect size.**
   Across subjects whose true HR variability was identical by construction, the
   MAD-derived scale ranged from 1.25 to 4.38 bpm -- a 45 s window cannot
   estimate HR variability to better than about 3.5x, so dividing by it injects
   between-subject noise. `transform` therefore defaults to `mode="delta"`.

   **Caveat:** the synthetic cohort gives every subject the same true
   variability, so scale normalisation could only ever hurt there. Real people
   differ in HR variability. Re-run this ablation on real data before treating
   the default as settled.

### A confound this ablation initially had

The first run reported that personal baselining *inverted* separability
(AUROC 0.000) at task responses below 6 bpm. The cause was in the generator,
not the method: HR drift used a period equal to the recording length, so its
positive half always covered rest and its negative half always covered the
task. Drift now uses a fixed period with a per-subject random phase, making it
a nuisance variable rather than a confound.

Worth recording because the failure was silent and plausible-looking: a
confound aligned with the protocol produces a confident, monotonic, completely
wrong result.

## 12. First results on faces: what SCAMPS changed

SCAMPS is the first data in this project containing an actual face. Four
defects surfaced within an hour of running it, none of which the flat-ellipse
generator could have shown. Recorded here because the pattern matters more than
the individual bugs: **every one of them was invisible on synthetic data and
three of the four ran in the same direction.**

POS, 10 rendered subjects, 10 s windows / 1 s stride, before and after:

| | initial | after fixes |
|---|---|---|
| MAE_all | 4.36 bpm | **0.53** |
| MAE worst subject | 52.69 bpm | **0.82** |
| coverage | 37% | **66%** |

Accuracy passes the design doc bar with a wide margin. **Coverage does not**
(66% against an 80% target), and that remains the open engineering problem.

### Defect 1 -- fixed skin locus excluded a dark-skinned subject entirely

The YCrCb locus in the detector carried a comment claiming it was "wide on
purpose" so as not to "bake a fairness failure into the sensor layer". That
claim was untested, and it was false. One subject -- dark-skinned, dim lighting
-- had **0%** of face pixels inside the locus, against 86-98% for the other
nine. The face was plainly visible and the detector found it; the mask threw it
away. The engine produced no output at all for that person.

Cause: low luminance compresses chroma toward neutral (128, 128). That subject's
median Cr was 131 against a floor of 133 and median Cb 131 against a ceiling of
127 -- outside on both axes by a hair. Darker skin starts closer to the
boundary, so the failure is systematic rather than incidental.

Fix: `vision.detector.adaptive_skin_mask`. The face box already says where skin
is, so the mask is built from that face's own chroma median and spread. Tone-
and luminance-agnostic by construction. The fixed locus survives only for
*finding* a face when no face model is available.

### Defect 2 -- exposure was judged on the frame, not the face

Three subjects were penalised for their backdrop. A dark-skinned subject
against a blown-out white background scored 0.66 on lighting because 34% of the
*background* was clipped; the face itself was 6% clipped and scores 0.94.

| subject | frame-level score | face-level score |
|---|---|---|
| P000005 | 0.88 | 1.00 |
| P000006 | 0.66 | 0.94 |
| P000009 | 0.53 | 0.73 |

All three were dark-skin, dim-light or bright-background cases. Exposure and
focus are now measured on the face box.

### Defect 3 -- the pulse-stability gate was a silent no-op

`hr_stability` defaulted to 10 s sub-windows. SCAMPS clips are 20 s, so they
are analysed at 10 s windows -- making all three sub-windows identical, the
spread exactly 0.0, and the gate pass everything unconditionally. It went
unnoticed because the synthetic cohort uses 20 s windows. The sub-window length
now scales with the analysis window, and a degenerate request returns None
rather than a flattering zero.

### Defect 4 -- the abstain threshold was costing coverage for nothing

0.45 was hand-picked. Swept against measured error on SCAMPS:

| threshold | coverage | MAE | max error |
|---|---|---|---|
| 0.00 | 100% | 10.15 | 99.07 |
| 0.15 | 70% | 1.64 | 49.37 |
| 0.20 | 66% | 0.35 | 1.35 |
| 0.45 | 42% | 0.35 | 1.35 |

Every window with error above 5 bpm had confidence at or below 0.164. Above
0.20 the threshold buys no accuracy whatsoever and only destroys coverage --
0.45 was discarding a quarter of all windows for nothing.

`neuroproxy.cli threshold` now derives it from data. The default is 0.20,
**fitted to SCAMPS**, and must be re-derived on real recordings.

### Defect 5 -- the stability metric punished low heart rates

Coverage was 66% and one subject was refused on 91% of its windows despite
full-window HR estimates accurate to 0.5 bpm throughout. Its sub-window
estimates looked like **[55.1, 55.1, 103.2]**: two agreeing exactly, and one
locked onto the **second harmonic**. At a low heart rate the harmonic sits well
inside the 42-180 bpm analysis band, so a short sub-window sometimes picks it.
`np.std` of that triple is 22.7 -- one outlier destroying a non-robust
statistic while the majority agreed perfectly.

Folding harmonics back onto the fundamental was considered and rejected: noise
estimates land at arbitrary ratios that frequently include 2x, so folding
rescues noise as readily as it rescues signal. A median-based spread separates
both without that risk -- 0.0 for `[55.1, 55.1, 103.2]`, 23 for the noise
triple `[129.5, 106.3, 52.7]`.

Effect: the affected subject went from 9% to 91% coverage, pooled coverage from
66% to 75%, worst-subject MAE from 0.82 to 0.66.

A residual limit worth stating: stability cannot detect a *consistently wrong*
estimate. The green baseline locks onto an artefact reliably enough to look
stable, which is why its coverage rose to 57% while its answered MAE is 80 bpm.
Stability tests periodicity, not correctness.

### Coverage: the honest decomposition, and a real frontier

Pooled window coverage of 75% conflates two different failures with different
fixes:

| | value |
|---|---|
| usable-session rate | 8/10 = **80%** |
| coverage within usable sessions | **93%** |
| pooled window coverage | 75% |

The two unusable sessions are overexposed -- 12% and 34% of frame pixels
clipped -- and their counterfactual MAE is 31 and 67 bpm. The signal is not
there, and refusing them is correct behaviour, not a coverage failure of the
engine. That is a capture and onboarding problem (tell the participant to fix
their lighting before the session starts), not a model problem.

`BenchmarkMetrics` now reports the decomposition. The design doc's literal
80% pooled-coverage criterion is **left unchanged and still fails** -- the
decomposition is a diagnostic, not a softer bar.

There is also a genuine frontier here, not just a tuning gap:

| threshold | coverage | MAE | max error |
|---|---|---|---|
| 0.15 | 80% | 1.52 | **49.37** |
| 0.18 | 76% | 0.39 | 1.35 |
| 0.20 | 75% | 0.38 | 1.35 |

Reaching the 80% coverage target on this cohort requires admitting a window
with 49 bpm of error. Coverage and accuracy are in direct conflict below 0.18,
so 75% with every error under 1.35 bpm is the honest operating point.

### What SCAMPS still does not settle

It is rendered, not recorded. Real skin tone response, real camera pipelines,
real ambient light and real motion artefacts remain untested, and the green
baseline collapsing here (72 bpm MAE against POS's 0.53) is a reminder of how
much the ellipse was hiding. The design doc's go/no-go bar stays a question for
UBFC-rPPG and PURE.

Licence: research-only, no commercial use. See docs/datasets.md.

## 13. Respiration: not implemented, because it could not be validated

The design doc lists a respiration proxy from head and torso micro-movement,
marked optional for v1. It is **not implemented**, and the reason is a measured
negative result rather than a scheduling decision.

Two routes were tested on SCAMPS, which provides a breathing waveform (`d_br`)
with rates spread from 8.3 to 21.5 breaths/min across the ten subjects.

**Colour route.** A naive rate estimator on the band-passed ROI trace returned
7.5 breaths/min for *every* subject -- the lower edge of the 0.1-0.6 Hz band.
That is a filter artefact, and taken alone it would have looked like a
systematic bias rather than the absence of any signal.

Cross-correlating the ROI trace against the known breathing waveform gave 0.20
to 0.32 with a lag search, which looks encouraging until it is compared with a
null:

| | mean abs correlation |
|---|---|
| matched subject pairs (n=10) | 0.233 |
| **mismatched subject pairs (n=90)** | **0.234** |

Mann-Whitney, matched greater than mismatched: **p = 0.553**. The correlation
is entirely what narrowband signals produce by chance under a lag search.

**Motion route.** Impossible. Head pose is exactly zero in all ten clips; the
avatars do not move.

So breathing is not observable in this data, by either route. Building an
estimator anyway and reporting a rate would have produced confident, plausible,
meaningless numbers -- the naive version already did.

The lesson generalises past respiration: **a rate estimator on a narrowband
signal will always return a rate.** Validation has to be against a null, not
against whether the output looks reasonable.

## 14. Ocular: blinks are observable, and the detector is validated

Same standard, opposite result. Vertical edge energy in a geometric eye band,
correlated against SCAMPS' `au45` blink ground truth:

| eye measure | matched | mismatched | p |
|---|---|---|---|
| **vertical edge energy** | **0.691** | 0.113 | < 0.0001 |
| mean brightness | 0.336 | 0.117 | < 0.0001 |

Edge energy is the far better carrier: an open eye has strong horizontal
structure (eyelid margins, iris boundary) and a closed eye is a smooth lid.

Event-level performance, matching detected onsets to ground-truth onsets within
300 ms, on 10 subjects and 31 ground-truth blinks:

**precision 0.80, recall 0.60, F1 0.66, blink-rate MAE 3.9/min.**
Seven of ten subjects score F1 >= 0.67; three find nothing at all.

### Three findings worth keeping

**MediaPipe is unusable here.** The design doc specifies MediaPipe Face
Landmarker. Version 1.0.1 crashes inside the graph on macOS arm64 ("Check
failed: service_ Service is unavailable"), with the CPU delegate forced as
well; 0.10.35 no longer ships the legacy `solutions` API. The eye band is
geometric as a result, which is a real downgrade.

**The geometric eye band is far more fragile than the cardiac ROI.** One
subject is a close-up against a skin-toned stone wall in a salmon hoodie, so
the skin blob merges face, wall and clothing and the face box covers 93% of the
frame. The cardiac path survives this -- at that magnification the ROI still
lands on skin, MAE 0.66 bpm -- while the eye band lands on the wrong part of
the image entirely and scores F1 0.00. **Ocular features need accurate
localisation in a way that rPPG does not**, so landmarks matter much more for
this modality than for the pulse.

**A statistical threshold cannot detect rare events.** The first blink detector
thresholded at robust sigmas below the median. Blinks are rare and brief, so
the spread of the series is dominated by the open-eye baseline: on a clean
signal with four blinks the scale estimate was zero and nothing was found. The
criterion is now proportional -- a drop to 90% of the open-eye baseline --
which is scale-free and matches the physics, openness being edge energy with a
meaningful zero.

### What is fitted, and what it cost

The 0.90 threshold is derived from this cohort (`neuroproxy.cli ocular
--sweep`), not chosen, and is fitted to ten rendered, frontal, motionless
faces. It must be re-derived on any real cohort.

One deliberate choice against the measurement: an earlier configuration scored
F1 0.71, but only by accepting one-frame (33 ms) dips as blinks. A blink lasts
100-400 ms. SCAMPS is clean and rendered, so the loose bound costs nothing
there and would generate false positives on real noisy video. The physiological
bound was kept and the lower F1 accepted.

Also fixed while measuring this: the evaluation reported mean F1 0.75 while
recall was 0.13, because subjects the detector found nothing for were scored as
"undefined" and dropped from the mean rather than counted as zero.

## 15. First results on real people (MCD-rPPG)

25 subjects, 50 recordings (rest and post-exercise), 60 s analysed each, 20 s
windows / 5 s stride, 640x480 MPEG-4 from a consumer webcam.

| method | MAE_all | MAE_answered | worst subject | r | SNR dB | coverage |
|---|---|---|---|---|---|---|
| pos | **4.00** | 0.49 | 35.15 | 0.734 | 1.0 | 45% |
| chrom | 4.78 | 0.78 | 32.21 | 0.724 | 0.7 | 44% |
| green | 14.46 | 9.18 | 65.89 | 0.420 | -0.4 | 25% |

**Accuracy passes the design doc bar** (4.00 against 5.0) -- but only just, and
**coverage fails badly** (45% against 80%).

### The size of the reality gap

Everything measured before this was optimistic, and by how much is worth
recording:

| | ellipse | SCAMPS (rendered) | **MCD-rPPG (real)** |
|---|---|---|---|
| POS MAE_all | 0.02 | 0.53 | **4.00** |
| coverage | 100% | 75% | **45%** |
| pulse SNR | 16.6 dB | 2.6 dB | **1.0 dB** |

Coverage decomposes as 52% usable sessions x 76% coverage within them. **Half
of real recordings yield no measurable pulse at all**, against 20% on rendered
faces. That is the dominant engineering problem, not accuracy.

### An apparent subgroup gap that is NOT established

Subgroup reporting -- possible for the first time, because MCD-rPPG ships
demographics -- shows a large difference by sex:

| group | n | median MAE_all | coverage | usable sessions |
|---|---|---|---|---|
| F | 26 | 1.49 | 56% | 62% |
| M | 24 | **8.97** | 33% | 42% |

A six-fold median difference is alarming enough to want to act on. **It should
not be acted on yet**, because it does not survive testing:

- Mann-Whitney, M worse than F: **p = 0.083**
- Usable-session rate, Fisher exact: **p = 0.257**
- Per-subject standard deviation is 10-13 bpm, larger than the difference

Individual results confirm the ambiguity: the worst subjects include both sexes
(two men and three women at 0% coverage), and one of the best is a man at
0.17 bpm.

**A hypothesis tested and rejected.** Inspecting frames from the failing
subjects suggested they sat further from the camera, so face size was tested as
the confound. It is not one: `spearman(face_area, MAE_all) = +0.170, p = 0.24`,
and the sign runs the wrong way -- men's faces are *larger* in frame than
women's (median 0.046 vs 0.037, p = 0.011), not smaller. Stratifying by face
size leaves no significant gap in either half (p = 0.443 small, p = 0.147
large).

So the position is: a suggestive difference, no established cause, and an
underpowered sample. This is resolvable rather than merely unknown -- MCD-rPPG
has **600 subjects** and only 25 were downloaded. Answering it is a download,
not a research programme.

Recording it here because the reverse error would have been easy and costly:
"6x fairness gap by sex" is a publishable-sounding headline that this data does
not support.

### What changed about the strategic picture

The engine now has a real-human, commercially licensed benchmark it can be
iterated against. Accuracy is adequate. **Coverage is the product problem**: at
45% pooled and 52% usable sessions, a research customer sending a link to 100
participants gets useful physiology from about half of them, and the engine
currently has no way to tell them why before the session runs.

## 16. Why real sessions fail, and a check that catches most of it early

Section 15 left coverage as the dominant problem: about half of real sessions
produce no usable pulse. This is the diagnosis.

### One metric separates working sessions from failing ones

Every quality dimension was compared between the 26 usable and 24 unusable
recordings. Exactly one differs:

| metric | usable | unusable | Mann-Whitney p |
|---|---|---|---|
| **skin fraction** | **0.845** | **0.643** | **0.004** |
| face confidence | 0.850 | 0.850 | 0.32 |
| lighting | 0.997 | 0.994 | 0.37 |
| sharpness | 0.970 | 0.971 | 0.90 |
| motion | 0.824 | 0.828 | 0.67 |
| compression | 0.181 | 0.174 | 0.26 |
| valid frame ratio | 1.000 | 1.000 | 1.00 |

It predicts continuously, not just categorically: `spearman(skin_fraction,
HR error) = -0.404 (p = 0.004)`, `spearman(skin_fraction, coverage) = +0.365
(p = 0.009)`, with a sharp cliff near 0.70:

| skin fraction | n | usable | median HR error |
|---|---|---|---|
| 0.27 - 0.71 | 17 | **18%** | 15.65 bpm |
| 0.71 - 0.86 | 16 | 69% | 0.81 bpm |
| 0.86 - 0.98 | 17 | 71% | 0.82 bpm |

Above the cliff the engine is *excellent* -- 0.8 bpm median error. The whole
coverage problem is getting subjects above it.

### This resolves the apparent sex gap from section 15

Skin fraction differs sharply by sex (F 0.845, M 0.660, p = 0.0013), and
**controlling for it removes the gap entirely**:

| stratum | F | M | p |
|---|---|---|---|
| low skin fraction | 5.83 | 9.94 | 0.187 |
| high skin fraction | 0.92 | **0.58** | 0.822 |

In the high-skin stratum men are slightly *better*. So the difference is
mediated by how much clear skin the ROI sees -- facial hair being the obvious
contributor -- not by sex itself. That is a considerably more tractable problem
than an intrinsic demographic bias, and it is fixable in the ROI layer rather
than requiring a different model.

### Two mechanism hypotheses, both tested and both wrong

Worth recording, because both were plausible and acting on either would have
wasted effort:

1. **"Failing subjects sit further from the camera."** Inspecting frames
   suggested it. Rejected: `spearman(face_area, HR error) = +0.170, p = 0.24`,
   and men's faces are *larger* in frame than women's (0.046 vs 0.037,
   p = 0.011), not smaller.

2. **"The shadowed side of the face is rejected as non-skin."** The left cheek
   and forehead are the failing ROIs while the right cheek is fine for
   everyone (0.963 vs 0.979), which looked like directional lighting. Rejected:
   those regions are *brighter* than the face median (1.28 and 1.30 relative),
   and the mask's luma gate only rejects below 0.45.

The residual explanation is that bright, near-specular skin desaturates toward
neutral chroma and falls outside the chroma radius derived from the face
median -- so `adaptive_skin_mask` is still brightness-dependent despite being
designed not to be. **This is not fully characterised and is the most promising
open lead for fixing coverage.**

### The readiness check

`neuroproxy/readiness.py` measures skin fraction from a short preview and
reports a verdict before the session starts. Scored against actual outcomes on
all 50 recordings, with a 2-second preview:

| | usable | failed |
|---|---|---|
| predicted ready | 21 | 10 |
| predicted poor | 5 | 14 |

- **70%** of sessions correctly flagged before running
- Of sessions it passes, **68%** produce a usable pulse, against a **52%**
  base rate -- a 16-point lift
- Of sessions it warns about, **74%** would have failed

**These thresholds were fitted on this same dataset, so this is an in-sample
figure and the honest expectation out of sample is lower.** It is a useful
improvement, not a solution: a fifth of passed sessions still fail, and the
check is predictive rather than diagnostic -- it cannot yet tell a participant
which of hair, beard, glasses or lighting is costing them the skin.

## 17. A coverage fix that looked right and measured wrong

Section 16 identified skin fraction as the only predictor of session failure
but left the mechanism open. Rendering the skin mask over failing subjects
finally showed it, after three colour-based hypotheses had been tested and
rejected:

**The forehead ROI was sitting on the hairline.** ROI boxes are fixed fractions
of the detector box, and a Haar box frames the *head*, not the face -- it
includes hair and varies in how loosely it does so. When the box ran high, the
"forehead" ROI at 12-28% of box height landed on hair.

That diagnosis is almost certainly correct. The fix built on it was not.

### What was tried

`vision.roi.skin_anchor` tightens the box onto the largest connected skin
region; `vision.roi.hairline_row` scans the central band top-down for the first
predominantly-skin row and re-seats the vertical origin there. Rendered side by
side, the ROIs visibly moved off the hair and onto forehead and cheeks.

### What it measured

| MCD-rPPG, POS | fixed boxes | skin-anchored |
|---|---|---|
| MAE_all | **4.00** | 4.28 |
| coverage | **45%** | 40% |
| worst subject | **35.15** | 47.76 |
| usable sessions | 52% | 52% |

Worse on every axis that moved. Reverted; `anchor_to_skin` defaults to False.

### Why, and what it implies

The answer was visible in the same renders, once the measurement said to look
again. A single rigid downward shift moves *all* ROIs: the cheeks slid toward
the jaw and beard, and the forehead toward the eyebrows where expression
movement lives. Hair contamination was traded for beard contamination.

So the placement problem is **per-ROI, not a global offset**. The forehead needs
the hairline; the cheeks need to stay high on the cheekbone regardless of where
the hairline is. Fixing the forehead by moving everything was the wrong shape of
solution.

Recorded because the failure mode is instructive and cheap to repeat: the
change was justified by a correct diagnosis, confirmed by a convincing picture,
and wrong. Nothing here is trusted on a picture.

### The per-ROI version, which worked

Moving only the forehead -- as a band measured downward from the hairline,
leaving the cheeks exactly where they were:

| MCD-rPPG, POS | baseline | all-ROI shift | **forehead only** |
|---|---|---|---|
| MAE_all | 4.00 | 4.28 | **2.87** |
| worst subject | 35.15 | 47.76 | **31.82** |
| usable sessions | 52% | 52% | **58%** |
| pooled coverage | 45% | 40% | 43% |
| coverage within usable | 76% | 71% | 69% |

**A 28% reduction in HR error and six more points of usable sessions**, from
moving one ROI rather than all three. The apparent sex gap narrowed with it
(F 1.22 vs M 6.09, from 1.49 vs 8.97) without any change aimed at it, which is
consistent with section 16: the gap runs through how much clear skin the ROI
sees.

Pooled coverage is slightly down (45% to 43%) while usable sessions are up.
That is not a contradiction: sessions that previously produced nothing now
produce something, and those marginal sessions have lower within-session
coverage, which pulls the within-usable average down (76% to 69%). More people
get a result; the newly-included ones get a patchier one. `usable_session_rate`
is the number that improved and the one a customer feels.

The forehead band is refused rather than applied when it would reach the eyes,
so subjects whose hairline sits very low keep the default box -- the worst
subject in the cohort is one of those, and is not helped by this change.

## 18. ROI selection: large headroom, no usable selector

After the forehead fix, the next question was whether the three ROIs should be
pooled at all. Pooling averages their pixels into one trace, so a single
contaminated region -- a cheek bordering a beard, a forehead partly on hair --
drags the whole estimate with it.

Measured on 16 recordings chosen to include the hardest subjects, mean absolute
HR error per recording:

| selector | median error |
|---|---|
| pooled pixels (current) | 2.99 |
| highest mean pulse SNR | coin flip, 6 of 12 |
| median of per-ROI estimates | 3.83, better on 8 of 16 |
| **oracle** (best available per window) | **0.67** |

Individual cases are dramatic. Subject 2246 at rest: pooled 40.41 bpm error,
while its right cheek alone gives **0.26**. Subject 5078 post-exercise: pooled
18.33, forehead alone 5.05.

### The finding

**There is roughly 4.5x of headroom in ROI selection, and neither obvious
quality proxy reaches any of it.** Both SNR-based selection and cross-ROI
consensus are coin flips against simply pooling.

That is consistent with what section 12 already recorded about the stability
metric: these proxies test whether a signal is *periodic and peaky*, not
whether it is *correct*. A contaminated ROI can produce a clean, stable,
high-SNR estimate of the wrong thing.

Caveat on the oracle: it is the per-window minimum across ROIs plus the pooled
trace, chosen with ground truth in hand. That is the most optimistic possible
oracle -- a real selector fixed per recording would land well short of 0.67.
The gap is a ceiling, not a forecast.

### A third combiner, also a coin flip

Spectral consensus was tried after the other two: extract a BVP per ROI, then
combine their *power spectra* rather than their pixels or their point
estimates. The reasoning was sound -- a real pulse sits at the same frequency
in all three ROIs and reinforces, while an artefact sits at its own frequency
in one and does not. The geometric mean of per-ROI normalised PSDs makes that
explicit: a peak must appear in every ROI to survive.

Paired on identical windows (n = 94):

| combiner | MAE | median |
|---|---|---|
| pooled pixels | 9.07 | 0.80 |
| PSD geometric mean | 8.98 | 0.75 |
| PSD arithmetic mean | 9.65 | 0.64 |

Geometric mean: better on 49 of 94 windows, mean difference **-0.08 bpm**,
Wilcoxon **p = 0.548**. Arithmetic mean: 55 of 94, p = 0.430. Neither is
distinguishable from pooling.

### What it implies for the roadmap

Three combination rules have now been tried over the per-ROI signals -- SNR
selection, HR-estimate consensus, and spectral consensus -- and all three are
coin flips against simply averaging the pixels. The oracle says 4.5x is
available; nothing built on the *extracted signals* reaches any of it.

The natural reading is that the information separating a good ROI from a
contaminated one is not present in the extracted waveforms at all. It is in the
pixels, which is what a learned model sees and a hand-built combiner does not.

This is the strongest argument so far for the neural stage in design doc
section 5.3. A learned rPPG model weights spatial regions implicitly, from
data, rather than from a hand-built quality heuristic -- which is exactly the
step that hand-built heuristics have now twice failed to take. The oracle
number gives that stage a concrete target to beat: **0.67 bpm on these
recordings, against 2.99 today.**

Recorded as a negative result with a number attached, rather than as "ROI
selection might help".

## 19. EfficientPhys: excellent in-domain, unusable across domains

Design doc section 8.3 lists "POS vs EfficientPhys" as a required ablation and
section 5.3 plans to build the state model on a frozen pretrained EfficientPhys
encoder. Both were run.

Pretrained checkpoints from the rPPG-Toolbox release, chosen deliberately
cross-dataset -- testing PURE-trained weights on PURE would measure
memorisation, not the property a product needs.

### In-domain, it clearly beats POS

SCAMPS-trained weights on SCAMPS, 5 subjects, 10 s windows:

| method | MAE_all | answered |
|---|---|---|
| pos | 1.05 | 7/15 |
| **efficientphys (SCAMPS weights)** | **0.21** | **12/15** |
| efficientphys (PURE weights) | 2.21 | 2/15 |

Better error *and* better coverage. The model does what the paper says it does.

### Across domains, it collapses

The same architecture on MCD-rPPG (real subjects, consumer webcam), 25
subjects, 50 recordings:

| method | MAE_all | coverage |
|---|---|---|
| **pos** | **2.87** | **43%** |
| efficientphys (PURE weights) | 31.73 | 11% |
| efficientphys (UBFC-rPPG weights) | 43.96 | 2% |

**11x and 15x worse than a method with no learned parameters at all.**

The in-domain result rules out an adapter bug, which was checked before drawing
any conclusion: if the preprocessing were wrong, the SCAMPS-on-SCAMPS run would
have failed too.

### What this changes in the roadmap

Design doc section 5.3 proposes Phase A as "freeze the pretrained rPPG backbone,
train only our state head". **That plan does not survive this measurement.** A
frozen encoder producing 32-44 bpm of HR error on our distribution is not a
feature extractor for a state model; it is noise with a pedigree. Building a
state head on it would be building on sand, and the failure would surface only
after the state model was already trained and blamed.

The neural path is still worth taking -- section 18 measured 4.5x of headroom
that hand-built heuristics cannot reach, and in-domain performance here shows
the architecture can reach it. But it must be **trained on our own data**, not
adapted head-only from someone else's. That is a substantially larger data
requirement than the design doc assumed, and it reinforces section 6's own
conclusion that consented first-party data is the moat rather than a nicety.

Until then, **POS remains the reference method**, and the ordering is measured
rather than assumed.

### A side result worth keeping

The confidence gate caught this without being told. Coverage fell to 11% and 2%
for the neural methods, meaning the engine refused almost everything they
produced. It had no knowledge that these models were failing; it simply found
no periodic, coherent pulse in their output. That is the abstention system
doing exactly the job section 10 claims for it, on a failure mode it was never
tuned against.

## 20. Fine-tuning pilot: RETRACTED -- it measured a training bug

**The conclusion in this section was wrong and is retracted.** It is kept
because the retraction is more useful than the deletion. What actually happened
is in section 21.

Original claim: fine-tuning on 25 subjects made EfficientPhys worse than the
pretrained model it started from (31.02 vs 25.99 bpm), and therefore the neural
path had no cheap version.

What was actually measured: a training configuration that could not learn
anything at all. Validation loss plateaued at 0.929, i.e. a correlation of
0.07 between prediction and label -- the signature of a loop that is not
training, which should have been checked before drawing any conclusion from it.

The original text follows for the record.

### (retracted) Fine-tuning pilot: 25 subjects is not enough

Section 19 concluded the neural path needs training on our own distribution.
This is the pilot that tested it before committing to the data, run on the 25
subjects already downloaded.

Three arms, same held-out subjects, subject-independent three-way split
(13 train / 4 validation / 8 test, rest and post-exercise always travelling
together):

| arm | MAE | median | p90 | within 5 bpm |
|---|---|---|---|---|
| **POS** | **11.01** | **2.33** | 32.28 | **53%** |
| EfficientPhys pretrained | 25.99 | 23.10 | 43.38 | 12% |
| EfficientPhys fine-tuned | 31.02 | 35.47 | 43.46 | **6%** |

**Fine-tuning made it worse than the pretrained model it started from.**

The training curve says why. Validation loss moved 0.970 to 0.929 and stalled.
The objective is `1 - pearson`, so 0.929 means a correlation of about **0.07**
between predicted and contact BVP -- the model never learned the task at all,
and the weight updates it did make degraded what the pretrained weights already
carried.

That is not surprising in hindsight: 13 training subjects at 30 s each is
roughly **13 minutes of video** for a 2.16M-parameter model.

### What this does and does not settle

**Settled:** at this data scale, fine-tuning is not merely unhelpful, it is
actively harmful. There is no cheap version of the neural path.

**Not settled:** whether 600 subjects would work. That is 24x the data, and the
in-domain result in section 19 shows the architecture can reach 0.21 bpm when
trained on enough of the right distribution. The pilot bounds the low end; it
does not predict the high end.

**Underpowered evaluation:** 32 windows across 8 subjects, because clips were
capped at 30 s for caching. The gaps here (11 vs 26 vs 31) are wide enough to
read directionally, but the absolute values are noisy. POS's own mean of 11.01
against a median of 2.33 shows how much a few bad windows move the mean at this
sample size.

### The decision it forces

Either commit to substantially more MCD-rPPG -- it is CC-BY-4.0, so training on
it commercially is permitted, which no other public rPPG dataset allows -- or
leave the neural path and keep improving the classical one. The pilot removes
the third option of trying it cheaply.

Cost of the first: the full set is 135 GB, and a meaningful subset is tens of
gigabytes plus training time well beyond an afternoon on CPU.

## 21. Three bugs behind a retracted result

Section 20 reported that fine-tuning failed. The standard check for that claim
-- can the loop overfit a handful of examples? -- had not been run. It should
have been, and running it showed the loop could not overfit **four** chunks
after 40 epochs, which no data shortage explains.

Three separate problems were found, in the order they surfaced.

### 1. The model predicts a derivative, and was trained against a waveform

EfficientPhys applies `torch.diff` to its input and is trained on
`DiffNormalized` labels. Its output is the frame-to-frame *change* in blood
volume, not the pulse.

Training it against the raw waveform is not an approximation, it is an
orthogonal target: for a sinusoid the correlation between a signal and its
derivative is exactly zero. The observed plateau at r = 0.07 matches that
prediction almost exactly.

Inference was unaffected -- differentiating preserves frequency, so the PSD
peak and the heart rate are unchanged -- which is why every HR benchmark in
this repo stayed valid while training silently did not.

### 2. The contact PPG is not phase-aligned to the video

Measured across 49 recordings by cross-correlating a POS estimate against the
label:

| | value |
|---|---|
| median waveform correlation at zero lag | 0.344 |
| median after per-recording lag correction | **0.565** |
| recordings above r = 0.4, zero lag | 17 / 49 |
| recordings above r = 0.4, lag corrected | **39 / 49** |
| recordings within +-0.2 s of zero lag | 12 / 49 |
| recordings off by more than 1 s | **22 / 49** |

Lags span -2.68 s to +2.14 s. MCD-rPPG's `ppg_sync` files are aligned in
*rate* but not in *phase*, and the offset differs per recording.

Again this leaves heart rate untouched, which is why it went unnoticed through
every benchmark. It makes waveform training impossible.

`preprocess.estimate_label_lag` now corrects it. Caveat recorded there: the lag
is found using a POS estimate, so a model trained on these labels has seen a
target that POS helped position, and that must be stated when comparing such a
model against POS.

### 3. The learning rate could not move the encoder

Design doc section 5.3 proposes encoder 1e-5, head 1e-3. At 1e-5 the encoder
barely moves, and the pilot ran roughly 195 optimiser updates in total.

The decisive check, on a single 180-frame chunk with dropout disabled:

| optimiser step | correlation |
|---|---|
| 0 | 0.032 |
| 50 | 0.942 |
| 100 | 0.992 |
| 200 | **0.998** |

The loop memorises one example completely. It was never broken -- it was
starved of step size and of steps. Encoder learning rate is now 1e-4 and the
default epoch count is 20.

### What this costs and what it teaches

Section 20's conclusion is withdrawn. Whether fine-tuning helps is **currently
unknown**, not answered, and the pilot has to be rerun.

The lesson is procedural rather than technical: the overfit check is cheap, it
is the standard first question for any training result, and skipping it turned
three findable bugs into a confident negative conclusion that was reported as
fact. Two of the three -- the derivative target and the label lag -- are
invisible to every HR-level metric in this repo, so nothing else would have
caught them.

## 22. Waveform training on MCD-rPPG is blocked by label synchronisation

Section 21 fixed three bugs and the pilot was rerun properly: 70 subjects,
37 train / 12 validation / 21 test, subject-independent, lag-corrected
derivative targets, encoder learning rate 1e-4, 20 epochs.

| arm | MAE | median | within 5 bpm |
|---|---|---|---|
| **POS** | **11.30** | **1.52** | **67%** |
| EfficientPhys pretrained | 32.17 | 24.71 | 13% |
| EfficientPhys fine-tuned | 27.88 | 21.14 | 12% |

Fine-tuning helped a little (32.17 to 27.88) and remains far worse than POS.
But the test scores are not the interesting part. **The training curve is:**

| epoch | train loss | val loss |
|---|---|---|
| 1 | 1.011 | 0.992 |
| 10 | 0.978 | 0.992 |
| 20 | 0.965 | 0.986 |

Train loss of 0.965 is a correlation of **0.035 on the data being fitted**. The
model cannot fit its own training set -- while the single-chunk check in
section 21 reached 0.998 on one example. It can learn one chunk and not 237.

### The cause: chunk-to-chunk alignment is inconsistent by whole pulse periods

Estimating the lag separately on two adjacent 15 s halves of the same
recording, across 12 recordings:

- median absolute difference between halves: **1.54 s**
- 11 of 12 differ by more than 0.3 s
- differences reach 2.4 s and 4.5 s

Half of that is explained by an unavoidable ambiguity: cross-correlating two
*periodic* signals produces near-equal peaks at every multiple of the pulse
period. Expressing each difference in periods, the distance to the nearest
whole period has median 0.19 and 6 of 12 land within 0.15 of an integer -- so
roughly half the estimates are period hops, and the rest look like genuine
drift.

**Either way it is fatal for waveform training.** If chunk A is aligned at lag
L and chunk B at L plus one period, the model is asked to map near-identical
inputs to targets a full cycle out of phase. Averaged over hundreds of chunks
that is not a hard target, it is an inconsistent one, and r = 0.035 is what an
inconsistent target looks like.

### What is and is not blocked

**Blocked:** waveform-level (or derivative-level) training on MCD-rPPG. This is
not a data-volume problem, a learning-rate problem, or an architecture problem.
More subjects will not fix it -- 70 behaved exactly like 25.

**Not blocked:** everything HR-level. Rate is invariant to lag and to period
hops, which is why every benchmark in this repo remained valid throughout, and
why POS scores 1.52 bpm median on the same recordings whose phase cannot be
pinned down.

### The way through: a phase-invariant objective

The obvious move is to stop asking for the waveform. The product needs heart
rate and state, not pulse phase, and a loss defined on the **power spectrum**
-- correlation of PSDs, or cross-entropy over heart-rate bins -- is invariant
to exactly the thing this dataset cannot supply.

That is a real change of plan rather than a tweak, and it is untested. What is
established is that the phase-based objective the architecture ships with
cannot be trained on this data, so it must be replaced or the data must be.

Note also that the alternative -- a dataset with dependable synchronisation --
means UBFC-rPPG or PURE, both of which need access requests, and neither of
which permits commercial use by default.

## 23. The phase-invariant objective works, and is still not enough

Section 22 blocked waveform training on MCD-rPPG and proposed replacing the
objective rather than the data. That was built and tested.

`spectral_cross_entropy` puts the predicted spectrum's mass at the reference
rate: Hann-windowed FFT, zero-padded 4x, softmax over the 0.7-3.0 Hz band,
cross-entropy against a Gaussian centred on the contact-PPG rate. Invariant to
phase, which is precisely what this dataset cannot supply, and it optimises the
quantity the product reports rather than a pulse phase nothing consumes.

Two checks before running anything, in the order that section 21 established
they should have been run:

1. **The loss is minimised at the true rate.** Sweeping a synthetic sinusoid
   from 50 to 110 bpm against a 60 bpm target gives 6.82 / **2.14** / 6.83 /
   12.13 / ... -- minimal at 60 and monotone away from it.
2. **The loop overfits.** Two chunks, dropout off, 40 steps: loss 6.59 to 2.15
   (2.14 is the floor), HR error **29.41 to 0.20 bpm**.

### Held-out result

21 test subjects, subject-independent, scored once:

| arm | MAE | median | p90 | within 5 bpm |
|---|---|---|---|---|
| **POS** | **11.30** | **1.52** | 40.32 | **67%** |
| EfficientPhys pretrained | 32.17 | 24.71 | 64.35 | 13% |
| EfficientPhys, waveform fine-tune | 27.88 | 21.14 | 62.67 | 12% |
| **EfficientPhys, spectral fine-tune** | **17.88** | **12.12** | 39.79 | **28%** |

Against the pretrained model it started from: error nearly halved, median
halved, and the share of windows within 5 bpm more than doubled. Against the
waveform objective it replaces, better on every column. **The diagnosis in
section 22 was right and the fix works.**

It is also still nowhere near POS -- a method with no learned parameters, on
the same windows.

### What limits it now

Validation loss stops improving at epoch 3-5 and the gap to training loss opens
after that, on 134 training chunks -- roughly **22 minutes of video**. That is
the signature of running out of data, not of a broken objective.

Stated as a hypothesis, not a conclusion. Two previous diagnoses in this
sequence were confident and wrong (sections 20 and 21), and the honest position
is that the objective is now demonstrably trainable while the data scale needed
to beat POS is unknown.

### Where this leaves the roadmap

- **POS remains the reference method**, by a wide margin, on real subjects.
- The neural path is no longer blocked -- it has a working objective and a
  validated training loop, which it did not have three sections ago.
- The next lever is data volume: 134 chunks came from 37 training subjects at
  30 s each. MCD-rPPG has 600 subjects at 180 s, roughly 100x what was used.
- What that would cost is tens of gigabytes and training time well past what a
  CPU handles comfortably, and it is a bet, not a plan with a known payoff.

## 24. The realtime path, and what transport costs it

The offline pipeline answers "what was this recording's heart rate". A product
needs a continuous answer while the session runs. `neuroproxy/inference/engine.py`
provides that, and `api/main.py` exposes it.

### Equivalence with the offline pipeline

The engine is frame-source agnostic -- a webcam, a file and a synthetic clip
drive identical code -- and it reproduces the offline result exactly. On a real
recording, 41 matched windows, **maximum difference 0.0000 bpm**. Pinned in CI
by `tests/test_engine.py::test_streaming_matches_offline_exactly`.

That matters more than it sounds: every benchmark, ablation and threshold in
this document was produced offline. Without exact equivalence, none of it would
describe what a live session actually does.

### Output contract

State is reported as **deviation from the subject's own baseline in bpm**, not
as an absolute 0-1 score. Nothing in this project has ever measured an absolute
arousal scale, and emitting one would imply a calibration that does not exist
(limitations 6 and 11). Before the baseline is fitted the engine reports
`state: null, reason: "calibrating"` rather than an uncontextualised number.

Every emission carries quality and confidence, and `state` is null with a reason
whenever the engine will not stand behind a number.

### Lossy transport makes the engine silent, not wrong

The API accepts frames over a WebSocket, so the encoding used in transit is a
design decision with a measurable cost. Same recording, same engine:

| transport | HR median | vs lossless | answered | mean confidence |
|---|---|---|---|---|
| lossless | 78.2 | -- | **57%** | 0.380 |
| JPEG q95 | 77.8 | -0.38 bpm | 54% | 0.362 |
| JPEG q75 | 77.4 | -0.85 bpm | **0%** | 0.320 |

At q75 heart-rate accuracy barely moves -- and the engine answers **nothing**.
Confidence falls just enough to push every window under the abstain threshold.

That is the abstention system behaving correctly on a genuinely degraded
signal, and it is also a product failure: a session that returns no data is
worthless regardless of how principled the refusal was. Note the source video
is already MPEG-4, so this measures a *second* generation of loss, which is
exactly what a naive frame-push API would add.

**Consequences:**

- The frame-push path must not use lossy encoding. It is marked provisional in
  the API docstring for this reason.
- The target architecture (design doc 10.2) extracts features on the client and
  sends only those. That fixes transport loss and the privacy question in the
  same move, and this measurement is the argument for prioritising it.

### The session demo

`apps/web_demo/index.html`, served at `/`, runs a full session: camera consent,
45 s calibration, live timeline, event markers, end-of-session summary.

Two deliberate choices in it are worth recording.

**It sends lossless PNG.** Given the measurement above, JPEG would make the
engine silent. The consequence is roughly 15 MB/s at 640x480, which is fine on
localhost and impossible over a network -- which is precisely the argument for
client-side feature extraction, stated on the page itself rather than hidden.

**It renders calibrating, answered and declined as three states, not two.** The
first version coloured anything without a state red, so the first 45 s of every
session -- normal calibration -- read as failure. A researcher would have
concluded the engine was broken while it was working exactly as designed.
Caught by looking at the rendered page, not the code.

### What exists now, and what does not

Built: streaming engine with exact offline parity, session API with events and
summary, a live camera source that reports whether auto-exposure and
auto-white-balance could actually be disabled rather than assuming it, and a
working single-session demo.

Not built: any state model beyond baseline-relative heart rate, a client-side
feature extractor, a study builder, multi-participant aggregation, or a
researcher dashboard. The engine emits `arousal_proxy` and nothing else,
because heart-rate deviation is the only state signal this project has evidence
for.

## 25. Client-side extraction

Section 24 left the frame-push path having to send lossless PNG -- about
15 MB/s -- because JPEG q75 transport dropped answered windows from 57% to
zero. Extracting features in the browser removes that trade rather than
managing it.

`apps/web_demo/extractor.js` computes, per frame, the spatial mean RGB over the
skin ROIs plus the quality scalars the confidence gate needs. Roughly **100
bytes per frame, ~3 KB/s** against ~15 MB/s, batched one message per second.
The API gained `WS /v1/sessions/{id}/features` alongside the frame path, and
the engine gained `push_features` next to `push`; everything downstream of
ingestion is shared.

Three things follow, and only the first is about bandwidth:

- **Raw video never leaves the device.** This is the privacy position both
  source documents take (design doc 11), reached by having nothing to send
  rather than by promising not to keep it.
- **There is no codec between sensor and signal.** Extraction runs on raw
  canvas pixels, before any encoding, so the compression damage that dominates
  section 3 and section 24 simply does not arise on this path.
- **The frame path stays** for offline replay and clients that cannot run an
  extractor, marked provisional.

### Equivalence, checked rather than assumed

Client-side extraction is only safe if the two implementations agree: every
threshold, ablation and benchmark in this document was measured on the Python
path, and a browser session inherits none of it otherwise.

Four synthetic subjects spanning light to dim skin, identical PNG input to
both:

| fixture | Python mean RGB | JavaScript mean RGB | difference |
|---|---|---|---|
| light | 198, 158, 140 | 198, 158, 140 | **0, 0, 0** |
| medium | 157, 117, 102 | 157, 117, 102 | **0, 0, 0** |
| dark | 76, 54, 43 | 76, 54, 43 | **0, 0, 0** |
| dim | 46, 34, 26 | 46, 34, 26 | **0, 0, 0** |

Mean RGB -- the quantity that drives heart rate -- is exact on all four,
including the two the fixed-locus bug of section 12 would have discarded. The
`lighting` scalar differs by up to 0.014 (dark: 0.867 against 0.881), reaching
the output only through the confidence weighting.

Separately, `tests/test_client_extraction.py` drives a full recording through
both ingest paths and requires identical heart rate to 1e-9, so the two paths
cannot drift apart in the parts that are shared.

**The JavaScript cannot run in pytest.** The test pins the Python reference
values the browser was checked against; if they move, the equivalence has to be
re-verified at `/static/equivalence.html` rather than presumed to hold.

### A note on the verification itself

The first attempt used 640x480 frames from real recordings and never completed
-- decoding a 300 KB PNG timed out in the automated browser pane, which
throttles when hidden. That is a harness limitation, not a code fault, and the
fix was to verify on small controlled fixtures that exercise the same code
paths. Worth recording so the next person does not conclude the extractor is
slow: on a 64x64 frame it runs in 6.8 ms.

## 26. Studies, consent and the guided setup

Three product layers, built in the order that the sensor work implied.

### Persistence

Sessions lived in a module-level dict: a restart lost everything and a second
worker saw nothing of the first. `api/store.py` moves studies, sessions,
samples, events and consent into SQLite. Summaries are now served from the
store, so `tests/test_api.py::test_sessions_survive_a_restart` clears the
in-memory map and still gets the session back.

### Consent, gated on the data path

Both ingest sockets refuse a session with no recorded grant. Putting the gate
on the socket rather than the interface means a client that skips the consent
screen gets nothing, rather than getting through.

Four decisions worth stating:

- **The notice is versioned and stored verbatim with each grant.** A consent
  record that cannot reproduce what the person was shown records nothing, and
  editing the text later would silently re-attribute old agreements. Consent
  against a superseded version is rejected with 409.
- **No partial consent.** There is no mode where the camera runs but the
  measurements are not kept, so the page does not offer a checkbox pretending
  otherwise.
- **Withdrawal erases.** Samples, events, consent and the session row are
  deleted. Flagging a row and trusting every future query to filter it is not
  withdrawal.
- **Retention has no unbounded option** and is applied by a purge endpoint.
  Retention that is documented but never executed is a policy, not a control.

The notice itself claims only what the code does, including that the engine
**will often decline to answer** -- roughly half of real sessions yield little
(sections 15-16). Telling participants that up front is more honest than a
progress bar implying success.

Legal posture is in `docs/privacy.md`, kept deliberately separate: the
mechanisms are engineering and are done; whether a study is lawful is not, and
is not claimed here.

### Guided setup

Section 16 found that skin fraction is the only pre-session predictor of
whether a session will work, and that it is measurable in about two seconds.
The setup step now uses it live: a gauge, and coaching that names the usual
causes -- hair over the forehead, glasses, a hard side light -- rather than a
verdict. The start button unlocks only after the reading has been stable for
~1.5 s, so one lucky frame cannot let a doomed session through.

**This is untested against outcomes.** Whether coaching actually moves skin
fraction, and whether that converts failing sessions into working ones, needs
real participants. The thresholds it coaches toward (0.70 marginal, 0.86 good)
are in-sample from 25 subjects and will need re-deriving.

### The visualisation refuses too

The session view is a WebGL orb that beats at the measured rate. When the
engine declines, the orb **stops and desaturates** rather than drifting on the
last known value.

That is not decoration. A visualisation that keeps animating through an
abstention quietly undoes the thing the confidence system exists to do: a
participant or researcher watching a smoothly pulsing orb has no way to know
the number underneath it is two minutes stale. Verified in a browser in both
states.
