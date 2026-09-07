# 03 — Weak Signal & Label Noise

Competitions about finding a faint deterministic signature buried in noise — our
"detect a generator artifact under music at −15 dB" problem.

---

## G2Net Gravitational Wave Detection (2021) — top-3 ★ [`ZiyueWang25/Kaggle_G2Net`](https://github.com/ZiyueWang25/Kaggle_G2Net)

*(repo states 3rd place; metric AUC; 1,219 teams)*

### 🔴 The transferable insight: low-SNR curriculum

> Deep learning generalizes from **low-SNR to high-SNR** signals **but not vice versa** —
> curriculum learning via synthetic injection proved essential.

They injected simulated signals at low SNR, sampling `SNR ~ max(N(3.6, 1), 1)`, and varied the
physical parameters that shape waveform morphology (total mass, mass ratio). **Gain: 2–8 bps.**

**Transfer to us** 🔴: our mixing-gain sweep should be **skewed toward the hard, quiet end**, not
uniform. A model trained on prominent fake components will not detect quiet ones; a model trained
on quiet ones handles both. This changes the gain distribution in
[data/02](../data/02-label-taxonomy.md) from `U(−15,+15)` to something biased toward low
component-SNR. Worth an explicit ablation.

### Signal conditioning
- **Custom whitening**: standard packages corrupted boundaries on short 2 s signals, so they
  computed the **average PSD over all negative training samples**, extended signals preserving
  derivative continuity at the boundaries, applied a **Tukey window (α=0.5)**, then whitened.
- 2D branch: **CQT** (`sr=2048, fmin=20, fmax=1000, 48 bins/octave`) on whitened data → 128×512.

**Transfer**: normalizing by a *class-conditional average spectrum estimated from the REAL class*
is a neat trick — it removes the common channel and leaves the anomaly. Directly testable as a
front-end for the fake heads.

### Architecture insight
1D models beat 2D. The winning design recognized that the signal is **"not just a signal of the
specific shape, but rather a correlation in signal between multiple detectors"** — so the network
used **separate per-detector branches with shared weights**, merging only at intermediate layers,
with progressive concatenation to absorb 10–20 ms timing shifts.

Best single 1D: CV/public/private **0.8819 / 0.8827 / 0.8820**. Best single 2D: 0.87875 / 0.8805 / 0.8787.

**Transfer to us**: our stereo channels and our voice/music components are analogous "detectors."
A shared-weight two-branch design merging late is a plausible architecture for the mixed case —
and the tiny CV↔private gap here shows what a well-designed validation looks like.

### Augmentation & training
1D: vertical flip, **channel shuffling**, Gaussian noise, time shift, temporal masking,
**64-fold MC dropout** (~1 bps). 2D: channel swapping as train-time aug **and TTA**;
**mixup applied in 1D before the CQT transform**.
Then: pretraining on simulated signals → soft pseudo-labeling (4–6 epochs) → **rank loss at low LR
for 2 epochs** (~1 bps).

**Transfer**: mixup *before* the spectrogram transform (i.e. on the waveform) is the correct order
for us too, since our composition happens in the time domain. A short **rank-loss fine-tune at the
end** is a cheap, metric-aligned trick for an EER competition.

### Ensembling & robustness
**CMA-ES optimization with logistic transformation** for blend weights beat scipy and NN blenders.
Final: 15×1D + 8×2D → CV 0.8829.
**Adversarial validation** confirmed train/test similarity (AUC 0.5). They also submitted a
variant simulating private-LB conditions by excluding 16% bootstrap samples.

**Transfer**: adversarial validation (train a classifier to distinguish train from test) is a
cheap leak/shift detector. ⚠️ We only have 3 dummy test files, so we can run it only against our
own splits — still useful for detecting corpus-identity confounds
([data/09 R2](../data/09-risks-and-checks.md)).

---

## SETI Breakthrough Listen (2021) ☆

**Setup worth noting**: no confirmed positives exist, so organizers **injected simulated "needles"
into real telescope noise** — the same construction we are forced into. Each sample is a *cadence*
of 6 spectrograms: 3 "on-target" and 3 "off-target"; a signal present in on-target but absent in
off-target is positive, one present in both is a human artifact.

**Transfer to us**: the on/off-target contrast is a **paired-negative design** — the same signal
chain, differing only in whether the phenomenon is present. That is exactly our T3 vocoder-twin
construction ([data/05](../data/05-synthesis-plan.md#paired-design)): identical channel, only the
synthesis artifact differs. Independent evidence that this is the right way to build data when
positives are synthetic.

---

## HMS – Harmful Brain Activity Classification (2024) ☆

Spectrogram classification with **heavily noisy labels** (expert vote counts varied per sample).

**Two-stage training** (recurring across top solutions):
1. Stage 1: train only on samples with **≥10 expert votes** (high-confidence consensus)
2. Stage 2: fine-tune on the full dataset including low-vote samples, with **progressively reduced
   weight for lower-vote data**

**Transfer to us**: our corpus has natural confidence tiers — self-generated audio has *exact*
labels; public corpora have *asserted* labels; separated pseudo-stems have *inferred* labels.
Train stage 1 on exact-label data only, then fine-tune on everything with down-weighting. Pairs
with the per-tier loss idea from `[Freesound 2019, 1st]` in [02](02-audio-classification.md).
