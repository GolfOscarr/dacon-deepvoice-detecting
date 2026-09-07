# 02 — Audio Competitions (the craft)

BirdCLEF is the deepest body of practical audio-ML knowledge on Kaggle: seven annual editions of
the same core problem — **clean training clips → noisy field recordings**, weak labels,
threshold-free ranking metric, hard inference runtime cap. Structurally our problem.

---

## BirdCLEF strategy playbook ★ [Benhamou, BirdCLEF+ 2026 playbook (PDF)](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/Birdclef_2026.pdf)

A distilled 50-page methodology. The highest-density source found. Quoted material below is ★.

### Six winning principles (verbatim headings)

1. **Validation first** — "Decide the fold logic before tuning models; otherwise the leaderboard will mislead you."
2. **Build a baseline ladder** — "do not jump directly to a giant ensemble."
3. **Keep inference in loop** — "Any promising idea must remain compatible with the final CPU notebook."
4. **Exploit unlabeled audio** — "BirdCLEF winners repeatedly gain from pseudo labels, soundscape reuse, and distillation." ❌ *for us — see [README](README.md)*
5. **Optimize diversity** — "Ensembling works when models differ in backbone, features, labels, or training recipe."
6. **Track every experiment** — "Model quality without lineage is not usable in the final push."

> **Working rule**: "Every branch should answer one question only: better validation score? more
> stable across folds? faster inference? more ensemble diversity? If the answer is unclear, prune it."

### Decision hierarchy for promoting an idea

```
1) Mean OOF score  2) Fold stability  3) Subgroup robustness  4) Runtime  5) Public LB (weak cross-check only)
```
> "A model with slightly lower mean but lower fold variance is often a better final candidate."

### CV scheme guidance

| Scheme | Verdict |
|---|---|
| Random stratification | "Fast and often optimistic. Good only as a sanity baseline." |
| **Group split by site/session** | "Best when metadata identifies shared acoustic conditions." |
| Time-aware split | For seasonality / acquisition waves |
| **Hybrid grouped + stratified** | "Usually the strongest practical option" |

Plus: **5 grouped folds as default**, rebalanced so rare classes appear often enough, and
**"one shadow split for stress testing domain shift"** kept separate from the tuning split.

### Augmentation plan (their risk tiering)

| Safe | Usually useful | Case-by-case | Use with care |
|---|---|---|---|
| time shift · gain · mild noise | SpecAugment · **background mix** · random crop | bandpass filters · tempo perturbation · **denoise transforms** | "heavy warping or unrealistic acoustic distortions" |

> "Augmentations should improve robustness to real field conditions … **without changing class
> semantics** so much that the offline metric becomes noisy."

🔴 Note their placement of **tempo perturbation and denoise transforms in "case-by-case"** — this
independently corroborates our decision to keep pitch/time-stretch at low rates
([data/06](../data/06-augmentation-spec.md)).

### Other high-value rules

- **Order of importance**: `1) good data split → 2) good sampler/loss → 3) architecture → 4) optimizer refinements`.
  "Common mistake: over-searching schedules before solving data shift and imbalance."
- **Loss, sampler and augmentation interact — "tune them as a small package, not in isolation."**
- **Fast replay protocol**: triage ideas on 1 fold / limited classes / fixed seed before full CV.
  "The best teams usually operate multiple experimental speeds: replay, medium CV, full final validation."
- **Sampling rule**: "Use sampling to make rare classes visible — not to create a fake training
  distribution." Watch calibration damage from aggressive oversampling.
- **Leakage guardrails**: fold assignment generated once and reused everywhere; every feature table
  carries a sample id + provenance column; **"OOF predictions, not public-LB scores, decide whether
  an idea survives."**
- **Segment grid**: default 5 s; try 2.5–3 s for brief events, 8–10 s for context. Mel 128–256 bins.
  "Keep this grid small."
- **Two submission tracks**: "a conservative, proven blend and a higher-upside experimental blend."

### EDA checklist (Phase 1)
Listen to samples · browse spectrograms (call shape, silence ratio, harmonics, clipping) · class
counts · duration bias · metadata audit (map every field to **feature / split key / leakage risk**)
· co-occurrence. Standard views: waveform, log-mel, **PCEN or denoised view**, **per-band
energy / SNR proxy** to detect low-information clips.

> "Do not advance to hyperparameter tuning until the data memo explains class imbalance, domain
> shift, and the first leakage hypothesis."

---

## BirdCLEF 2024 — top-3 techniques ☆ [aggregated knowledge file](https://github.com/Galaxy-Dawn/claude-scholar/blob/main/skills/kaggle-learner/references/knowledge/time-series/birdclef-2024.md)

Metric: macro ROC-AUC. 120-min **CPU-only** inference cap. Note the public→private drop:
**public 0.729 → private 0.690** for the winner.

### 1st — Team Kefir ☆
| Technique | Detail |
|---|---|
| **Statistics-T noise filtering** | `T = std + var + rms + pwr`, filter at the 0.8 quantile to drop low-quality chunks |
| External pre-labeling | Google bird vocalization model used to validate annotations and produce pseudo-labels (coefficient 0.05) |
| **CE loss + sigmoid inference** | Train multi-class with softmax; apply **sigmoid at inference** for multi-label output. Worked because most clips contain 1–2 species |
| **`min()` ensemble** | Take the **minimum** probability across 6 models (3 EfficientNet + 3 RegNetY) instead of averaging — reduces uncertain/false-positive predictions |
| OpenVINO | Fixed input sizes for CPU speed |

### 2nd — ADSR ☆
Three cycles of ensemble pseudo-labeling (25–45% weighting) · **checkpoint soup** (average weights
from epochs 13–50 instead of early stopping) · neighbour-window smoothing at 0.5× weight.

### 3rd — NVBird ★ [repo](https://github.com/TheoViel/kaggle_birdclef2024)
- Two-stage: level-1 CNNs (EfficientNet, MobileNet, TinyNet, MNASNet, MixNet) + EfficientViT
  (b0/b1/m3) on 224×224 log-mel → level-2 EfficientViT-b0 + MNASNet-100 trained on pseudo-labels
- **Mixup with `max()` labels**: "mixed labels are the max of the labels of the two audios"
- Random crop of 5 s from the **first 6 s (or optionally last 6 s)**; 1-second time shift
- Capped records per class at 500, keeping the most recent
- Large batch (128) when mixing pseudo-labels with clean data at similar quantity
- ONNX; "5 folds take 40 minutes to submit"

**Transfer to us**: `min()` ensembling and mixup-with-max are directly usable. The `max()` label rule
is precisely how our composition assigns labels when mixing components
([data/02](../data/02-label-taxonomy.md)) — independent confirmation.

---

## BirdCLEF 2025 — 2nd place ★ [repo](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)

Public 0.925 / private 0.928. Title: *"Tackling Domain Shift in Bird Audio Classification via
Transfer Learning and Semi-Supervised Distillation"*.

| | |
|---|---|
| Backbones | `tf_efficientnetv2_s_in21k`, `eca_nfnet_l0` — both **pre-trained on extended bird audio first**, then fine-tuned |
| Clips | 5 s; audio precomputed to HDF5 |
| Loss | **Focal BCE with label smoothing** |
| Balancing | square-root balancing and equal balancing variants |
| Optimizer | AdamW 1e-4 / RAdam 1e-3, cosine to 1e-6, batch 64, 50 epochs, 5-fold |
| Pseudo-label params | threshold 0.5, min-threshold 0.1, confidence 0.4, **2–3 iterations**, minor oversampling of rare classes |
| Deployment | ONNX → **fp16 OpenVINO** |

**Transfer to us**: two-stage pretraining (domain-adjacent audio → task) maps onto
*pretrain on broad real/fake audio → fine-tune on the competition label structure*. Focal BCE +
label smoothing is a sensible default for our imbalanced multi-head setup.

## BirdCLEF 2025/2026 — 1st place ☆
Nikita Babych, *"Multi-Iterative Noisy Student Is All You Need"* (2025) and
*"Noisy Student Meets Distillation"* (2026). Recurring recipe: iterative teacher→student
self-training with confidence-filtered pseudo-labels, then distillation into a fast student.
☆ Reported gains from pseudo-labeling alone: **+0.003–0.005** on both public and private.
❌ Not directly usable on DACON's test set — apply to our own held-out proxy pool instead.

---

## Cornell Birdcall Identification (2020) — 1st place ★ [Ryan Wong's write-up](https://ryaniswong.com/post/kaggle-bsr/)

The canonical domain-gap competition: train on isolated xeno-canto recordings, test on ~150
ten-minute field soundscapes.

- **SED architecture from PANNs**, CNN feature extractor swapped to pretrained **DenseNet121**;
  `torch.tanh` instead of `torch.clamp` in the attention layer; AttBlock reduced to 1024 — all to
  fight overfitting with ≤100 labelled samples per class
- **Augmentations**: mixup, **pink noise ("very important")**, SpecAugment, 30-second clips
- AdamW + weight decay, LR warmup + cosine annealing
- **Ensemble of 13 models with voting**: require **4 of 13** models to agree; clipwise threshold
  0.3 **and** framewise threshold 0.3, "reducing the number of false positives"
- **Model selection on F1 that didn't sacrifice precision**, not on CV/LB correlation

**Transfer to us** 🔴:
- **Pink noise specifically** (not white) was the domain-gap fix. Cheap to add to
  [data/06](../data/06-augmentation-spec.md).
- **Clipwise + framewise dual outputs** is exactly the structure we need: file-level
  `*_FAKE_PROB` plus segment-level supervision ([survey/01 PartialSpoof](../survey/01-sota-speech.md)).
- **Voting-based agreement** across an ensemble is a third aggregation option alongside `min()`
  and geometric mean.

---

## Freesound Audio Tagging 2019 — 1st place ★ [`lRomul/argus-freesound`](https://github.com/lRomul/argus-freesound)

Multi-label, 80 classes, small curated set + large noisy web set. Metric lwlrap (ranking).

**Mel parameters** (concrete, worth copying as a starting grid):
`sr 44100 · n_fft 2560 · hop 690 · 128 mel bins · fmin 20 · fmax 22050 · min duration 0.5 s`

**Model**: CNN with attention, skip connections, and **auxiliary classifiers at intermediate layers**.

**Augmentations**:
- PadToSize (wrap or constant), random crop to 256 time steps
- **RandomResizedCrop** (scale 0.8–1.0, ratio 1.7–2.3) at p=0.33 — the author notes:
  > "random resize crop helps a lot, but I can't explain why"
- **SpecAugment**: 2 masks, 15% frequency masking, 20% time masking, p=0.5
- **Delta and delta-delta features as 2 extra channels**
- **MixUp**, plus **SigmoidConcatMixer** — "smooth (sigmoid-based) transition from one audio clip
  to another over time"

**Noisy-label handling** 🔴:
- **Different loss per data tier**: BCE on curated, **Lsoft (β=0.7) on noisy**
- Manual relabeling of curated samples with low prediction scores from earlier models
- Train with BCE on noisy samples that earlier models scored highly
- Different sampling probability for curated vs noisy

**Training**: Adam lr 9e-4, ReduceLROnPlateau (patience 6, factor 0.6), batch 128 with AMP, 5 folds.
**Ensemble**: geometric mean of 7 first-level models and 3 second-level MLPs.

**Transfer to us** 🔴:
- **SigmoidConcatMixer is almost exactly our sequential-composition operator** — a smooth
  transition between two clips. Use it as the crossfade in [data/06](../data/06-augmentation-spec.md)
  rather than a hard cut.
- **Per-tier losses** map onto our label-confidence tiers: exact labels for self-generated audio,
  weaker confidence for public corpora.
- Delta/delta-delta as extra channels is a nearly free input-representation upgrade.

---

## Bengali.AI Speech Recognition (2023) ☆

Test set was **OOD by construction** — 17 domains absent from training (TV drama, audiobook, talk
show, online class, sermons). Top solutions used `audiomentations` with **heavy augmentation for
read speech and lighter augmentation for spontaneous speech**, including `TimeStretch`,
`RoomSimulator`, `AddBackgroundNoise`, `Gain`.

**Transfer to us**: *augmentation strength should be conditioned on source cleanliness* — heavy on
clean studio corpora (LibriTTS, VCTK), light on already-degraded sources (ASVspoof21 LA telephony).
A detail we had not specified.
