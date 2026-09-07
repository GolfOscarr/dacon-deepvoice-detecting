# 05 — Transferable Playbook

Everything above, reorganized by your four themes. Each item tagged with its source competition.
🔴 = changes our current plan · ➕ = new idea we had not planned · ✅ = confirms an existing choice.

---

## A. EDA

| # | Practice | Source |
|---|---|---|
| A1 | ✅ **Standardize four inspection views**: waveform (clipping, silence, amplitude drift), log-mel, **PCEN / denoised view**, and **per-band energy / SNR proxy to detect low-information clips** | ★ `[BirdCLEF playbook 2026]` |
| A2 | 🔴 **Map every metadata field to one of three roles: feature, split key, or leakage risk.** Quantify missingness and cardinality before using any field | ★ `[BirdCLEF playbook 2026]` |
| A3 | ✅ Produce a **one-page data memo** before modeling: file inventory, schema, label counts, suspected leakage variables, risk list. "Do not advance to hyperparameter tuning until the data memo explains class imbalance, domain shift, and the first leakage hypothesis" | ★ `[BirdCLEF playbook 2026]` |
| A4 | ➕ **Adversarial validation** — train a classifier to separate train from test/proxy. AUC ≈ 0.5 means distributions match | ★ `[G2Net 2021, 3rd]` |
| A5 | ➕ **Statistics-T quality filter**: `T = std + var + rms + pwr`, drop below the 0.8 quantile, to remove low-information audio chunks | ☆ `[BirdCLEF 2024, 1st]` |
| A6 | ⚠️ **Duration-vs-label bias is a known trap in this exact dataset family** — FoR ships a `for-2sec` variant purely to eliminate it | ☆ `[FoR dataset]` |
| A7 | ✅ Automate: hash duplicates, near-duplicates, label-map consistency, corrupt audio, fold leakage. Manually review: suspicious top-scoring validation clips, impossible co-occurrences | ★ `[BirdCLEF playbook 2026]` |

## B. Augmentation

| # | Practice | Source |
|---|---|---|
| B1 | 🔴 **Risk-tier your augmentations.** Safe: time shift, gain, mild noise. Usually useful: SpecAugment, background mix, random crop. **Case-by-case: bandpass, tempo perturbation, denoise transforms.** Use with care: heavy warping | ★ `[BirdCLEF playbook 2026]` |
| B2 | 🔴 **Pink noise specifically** (not white) was called "very important" for closing the clean→field domain gap | ★ `[Cornell 2020, 1st]` |
| B3 | 🔴 **SigmoidConcatMixer** — "smooth (sigmoid-based) transition from one audio-clip to another over time". This *is* our sequential-composition operator; use it instead of hard cuts | ★ `[Freesound 2019, 1st]` |
| B4 | ✅ **Mixup with `max()` labels** for multi-label audio; mix additively on the **waveform, before** the spectrogram transform | ★ `[BirdCLEF 2024, 3rd]`, ★ `[G2Net 2021, 3rd]` |
| B5 | 🔴 **Structured dropout** (GridMask-style removal of regions) to prevent reliance on any single artifact area — our analogue is masking frequency bands / time regions | ★ `[DFDC 2020, 1st]` |
| B6 | 🔴 **Vary the interpolation/resampling method itself**, so no single processing path becomes the signal | ★ `[DFDC 2020, 1st]` |
| B7 | 🔴 **Condition augmentation strength on source cleanliness** — heavy for clean read speech, light for already-degraded spontaneous audio | ☆ `[Bengali.AI 2023, top]` |
| B8 | ✅ Codec/compression augmentation with a **quality range** (`quality_lower=60, quality_upper=100, p=0.5`) rather than a fixed setting | ★ `[DFDC 2020, 1st]` |
| B9 | ✅ SpecAugment concrete params: **2 masks, 15% frequency, 20% time, p=0.5** | ★ `[Freesound 2019, 1st]` |
| B10 | ➕ **RandomResizedCrop on the spectrogram** (scale 0.8–1.0, ratio 1.7–2.3, p=0.33) — "helps a lot, but I can't explain why" | ★ `[Freesound 2019, 1st]` |
| B11 | ✅ `audiomentations` stack: `TimeStretch`, `RoomSimulator`, `AddBackgroundNoise`, `Gain` | ☆ `[Bengali.AI 2023, top]` |
| B12 | ➕ Reverb via **impulse-response convolution at p=0.2 with a dry/wet mix control** | ☆ `[birdcall community]` |
| B13 | 🔴 Selection principle: augmentations must improve robustness to real conditions **"without changing class semantics so much that the offline metric becomes noisy"** | ★ `[BirdCLEF playbook 2026]` |

## C. Data synthesis & sourcing

| # | Practice | Source |
|---|---|---|
| C1 | 🔴 **Fine-tune the generators on the target domain before generating fakes** — they CLM-tuned 9 LLM families on the in-domain corpus so fakes matched the domain, not just the task | ★ `[LLM-Detect-AI 2024, 1st]` |
| C2 | ➕ **Generate deliberately adversarial samples** with instruction-tuned models, not just typical ones | ★ `[LLM-Detect-AI 2024, 1st]` |
| C3 | 🔴 **Distribution matching beats volume.** "Adding other data caused severe data drift which further increased the CV/LB gap." The winning submission was a simple model over carefully curated data | ☆ `[LLM-Detect-AI 2024, efficiency prize]` |
| C4 | 🔴 **Skew synthetic-signal injection toward LOW SNR.** `SNR ~ max(N(3.6,1),1)`. Models generalize low→high but not high→low. Gain 2–8 bps | ★ `[G2Net 2021, 3rd]` |
| C5 | ➕ **Vary the parameters that shape the artifact morphology**, not just the count — for them mass/mass-ratio; for us sampling temperature, CFG, seed, speaker, vocoder | ★ `[G2Net 2021, 3rd]` |
| C6 | ✅ **Paired positive/negative through an identical channel** (on-target vs off-target cadence) is the right construction when positives are synthetic | ☆ `[SETI 2021]` |
| C7 | ✅ Cap samples per class (e.g. 500, keeping most recent) to control head-class dominance | ★ `[BirdCLEF 2024, 3rd]` |
| C8 | ✅ **Two-stage pretrain**: broad domain-adjacent audio first, then task fine-tune | ★ `[BirdCLEF 2025, 2nd]` |

## D. Feature engineering & input representation

| # | Practice | Source |
|---|---|---|
| D1 | ➕ **Delta and delta-delta as 2 extra input channels** — nearly free | ★ `[Freesound 2019, 1st]` |
| D2 | 🔴 **Whiten by an average PSD estimated over the negative (real) class**, with Tukey window (α=0.5) and boundary-continuous extension for short clips. Removes the common channel, leaves the anomaly | ★ `[G2Net 2021, 3rd]` |
| D3 | ➕ **PCEN** as an alternative front-end when background energy dominates | ★ `[BirdCLEF playbook 2026]` |
| D4 | ✅ Mel grid to search: **128–256 bins**, task-aware low/high cut. Segment 5 s default, 2.5–3 s for brief events, 8–10 s for context. "Keep this grid small" | ★ `[BirdCLEF playbook 2026]` |
| D5 | ✅ Concrete mel starting point: `sr 44100 · n_fft 2560 · hop 690 · 128 mels · fmin 20 · fmax 22050` (rescale for our 16 kHz) | ★ `[Freesound 2019, 1st]` |
| D6 | ➕ **Auxiliary classifiers at intermediate layers** | ★ `[Freesound 2019, 1st]` |
| D7 | 🔴 **Separate branches with shared weights, merged late**, when the signal is a *relationship* between channels rather than a shape in one | ★ `[G2Net 2021, 3rd]` |
| D8 | ✅ Classical-feature baseline (MFCC/log-mel/PCEN summaries + linear or tree model) as a **bug detector** — "should be clearly worse than the best deep baseline, otherwise something may be wrong in the deep pipeline" | ★ `[BirdCLEF playbook 2026]` |

## E. Validation design

| # | Practice | Source |
|---|---|---|
| E1 | 🔴 **Decision hierarchy**: OOF mean → fold stability → subgroup robustness → runtime → public LB as weak cross-check only. **"OOF predictions, not public-LB scores, decide whether an idea survives"** | ★ `[BirdCLEF playbook 2026]` |
| E2 | 🔴 **Hybrid grouped + stratified 5-fold** is "usually the strongest practical option"; random stratification is "fast and often optimistic, good only as a sanity baseline" | ★ `[BirdCLEF playbook 2026]` |
| E3 | 🔴 **Keep a separate "shadow split" purely for domain-shift stress testing**, distinct from the tuning split | ★ `[BirdCLEF playbook 2026]` |
| E4 | ✅ **Fold assignment generated once and reused everywhere**; every feature table carries a sample id + provenance column | ★ `[BirdCLEF playbook 2026]` |
| E5 | 🔴 **"A model with slightly lower mean but lower fold variance is often a better final candidate"** | ★ `[BirdCLEF playbook 2026]` |
| E6 | ➕ **Fast replay protocol**: triage new ideas on 1 fold / limited classes / fixed seed before full CV. Operate at three speeds: replay, medium CV, full validation | ★ `[BirdCLEF playbook 2026]` |
| E7 | ✅ Track **fold variance, seed variance, subgroup robustness, runtime margin, reproducibility** — mean alone is weak evidence | ★ `[BirdCLEF playbook 2026]` |
| E8 | 🔴 **Order of importance**: `1) data split → 2) sampler/loss → 3) architecture → 4) optimizer`. "Common mistake: over-searching schedules before solving data shift and imbalance" | ★ `[BirdCLEF playbook 2026]` |
| E9 | ✅ **Tune loss, sampler and augmentation as a package**, not in isolation | ★ `[BirdCLEF playbook 2026]` |
| E10 | ➕ Select models on a metric that **doesn't sacrifice precision**, rather than on CV/LB correlation | ★ `[Cornell 2020, 1st]` |

## F. Training & label noise

| # | Practice | Source |
|---|---|---|
| F1 | 🔴 **Different loss per label-confidence tier**: BCE on trusted data, **Lsoft (β=0.7)** on noisy data; different sampling probability per tier | ★ `[Freesound 2019, 1st]` |
| F2 | 🔴 **Two-stage clean→noisy**: train on high-confidence subset, then fine-tune on everything with progressively reduced weight for low-confidence samples | ☆ `[HMS 2024, top]` |
| F3 | ✅ **Focal BCE + label smoothing** for imbalanced multi-label audio | ★ `[BirdCLEF 2025, 2nd]` |
| F4 | ➕ **Rank-loss fine-tune at low LR for ~2 epochs at the end** (~1 bps) — aligns the model with a ranking metric | ★ `[G2Net 2021, 3rd]` |
| F5 | ➕ **Checkpoint soup**: average weights across many epochs instead of early stopping | ☆ `[BirdCLEF 2024, 2nd]` |
| F6 | ✅ **"Use sampling to make rare classes visible — not to create a fake training distribution."** Watch calibration damage from aggressive oversampling | ★ `[BirdCLEF playbook 2026]` |
| F7 | ➕ **CE + softmax training, sigmoid at inference** — valid when most clips carry 1–2 positives | ☆ `[BirdCLEF 2024, 1st]` |
| F8 | ✅ Ranking metric ⇒ "the best loss may be the one that yields the strongest OOF ordering rather than the nicest raw probabilities" | ★ `[BirdCLEF playbook 2026]` |

## G. Inference, pooling & ensembling

| # | Practice | Source |
|---|---|---|
| G1 | 🔴 **Aggregation beats architecture.** A confidence-gated frame-aggregation rule moved a single model **0.25 → 0.22**: mean of predictions >0.8 if >11 frames exceed it and they are >40% of frames; mean of <0.2 if >90% are below; else plain mean | ★ `[DFDC 2020, 1st]` |
| G2 | 🔴 **The ensemble aggregation function is a hyperparameter**: `min()` across models to cut false positives ☆ `[BirdCLEF 2024, 1st]`; geometric mean ★ `[Freesound 2019, 1st]`; **CMA-ES-optimized logit blend** ★ `[G2Net 2021, 3rd]`; 4-of-13 voting ★ `[Cornell 2020, 1st]` |
| G3 | ✅ **Dual clipwise + framewise outputs, thresholded separately** (both at 0.3) | ★ `[Cornell 2020, 1st]` |
| G4 | ✅ Multi-crop / TTA with channel swapping; **64-fold MC dropout** (~1 bps) | ★ `[G2Net 2021, 3rd]` |
| G5 | ✅ **Ensemble only models that are both strong and different.** "Three nearly identical models rarely beat two diverse ones." Check OOF correlation | ★ `[BirdCLEF playbook 2026]` |
| G6 | ✅ **Start with weighted blending before any stacker**; "a fragile stacker can fail under time pressure" | ★ `[BirdCLEF playbook 2026]` |
| G7 | ✅ ONNX → **fp16 OpenVINO**; B0-class backbones; "5 folds take 40 minutes to submit" | ★ `[BirdCLEF 2024, 3rd]`, ★ `[BirdCLEF 2025, 2nd]` |
| G8 | 🔴 **"A model that cannot survive notebook deployment is a teacher candidate, not a final submission candidate."** | ★ `[BirdCLEF playbook 2026]` |
| G9 | ➕ **Keep two submission tracks**: a conservative proven blend and a higher-upside experimental blend | ★ `[BirdCLEF playbook 2026]` |

---

## ❌ What does NOT transfer (blocked by our rules)

| Kaggle staple | Why it's blocked |
|---|---|
| **Pseudo-labeling / noisy-student on the test set** — the single most repeated BirdCLEF winner | DACON rule 2.3 explicitly names Pseudo-Labeling on 비공개 평가 데이터셋 as prohibited. **Workaround**: run the identical recipe on our own held-out proxy pool |
| **Cross-file score normalization, rank averaging over the test set, adaptive thresholds** | DACON rule 2.4 requires 파일 단위 독립 예측. Within-file segment pooling is explicitly allowed |
| **Heavy public-LB probing** | 3 submissions/day, and Private = Public — probing risks overfitting 1,200 samples. Use the decomposition probe ([competition/03](../competition/03-evaluation.md)) then stop |
| **Adversarial validation against the test set** | We have 3 dummy files. Run it against our own splits instead |
| **Test-set EDA to guide augmentation** | Same limitation — the dummy-file forensics ([data/07](../data/07-eda-plan.md)) is the substitute |

## Net changes to our existing plan

| Change | Where |
|---|---|
| Skew mixing-gain sampling toward **low component-SNR** rather than uniform `U(−15,+15)` | [data/02](../data/02-label-taxonomy.md), [data/06](../data/06-augmentation-spec.md) |
| Use **SigmoidConcatMixer**-style smooth transitions for sequential composition | [data/06](../data/06-augmentation-spec.md) |
| Add **pink noise**, **structured frequency/time dropout**, **varied resampler choice**, **RandomResizedCrop** | [data/06](../data/06-augmentation-spec.md) |
| **Condition augmentation strength on source cleanliness** | [data/06](../data/06-augmentation-spec.md) |
| Add **adversarial validation** and the **statistics-T quality filter** to the EDA pass | [data/07](../data/07-eda-plan.md) |
| Adopt **per-tier losses** and **two-stage clean→noisy training** for our label-confidence tiers | [data/08](../data/08-build-plan.md) |
| Make `FILE_FAKE_PROB` pooling a **tuned confidence-gated rule**, not a mean | [data/02](../data/02-label-taxonomy.md) |
| Add a **shadow split** for domain-shift stress testing, separate from the tuning split | [data/08](../data/08-build-plan.md#splits) |
| Generate **adversarial** (deliberately hard) fakes, not only typical ones | [data/05](../data/05-synthesis-plan.md) |
| Consider **real-class average-PSD whitening** as a front-end | new experiment |
| 🔴 **SED attention head + GeM freq pooling** for all five heads; learnable `p` settles the max-vs-mean pooling question | [06 §1](06-notebook-code.md) |
| 🔴 **Loss and inference blend `0.5·clip + 0.5·frame_max`** — matches our OR-over-segments label semantics | [06 §2](06-notebook-code.md) |
| 🔴 **Distill frozen XLS-R / BEATs into a small student with stop-gradient** — the runtime fix | [06 §3](06-notebook-code.md) |
| 🔴 **Cross-domain MixUp**: mix self-generated clean audio with real degraded audio (ASVspoof21 LA, MUSAN, Jamendo). Legal domain bridging where pseudo-labeling is not | [06 §6](06-notebook-code.md) |
| **Silero VAD** to verify Pool C is vocal-free, and as a presence-head prior | [06 §9](06-notebook-code.md) |
| Augmentation is **milder than we planned** in practice (gain ±6 dB, SNR 10–30 dB); MixUp does the heavy lifting | [06 §5](06-notebook-code.md) |
