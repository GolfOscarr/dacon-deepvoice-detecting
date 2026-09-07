# 07 — EDA Method Catalog (tiered)

**Why this matters more than usual**: the license gate ([01](01-rules-check.md)) makes our data
pool small and hard to expand. When you cannot buy score with more data, you buy it by
(a) knowing exactly what the test chain looks like, (b) making sure every bit of the pool is clean
and shortcut-free, and (c) spending the augmentation budget where EDA says it is needed.

## Tier legend (used in [05](05-synthesis-plan.md), [06](06-augmentation-spec.md), [07](07-eda-plan.md))

| Tier | Meaning |
|---|---|
| **S** | Blocking. Later work is invalid without it. Phase 0–1. |
| **A** | High expected value, evidence-backed. Phase 1–2. |
| **B** | Worth doing; moderate value or moderate cost. Phase 2–3. |
| **C** | Speculative or low-yield. Only with spare time. |
| **X** | Rejected — recorded so it isn't rediscovered. |

---

## Tier S

| # | Method | What it analyzes / why | Output artifact |
|---|---|---|---|
| **E-S1** | **Dummy-file forensics** on `TEST_0000–0002.wav`: `ffprobe` headers, encoder tags, LTAS rolloff, near-Nyquist resampler shelf, LUFS/peak/DC, dither, leading/trailing silence, inter-channel correlation | 🔴 The **only direct evidence of the organizers' signal chain**. Everything in [06 step 5](06-augmentation-spec.md) depends on reading it right. A rolloff near ~3.4 kHz would reveal how they represent 전화채널 | **`signal_chain.yaml`** — resampler, target sr, container set, channel policy, loudness policy, silence policy. Implemented literally by the normalizer |
| **E-S2** | **Shortcut audit**: logistic regression on **metadata-only** features (duration, loudness, silence ratio, effective bandwidth, channel count, container, source bitrate), no learned audio representation | Detects the failure mode that kills this kind of project — a confound separating real/fake with no acoustic content. ☆ `[LLM-Detect-AI 2024]`: off-distribution data caused "severe data drift" and a widening CV/LB gap | **`shortcut_audit.md`** — AUC per head and per cell. **Gate: AUC < 0.60**; above that, neutralize before training anything |
| **E-S3** | **Data memo**: file inventory, schema, label counts per cell, suspected leakage variables, risk list | ★ `[BirdCLEF playbook 2026]`: *"Do not advance to hyperparameter tuning until the data memo explains class imbalance, domain shift, and the first leakage hypothesis"* | One-page **`data_memo.md`**, versioned |
| **E-S4** | **Metadata role assignment** — map every field to exactly one of `feature` / `split key` / `leakage risk`; quantify missingness and cardinality | ★ `[BirdCLEF playbook 2026]`. Determines grouping keys for [08 splits](08-build-plan.md#splits) before any fold is built | **Frozen `metadata.parquet`** + role table, reused by every experiment |

---

## Tier A

| # | Method | What it analyzes / why | Output artifact |
|---|---|---|---|
| **E-A1** | 🔴 **16 kHz survivability probe** — train one small music-fake model on identical data at 48 kHz vs 16 kHz, measure the gap | Tests the **untested premise the whole music-head strategy rests on** ([survey G1](../survey/10-open-questions.md)). No published work evaluates AI-music detection under an 8 kHz Nyquist | **`16k_ablation.md`** — EER at both rates, per generator. Confirms the strategy or forces a rethink in week 1 |
| **E-A2** | **Adversarial validation** — classifier trained to separate train split from proxy-eval split | ★ `[G2Net 2021, 3rd]` used it to confirm train/test similarity (AUC 0.5). Detects corpus-identity leakage ([09 R2](09-risks-and-checks.md)) | AUC + ranked discriminating features. **Target ≈ 0.5** |
| **E-A3** | **Class-conditional LTAS** per cell and per generator family, **0–8 kHz only** | Finds which spectral signatures survive band-limiting. TISMIR's most-indicative features (mel-band skewness std, bark-band kurtosis) may not ([survey/02](../survey/02-sota-music.md)) | LTAS overlay plots + **`band_stats.csv`** (8 × 1 kHz band energies) |
| **E-A4** | **Per-band SNR / energy proxy** view, 8 × 1 kHz bands | ★ `[BirdCLEF playbook 2026]` standard view; ★ the hybrid-stems paper used exactly these as **model input features**, not just diagnostics ([survey/03](../survey/03-sota-singing-mixed.md)) | Per-file band-SNR feature table — reusable as model input |
| **E-A5** | 🔴 **Silero VAD sweep over Pool C** (`snakers4/silero-vad`; default threshold 0.5, community suggests 0.4) | Verifies "instrumental" really is vocal-free — Jamendo's tag is uploader-supplied and untrustworthy ([04](04-sources.md)). ★ `[BC25 separation notebook]` | **`pool_c_quarantine.csv`** — files with vocal energy, excluded or relabelled to a mixed cell |
| **E-A6** | **Statistics-T quality filter**: `T = std + var + rms + pwr`, drop below the 0.8 quantile | Removes low-information chunks. ☆ `[BirdCLEF 2024, 1st]` — treat as hypothesis; validate the threshold on our own data | `low_information.csv` + kept/dropped score histogram |
| **E-A7** | **Confound distribution audit** — duration, loudness, silence ratio, bandwidth, channels, container, **split by label and by cell** | The FoR dataset ships a `for-2sec` variant purely to kill duration-vs-label bias ([kaggle/04](../kaggle/04-datasets.md)) — a known trap in this exact dataset family | Per-feature distribution plots; anything with visible label separation goes to E-S2 for neutralization |
| **E-A8** | **T3 matched-pair mel gallery** — same utterance, real vs vocoder/codec resynthesis, side by side | The most legible figure available for the 2nd-stage report (결과 해석, 15 pts), and the clearest visual answer to "what does the artifact look like" | Figure set + short written characterization per generator family |
| **E-A9** | **Per-generator separability probe** — train a small model on one family, test on all others | Produces the **cross-generator confusion structure**: which families are trivially separable (low marginal value) and which are hard (spend budget there). ★ `[survey/02]` reports 46.4% EER cross-generator for music | **`generator_difficulty.csv`** — ranks families; drives Phase 3 data repair |

---

## Tier B

| # | Method | What it analyzes / why | Output artifact |
|---|---|---|---|
| **E-B1** | **Energy-threshold segmentation** — 0.05 s chunks, power in dB, −50 dB crossings ★ `[BC25 separation notebook]` | Dependency-free segment boundaries; feeds sequential composition and segment-level labels | Segment boundary table; component-duration statistics |
| **E-B2** | **Voice prosody features** — F0 jitter, shimmer, harmonic-to-noise ratio, formant trajectory smoothness | Synthetic speech is characteristically over-smoothed; low-frequency prosodic cues survive band-limiting where high-frequency fingerprints don't | Distribution plots per cell; candidate auxiliary features |
| **E-B3** | **Music structure features** — harmonic/percussive ratio, **beat-grid regularity**, long-range repetition | AI music is often metronomically exact; structural cues survive 16 kHz ([survey/02](../survey/02-sota-music.md), Segment Transformer) | Distribution plots; feasibility note for a structure branch |
| **E-B4** | **Inter-channel correlation** on stereo files | Detects mono-duplicated "stereo". ⚠️ Could be a genuine cue *or* a trap — must not correlate with label in our pool | `stereo_report.md`; feeds the channel-randomization rule in [06](06-augmentation-spec.md) |
| **E-B5** | **Hash + near-duplicate dedup** across pools and splits | ★ `[BirdCLEF playbook 2026]` automated guardrail — prevents the same source in train and val | `duplicates.csv`, enforced in fold construction |
| **E-B6** | **Component-SNR vs accuracy curve** (needs a model) | The hybrid-stems paper showed detection tracks stem energy (vocals 65–80% TPR vs accompaniment 97–98%). Shows where the model actually fails | Accuracy-vs-gain curve per head — also strong report material |

---

## Tier C

| # | Method | Note |
|---|---|---|
| **E-C1** | **PCEN view** | ★ `[BirdCLEF playbook 2026]` standard view "when background energy dominates" — relevant for quiet components under loud music |
| **E-C2** | Modulation spectrum / long-range temporal statistics | Plausible band-limiting survivor; unproven here |
| **E-C3** | Attention-map visualization | Report material; needs the SED head from [kaggle/06](../kaggle/06-notebook-code.md) |
| **E-C4** | **Error-analysis loop**: OOF errors → slice by cell / generator / gain → inspect audio + spectrogram → targeted fix | ★ `[BirdCLEF playbook 2026]`: *"Error analysis is where the next worthwhile experiment should come from."* Tier C only because it needs a trained model — **becomes Tier A in Phase 3** |

---

## Tier X — rejected

| Method | Why not |
|---|---|
| EDA on the real test set | We have 3 dummy files. **E-S1** is the substitute |
| Adversarial validation against DACON's test set | Same limitation — run it against our own splits (**E-A2**) |
| **Hand-engineering features from EDA findings** | SSL frontends beat handcrafted features throughout this literature — CtrSVDD baselines: raw waveform **13.75%** vs MFCC **26.67%** EER ([survey/03](../survey/03-sota-singing-mixed.md)). **Use EDA to decide what *data* to add, not what features to compute** |

---

## Standing rule

Re-run **E-S2** (shortcut audit) and **E-A2** (adversarial validation) after *every* corpus change.
They are cheap and they catch most of the failure modes in [09](09-risks-and-checks.md) before
those cost weeks.
