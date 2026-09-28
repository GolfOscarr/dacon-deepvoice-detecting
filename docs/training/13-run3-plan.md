# Run 3 plan — scale up and generalise

> **Superseded:** the rounds below were replaced by the data-first plan ([15](15-data-plan.md)) and the 1B DDP
> run (global batch 112); outcomes in [16](16-lessons-learned.md) and [17](17-final-runs.md).

*Rev 2, 2026-09-26 ~17:15 KST. Rev 1 (`3823472`, `fb1cf8d`) was reviewed adversarially and returned
REVISE (1 blocker, 7 major); every finding is folded in here and logged in §10. DDP is out of scope
(the owner implements it). Numbers are measured unless marked ESTIMATE. Sources:
docs/training/10 (run 1 diagnosis), 11 (run 2 design), 12 (run 2 results + sizing).*

## 0 · Goal and clock

- The leaderboard closes **2026-09-29 10:00 KST**. 3 submissions per day; Private = Public.
- **Where we are:**

  | | score | ADS | CPS |
  |---|---|---|---|
  | run 2 (T7) | **0.81144** | 0.79302 | 0.97730 |
  | leader | 0.89871 | 0.88756 | 0.99907 |

  **97.5 % of the gap is ADS**: 0.9 × (0.88756 − 0.79302) = 0.0851 of 0.0873.
- Runtime: 3 m 09 s of 60 min for today's model (19× headroom); **~6× for XLS-R-1B@48**
  (est. 10.1 min).
- **Goal:** close the ADS gap. It is a generalisation gap:
  - v3 VAL reads 0.956 and v4 VAL 0.944, while the LB reads 0.811;
  - PROBE over-reads LB gains by about 3×.

## 1 · Evidence

| # | finding | source |
|---|---|---|
| E1 | New real diversity + modern clones fixed run 1's diagnosed failures on v3 VAL (real LJ called fake .52 → .27, XTTS miss .31 → .09, musan called fake .49 → .25), but the LB rose only +0.020 | 12 §2, §6 |
| E2 | That fix came from **data**, not steps (T5). **Init from the previous run beats scratch** at equal steps (T4) | 12 §2 |
| E3 | Unseen VITS-style TTS regressed on v3 VAL: melo .05 → .19, mms .003 → .17 | 12 §2 |
| **E8** | **Run 2's own v4 VAL is where the largest held-out errors are.** Rates at the pooled threshold, `run_breakdown.py` on T1–T3: see the table below. Both **H1 (modern zero-shot cloners)** and **H2 (in-the-wild Korean real)** are live, not only VITS | review of rev 1; re-verified |
| E4 | Music EER did not move (PROBE .090 → .091). The **music and presence heads read BEATs only** (`c_run2.yaml` branches), so a bigger XLS-R cannot help music | 12 §3 |
| E5 | Averaging the two *sibling* run-2 models gives nothing (PROBE 0.9107 vs 0.9103). It says nothing about a **cross-architecture** ensemble | 12 §7 |
| E6 | XLS-R-1B@48 fits L4 inference: est. 10.1 min / 1200 files; 12.6 GiB (batch 8 × 60 s); weights +3.6 GB → zip ≈ 4–5 GB (limit 10). Encoder training ≈ 5.1×, full step ≈ 3.7× | 12 §7 |
| E7 | Codec and telephone chains are not where the errors are | 10 F4 |
| E9 | **The FILE mode max3 beats the learned FILE head on run 2 too**: file EER v4 .0727 → .0675, v3 .0565 → .0535, better on 4/4 folds each (≈ +0.002 score). Not implemented yet | review of rev 1 |

**E8 detail** (v4 VAL, rate at the pooled threshold):

| fold | held-out fake missed | real called fake |
|---|---|---|
| 1 | maskgct-en .41 (n 347), maskgct-ko .27 (n 471) | emilia-ko .18 (n 1031), emilia-en .11 |
| 2 | seedvc-en .22 (n 190), seedvc-ko .11, melo .21 (n 410) | emilia-ko .15 |
| 3 | mms .22, cosyvoice-en .15 | emilia-ko .18 |

**OPEN:** the per-head LB split. The run-1 diag A/B packages exist, but see §6 on re-packaging
them on T7.

## 2 · Label rule that constrains data work (blocker in rev 1)

DACON #417333 A1 (docs/competition/05 lines 64–90):
- post-processing that **does not generate a new voice/music component is REAL**, and that
  includes neural codec round-trips;
- S1 (`interim/proc`) already trains EnCodec / DAC / XCodec2 / Mimi round-trips of real speech
  as **REAL**.

Therefore:
- **No codec-decoder copy-synthesis labelled fake.** Rev 1's "C1" is withdrawn. It would teach
  the inverse of the target and contradict S1.
- **Mel → neural-vocoder copy-synthesis is ambiguous under A1.** The team's own reading
  (05 lines 82–86) treats "self vocoder/codec resynthesis" as REAL.
  - **Owner question Q-L1:** WaveFake is exactly vocoder copy-synthesis of LJ, and it is 181 h of
    our English *fake*. Keep it (it is a published deepfake benchmark), down-weight it further,
    or ask DACON?
  - Until answered, no *new* vocoder copy-synthesis is generated.
- New fake data must be **genuine generation**: TTS / cloning / voice conversion, new text.

## 3 · Levers (IDs are used everywhere below)

### Model
- **M1 XLS-R-1B** with a learnable **layer-weighted fusion** over hidden layers (M2). Depth:
  **@48 vs @24** is a real choice.
  - @24 costs 6.0 min inference and 4.0× encoder training; @48 costs 10.1 min and 5.1×.
  - M2's own mechanism says the top layers carry speaker/content, which feeds the E1 shortcut.
  - **Default @24 + fusion** unless the owner prefers @48. Fusion can down-weight the top.
- **M2 layer-weighted fusion.** Softmax weights over layers (or per-layer attentive pooling).
  Low and mid SSL layers keep the acoustic detail where generation artifacts live. Near-zero cost;
  inspect the learned weights.
- **M3 init.** XLS-R-1B has width 1280 vs 1024, so **the voice and file heads (0.63 of the score
  weight) cannot load strictly** (`models/model.py:76`, `init_from` strict).
  - Only BEATs + its LoRA and the BEATs-only heads (music, v_pres, m_pres) transfer from T7.
  - Needs a **non-strict partial init** (or a 1280 → 1024 projection) and a cold-start budget for
    the voice/file heads and the new LoRA.
- **M4 fallback: XLS-R-300M@24.** Loads the first 12 layers + all heads from T7 strictly-compatible
  (same width); ≈ 1.2× step. It is **actually trained** in R2 (§7) so it exists if 1B slips.

### Objective
- **O1 one-class margin loss** (OC-Softmax) on the voice/file heads. Binary CE bounds the
  *training* fakes; OC makes **real** the compact class, so an unseen fake only has to be
  "not real".
  - Fits E3/E8: the missed families sit at p̄ ≈ .5.
  - Risk: in-the-wild real (Emilia) is diverse, so a tight real cluster can raise false alarms
    (E8's H2).
  - Needs new head parameters → non-strict init.
- **O2 paired real ↔ own-fake contrastive.**
  - Clone pairs share the **speaker** (not text or room).
  - Needs a pair-aware sampler that puts a real clip and its clone in one batch — does not exist.
    **Deferred.**
- **O3 corpus-adversarial head** (gradient reversal on source corpus). Needs corpus labels through
  collate — do not exist. **Deferred.**
- **O4 mixture of LoRA experts** (MoLE). Risk of memorising training families, routing collapse,
  no tuning time. **Deferred** (owner decision).
- **O5 FILE mode max3** (E9): max(learned, v·vp, m·mp). **Implement and ship** — measured +≈ 0.002
  on 8/8 fold readings.

### Data (all genuine generation or real; §2)
- **D1 more modern zero-shot cloners, ko + en** (E8's H1): new families not yet in the corpus
  (e.g. F5-TTS, Spark-TTS, IndexTTS, OpenVoice-v2 / GPT-SoVITS-style VC), plus **more hours of the
  weakest-generalising ones** (maskgct, seedvc).
  - Prompts from Emilia speakers, dataset transcripts (owner rule).
  - Licence check per model.
- **D2 VITS-style TTS** (E3): run 3a's answer (ko-synth ×2/×3, due ≈ 19:00). If insufficient, add
  MMS-TTS / Piper / Melo EN + KO.
- **D3 more in-the-wild Korean real** (E8's H2: Emilia-ko false alarms .15–.18).
  - More Emilia-ko / YODAS-ko hours and speakers.
  - Channel-diverse Korean real (phone-band, broadcast) where licensable.
- **D4 music** (E4): only if the per-head split says so. BEATs-side work.
- **D5 draw symmetry** kept (11 §1.1): repeats, processed share and language balanced per label;
  the audit gates run on every config change.

### Training recipe
- **R-init:** every module that fits loads from T7 (E2); new parts are cold (M3).
- **R-len:** stated in the **global** batch. ESTIMATE for 1B: global batch 128 (8 GPUs × 16) ×
  4–6k steps ≈ 0.5–0.8 M samples.
  - Check the memory: 1B@48 takes 40 GiB at batch 4 in the bench, so per-GPU 16 needs activation
    checkpointing or @24.
  - The owner sets this with DDP.
- **R-SAM:** sharpness-aware minimisation, ≈ 2× step. Only if time allows.

### Inference
- **I1 per-file TTA** (K crops/shifts averaged within one file; rule 2.4 compliant).
  - **Cap K so the *timed* total ≤ 30 min**: with 1B@48 (≈ 10 min), K ≤ 2–3.
  - Verify that batch padding does not make a file's output depend on its batch-mates.
- **I2 cross-architecture ensemble** (T7 + the 1B model, ≈ 13 min). E5 does not rule it out;
  measure it.

## 4 · Hypotheses for the LB gap (what each experiment must separate)

| # | hypothesis | on VAL today | what tests it |
|---|---|---|---|
| H1 | test fakes = generator families we have not seen | **yes**: maskgct .41/.27, seedvc .22, melo .21, mms .22 (E8) | D1/D2, O1, M1/M2: held-out-family EER |
| H2 | test real = recording domains we under-cover → false alarms | **yes**: Emilia-ko .15–.18 false alarm (E8) | D3, O1 (watch it), M1: unseen-real EER |
| H3 | music side weak on the test | unknown (VAL music fine) | per-head LB split |
| H4 | file head mis-combines voice + music | partly: max3 beats learned (E9) | O5; per-head split |
| H5 | presence | ≤ 0.0022 of score | per-head split (CPS) |

## 5 · Evaluation and decision rule (rev 1's rule was not meaningful)

**Metrics** (every ablation task; both v3 VAL = run 1's specs and v4 VAL):
- **S1 held-out-family EER**: that family's fakes vs *all* reals of the fold. Threshold-free, so it
  does not trade off against S2 through a shared threshold.
- **S2 unseen-real EER**: that corpus's reals (Emilia-ko, musan, CV-ko) vs all fakes.
- **S3 pooled score** (guard).

**Decision:** keep a component only if it improves S1 or S2 by **≥ 2 binomial SE on both folds**
without S3 dropping > 0.005.
- SE is computed from the slice n; with n ≈ 300–1000 that is ≈ 0.02–0.04 absolute.
- **Seed noise**: measured only if a 7th GPU is free in R1 (§7); otherwise the readout says it is unmeasured.
- Effective n is below nominal (specs reuse speakers), so this rule is a floor, not a guarantee.

**PROBE** is not used for ablations. HANDOFF §4 forbids it outside the final check, and its voice
side is Chinese.

**The LB is the final judge.** Ablations cannot reach it (they are fold models), so R2 also trains
**all-data variants that are submitted on Sunday** (§7).

## 6 · Per-head LB split — re-package on T7

- Diag A/B were packaged on run 1. Run 2's voice head changed a lot (PROBE voice .276 → .061), so
  a run-1 split mis-weights today's decisions.
- **Re-package diag A/B on `run2-T7`** (the same `model/diag_constant.json` mechanism, `2cb5611`)
  and submit tonight or Sunday morning, before the main run's content is fixed.
- Limitation: EER is symmetric, so a per-head EER cannot separate misses from false alarms (H1
  vs H2 inside a head). VAL (E8) is the evidence for that split.
- It uses 2 of the day's 3 submissions; the owner schedules it.

## 7 · Rounds and GPUs

Owner's DDP/1B development and smoke tests need GPUs: **2 GPUs are reserved for the owner** from
19:30 until the main run. The ablation rounds use 6.

| round | when (KST, ESTIMATE) | GPUs | tasks | question |
|---|---|---|---|---|
| R0 | running → ≈ 19:00 | 4 | run 3a: ko-synth ×2 / ×3, folds 2, 3 | D2: does up-weighting recover melo/mms? |
| R1 | Sat ≈ 19:30 → Sun ≈ 00:30 | 6 | folds 1 + 2 × {(a) baseline, (b) + O1 one-class, (c) LJ-voice fakes ×0 (Q-L1; `configs/processing_run3_nowf.yaml`)}; proxy = 300M@12 from run-2 fold models, 10k steps. Seed replicate of (a) fold 1 **only if a 7th GPU is free**; otherwise the 2-SE rule stands without a measured seed floor (stated in the readout) | does O1 help S1 without hurting S2? does WaveFake help or hurt? |
| gen | Sat ≈ 19:00 → Sun ≈ 06:00 | shares R1's 6 when idle, else CPU | D1 new cloner families + more maskgct/seedvc hours; D3 more Emilia-ko | new data for v5 |
| v5 | Sun ≈ 06:00 → 07:30 | CPU | strategy-v5 = v4 + D1 + D2 fix + D3; pinned folds; audits (build_strategy_v4.sh pattern) | — |
| R2 | Sun ≈ 07:30 → 12:00 | 6 | **all-data** T7 fine-tunes: (a) v5, (b) v5 + O1 (if R1 kept it); **M4 300M@24 + M2 fusion** on v5 (all-data, the fallback); + fold 1 of (a) for a VAL reading | LB tests on Sunday; the trained fallback |
| main | Sun ≈ 12:00 → Mon early | 8 (DDP) | **1B + M2 + v5 (+ O1 if kept)**, all-data | the submission |
| eval | alongside | CPU / idle GPU | O5 max3 on every candidate; I1 TTA K ∈ {1, 2, 3}; I2 T7+1B ensemble | inference-only gains |

- **Sunday submissions:** best R2 all-data model (with max3), plus diag A/B on T7 if not already
  done.
- **Monday:** 1B main (+ I1/I2 if measured better); final picks Tuesday morning.
- **Proxy → 1B transfer** is plausible for *data* effects, much less for *objective* effects (a
  warm 300M fine-tune vs a cold 1B LoRA). A 1B fold-1 check (± component) is what would falsify
  it; it needs 2 GPUs during the main run, which DDP takes. **This is not measured, and we say
  so.**

## 8 · Other methods considered (not planned)

| method | why not now |
|---|---|
| AASIST-style graph back-end | our SED heads + presence masking would need re-validation |
| label smoothing / mixup | weak evidence for unseen generators |
| distillation from a larger teacher | the teacher has to be trained first |
| pseudo-labelling the test | the test is hidden |
| test-set statistics / rank normalisation | forbidden (rule 2.4) |
| sub-band models | revisit only if the split shows the file head is the problem |
| longer training on the same data | flat after ~36k (10 F7) |
| same-init weight soup | only valid for all-data fine-tunes of the shipped architecture; the R2 all-data variants make it testable, otherwise dropped |
| external real-world eval set (e.g. In-the-Wild) | useful to calibrate VAL against the LB; eval-only, licence check first — *owner call* |

## 9 · Owner decisions (answered 2026-09-26 ~17:30 KST; primary goal: the score)

| # | question | decision |
|---|---|---|
| Q-L1 | WaveFake vs DACON A1 | **Keep as is (×0.25), and ablate it**: R1 adds a "WaveFake ×0" variant. WaveFake sits wholly in fold 0's VAL, so folds 1–2 train on it and can show whether it hurts |
| 2 | 1B depth | **@24 + M2 layer fusion** |
| 3 | O1 one-class | **Yes, tested in R1** (kept only if it passes §5) |
| 4 | O4 MoLE | **Deferred** |
| 5 | Diag A/B on T7 | **Tonight** (2 of today's 3). Packaged as `run2-T7-diagA/B.zip` |
| 6 | External eval-only set | **Skip** |

## 9b · Q-L1 ablation config — measured

- **Domain weights alone cannot zero WaveFake.** Ten MLAAD TTS families use the LJSpeech voice and
  share WaveFake's speaker bucket (`ljspeech_LJ`). An MLAAD-LJ anchor tiles from that bucket, which
  is ~all WaveFake (101k files).
- `processing_run3_nowf.yaml` therefore zeroes `wavefake|`, `proc-*|wavefake|` **and**
  `mlaad|tts_models_en_ljspeech` (≈ 290 files): an "**all LJ-voice fakes removed**" variant.
- Measured on fold 1, 4,000 specs:

  | config | WaveFake share of fake voice tiles |
  |---|---|
  | run 2 draw | 1734 of ~20.2k ≈ 8.6 % |
  | `wavefake|` ×0 alone | 756 |
  | + MLAAD-LJ ×0 | **151 of 20,236 = 0.75 %** (residual: bucket fallbacks, processed MLAAD-LJ) |

- Audits at n = 40,000: **audit ok on folds 1 and 2** (I1c passes; processed .200/.265 and
  .184/.239; ko/en .556/.444 real).
- Side finding: the tile fallback (`sampler._tiles`, bucket exhausted → uniform over the pool,
  weights ignored) fires 14–25 times per 40k voice tiles. Rare, but it ignores `domain_weights`.

## 10 · Review log

Rev 1 adversarial review (critic agent, read-only + CPU checks; `run_breakdown.py` on v4 VAL):

| finding | resolution |
|---|---|
| **BLOCKER** copy-synthesis labelled fake contradicts DACON A1 and S1 | withdrawn (§2); new fake data = genuine generation; WaveFake raised as Q-L1 |
| MAJOR largest held-out failures (v4 VAL) ignored | E8 added; H1/H2 re-weighted; D1/D3 added |
| MAJOR decision rule not meaningful | threshold-free per-slice EER, ≥ 2 SE both folds, seed replicate (§5) |
| MAJOR no ablation reaches the LB | R2 is all-data variants submitted Sunday (§7) |
| MAJOR GPU/time conflicts; "small loss additions" false | 2 GPUs reserved for the owner; O2/O3 deferred (need a sampler / labels); one v5 time (§7) |
| MAJOR heads cannot load from T7 at width 1280 | M3: partial init + cold-start budget |
| MAJOR D3 soup unshippable | dropped unless it is all-data variants of the shipped architecture (§8) |
| MAJOR diag A/B on the wrong model; over-claimed | re-package on T7; limitation stated (§6) |
| minor: gap share 97.5 %; headroom ~6× for 1B; ID collisions; @24 vs @48; zip size; step budget in global batch; max3 decidable now; PROBE use; pair claim; E5 overstated; §4/§9 consistency; fold 2 also holds openvoice | all fixed in place (E5, E6, E9, M1, M3, O2, O5, R-len, I1, §5) |

Literature named in rev 1 (Tak 2022 SSL+AASIST+RawBoost; Zhang/Jiang/Duan 2021 OC-Softmax;
Wang & Yamagishi 2023 vocoded training data; Zhang 2024 SLS) was judged correct by the reviewer.
The SAM-for-anti-spoofing attribution is unsure. Rev 2 names methods, not citations.
