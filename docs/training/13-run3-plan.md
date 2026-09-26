# Run 3 plan — scale up and generalise (BRIEF, to be expanded)

*Draft 2026-09-26 16:30 KST. DDP is out of scope here; the owner implements it. Numbers are
measured unless marked ESTIMATE. Sources: docs/training/10 (run 1 diagnosis), 11 (run 2 design),
12 (run 2 results and sizing).*

## 0 · Goal and clock

- The leaderboard closes **2026-09-29 10:00 KST**. 3 submissions per day; Private = Public.
- **Where we are:**

  | | score | ADS | CPS |
  |---|---|---|---|
  | run 2 (T7) | **0.81144** | 0.793 | 0.977 |
  | leader | 0.89871 | 0.888 | 0.999 |

  96 % of the gap is ADS, i.e. the detection EERs.
- Server runtime is 3 m 09 s of the 60 min allowed, so we have **≈ 19× compute headroom** at inference.
- **Goal:** close the ADS gap, which is a **generalisation** gap:
  - VAL reads ≈ 0.95–0.96 while the LB reads 0.81;
  - PROBE over-reads LB gains by about 3× (+0.058 on PROBE vs +0.020 on the LB, run 1 → run 2).

## 1 · What the evidence says (drives every choice below)

| # | finding | source |
|---|---|---|
| E1 | New real diversity (Emilia) plus modern clones fixed the diagnosed failures: real LJ called fake .52 → .27, XTTS miss .31 → .09, musan called fake .49 → .25. The LB rose only +0.020 | 12 §2, §6 |
| E2 | The data fix is from **data**, not steps (control T5). **Init from the previous run beats scratch** at equal steps (T4) | 12 §2 |
| E3 | New regression: **unseen VITS-style TTS** (melo, mms) missed .05 → .19 and .003 → .17 | 12 §2 |
| E4 | Music EER did not move (PROBE .090 → .091); nothing in run 2 touched music | 12 §3 |
| E5 | Ensembling run-2 models gives nothing (errors are correlated) → the gain has to come from a **different or bigger representation**, not averaging | 12 §7 |
| E6 | XLS-R-1B@48 fits L4 inference: est. 10 min / 1,200 files, 12.6 GiB (batch 8 × 60 s), zip ≈ 5.6 GB. Encoder training ≈ 5× today's | 12 §7 |
| E7 | Codec and telephone chains are **not** where the errors are (VAL error flat across them) | 10 F4 |
| **OPEN** | Per-head LB split (file / voice / music EER; presence AUCs) from diag A/B — **not yet submitted or reported** | 12 §1 |

## 2 · Levers, ranked

### A. Model scaling (owner decision: XLS-R-1B)
1. **XLS-R-1B, all 48 layers**, LoRA as today. BEATs and the heads start from run 2 T7; the encoder
   LoRA starts fresh.
2. **Layer-weighted fusion over all hidden layers** (learnable softmax weights, or attentive fusion)
   instead of a truncated stack. Spoofing cues sit in the low and mid layers. Near-zero cost.
3. Fallback if 1B is late or unstable: **XLS-R-300M, all 24 layers** (~1.2× step; the first 12 layers
   start from run 2).

### B. Objective
4. **One-class margin loss** (OC-Softmax style) on the voice and file heads. Real is the compact
   class, so unseen fakes fall outside it.
5. **Paired contrastive loss:** real ↔ its own fake, from the clone pairs and copy-synthesis (C8).
6. **Corpus-adversarial head** (gradient reversal on source corpus). It removes corpus and speaker
   shortcuts (E1's mechanism).
7. *Stretch:* mixture of LoRA experts (4 experts, rank 8, utterance router, top-2, load-balancing
   loss) in the top layers; or a small gated fusion across BEATs and the XLS-R layer groups.

### C. Data mixture and new data
8. **Vocoder / codec copy-synthesis of our real speech** (Emilia-ko/en, Zeroth, LibriTTS-R) through
   4–6 neural vocoders and codec decoders, labelled fake. Speaker-, text- and room-matched pairs.
   This is the strongest data lever for unseen generators (E3) and shortcuts (E1). GPU-hours, not
   days.
9. **VITS-style coverage** (E3): answer from run 3a (ko-synth ×2/×3, due ≈ 19:00). If the up-weight
   does not recover melo/mms, add VITS/MMS-TTS/Piper-style EN + KO synthesis.
10. **Music** (E4, pending diag): more fake-music families; check the music head on real mixtures.
    Owner call once the diag split is known.
11. Keep the run-2 draw-symmetry rules: repetition, processed share and language balanced per
    label; audit gates on every config change.

### D. Training recipe
12. Init policy: every module that exists in run 2 T7 starts from T7 (E2).
13. Length: 1B needs more steps than a fine-tune. ESTIMATE ≥ 30k steps × batch 16 equivalents
    (DDP). Checkpoint every pass for learning curves.
14. *Optional:* SAM (≈ 2× step) — only if time allows.

### E. Evaluation and selection
15. Judge on the diagnosed slices (E1/E3), not pooled VAL (mistake 16). Use run 1's exact v3 VAL
    specs plus v4 VAL.
16. PROBE is for ranking only; it over-reads LB gains by about 3×. The **LB is the final judge**:
    reserve submissions for it.
17. Diag A/B (already packaged) settle which head to invest in.

### F. Inference
18. **Per-file test-time augmentation** (a few crops or time shifts averaged within a file). Legal
    under rule 2.4. It uses the runtime headroom; measure the gain on VAL first.
19. Real `script.py` timing on the packaged 1B model before any submission (the E6 estimate is a
    same-GPU ratio).

## 3 · Proposed sequence (ESTIMATE; refine after DDP lands)

| when (KST) | step |
|---|---|
| now → Sun AM | owner: DDP + 1B + layer fusion. Lead: copy-synthesis data (C8), run-3a readout (C9), objective code behind flags (B4–B6) |
| Sun AM | strategy-v5 build (v4 + copy-synthesis + VITS fix) with pinned folds and audits; short smoke of 1B-DDP |
| Sun midday → Sun night | **run 3 main**: 1B all-data (DDP). A small fold check in parallel if GPUs allow |
| Mon AM | package, time on the server mirror, PROBE → **submit** |
| Mon → Tue 10:00 | one more iteration (fine-tune / TTA / music), final 1–2 submissions |

## 4 · Decisions for the owner

1. **Objective:** one-class and paired losses in run 3 (B4, B5), or data-only first?
2. **MoLE** (B7): run 3 or later?
3. **Music investment** (C10): waits on the diag A/B scores.
4. **Fold runs for 1B:** some validation runs, or all-data only (faster to a submission)?

## 5 · Sections to expand (owner)
- [ ] A: exact architecture (fusion form, LoRA targets and rank for 1B, heads)
- [ ] B: loss definitions, weights, which heads
- [ ] C: copy-synthesis vocoder list, hours, label and fold rules; v5 draw weights
- [ ] D: step budget, LR, schedule, batch under DDP
- [ ] E: acceptance criteria per slice; submission schedule
- [ ] F: TTA scheme and its runtime budget

---

# Details (expanded 2026-09-26 evening)

## 6 · Why the leaderboard moved so little — hypotheses to test

Run 1 → run 2 moved the LB +0.020 but PROBE +0.058 and v3 VAL +0.006. The LB-to-VAL gap is the
real problem, so every experiment should separate these:

| # | hypothesis | predicts | discriminating evidence |
|---|---|---|---|
| H1 | **Test fakes come from generator families we never trained on** (commercial TTS, newer cloners, VITS-style) | voice and file EER high on the LB; held-out-family misses high on VAL | diag B voice EER; held-out family miss per fold (12 §2: melo .19, mms .17) |
| H2 | **Test real audio is a recording domain we under-cover** (phone, broadcast, lossy uploads, spontaneous speech) → false alarms | real-side false alarms dominate; pooled EER inflated by real | CV-ko / Emilia false alarms on VAL; diag B voice EER vs diag A file EER |
| H3 | **Music side is weak on the test** (fake-music generators, real mixtures) | diag A plus run-1 ADS → music EER high | diag A/B → music EER by subtraction |
| H4 | **The file head mis-combines voice and music** on real mixtures | file EER ≫ max(voice, music) contribution | diag A file EER vs B voice EER |
| H5 | **Presence** is harder on the test (CPS .977 vs leader .999) | worth ≤ 0.0022 of score; low priority | CPS from both diag submissions (they isolate each AUC) |

**Diag A/B are the cheapest experiment we have:** 2 submissions settle H3–H5 and weight H1 vs H2.
They are packaged; the owner submits.

## 7 · Methods — why each should help, cost, risk, how we measure it

Mechanisms are stated against our own evidence. Literature is named only where the method is a
well-known published one. **Citations are from memory: verify before quoting outside the team.**

| method | why it should generalise (mechanism) | evidence it fits *our* failure | cost | risk | measured by |
|---|---|---|---|---|---|
| **A1 XLS-R-1B** (all 48 layers) | A larger SSL model pre-trained on more speech has a richer "what natural speech sounds like" prior. Artifacts of unseen generators fall outside it. The SSL front-end + light back-end recipe (e.g. Tak et al. 2022, wav2vec 2.0 + AASIST + RawBoost) is the standard strong baseline for unseen attacks | H1/H2 are generalisation failures; E5 says averaging our current representation adds nothing | inference ~3× (est. 10 min / 1200), encoder training ~5× | new encoder LoRA starts cold; more steps needed; overfits the training generators if data stays narrow | fold VAL diagnosed slices; PROBE; LB |
| **A2 layer-weighted fusion** (softmax over all hidden layers, or per-layer attentive pooling) | Low and mid SSL layers keep acoustic/phase detail where vocoder and codec artifacts live; top layers drift to phonetic and speaker content, which feeds the speaker shortcut (E1). Letting the model choose layers keeps the artifact layers. Layer-selection classifiers on XLS-R (e.g. "SLS", Zhang et al. 2024) report gains | E1: the pooled number was carried by speaker/corpus identity | ~0 params, ~0 time | little; can collapse to one layer (inspect the weights) | same + learned layer weights |
| **A3 XLS-R-300M@24** (fallback) | same as A1, smaller | — | ~1.2× step | smaller gain | same |
| **B1 one-class margin loss** (OC-Softmax, Zhang et al. 2021) | Binary CE learns a boundary around the *training fakes*; OC pulls **real** into a tight region and pushes anything else out by a margin, so an unseen fake only has to be "not real" | H1 (unseen families): run 2 misses melo/mms with p̄ ≈ 0.5, i.e. they sit between the classes | ~0 | if real is too diverse (Emilia in-the-wild), the real cluster is loose → false alarms rise (H2). Tune the margin | held-out-family miss vs real false alarm |
| **B2 paired real ↔ own-fake contrastive** | The same speaker, text and room on both sides leave the generator as the *only* difference; the loss forces features that encode the artifact, not identity | E1 (T3 gap .083 still), H1 | small; needs pairs (clone pairs exist; C1 gives many more) | little | T3 gap; held-out-family miss |
| **B3 corpus-adversarial head** (gradient reversal on source corpus / recording chain) | Removes "which corpus is this" from the features; the LJ/musan false alarms were corpus identity | E1 mechanism | small | can also remove useful channel cues; keep its weight small | real false alarm per unseen corpus |
| **B4 mixture of LoRA experts** (MoLE; top-k LoRA experts with a router) | Generator families differ (vocoder / codec-LM / VC / VITS); experts can specialise while sharing the backbone | H1 | moderate code; more params | memorises training families; routing collapse; hard to tune in 2 days | same slices; expert utilisation |
| **C1 vocoder/codec copy-synthesis of our real speech** (fake labels; Wang & Yamagishi 2023 show vocoded data is an efficient spoofing training set) | Every modern TTS/cloner ends in a vocoder or codec decoder; copy-synthesis teaches those artifacts on **our** speakers and rooms, so real/fake differ *only* in the artifact (kills E1-type shortcuts and covers H1) | E1, E3, H1 | GPU-hours; the S1 code path already runs codecs (`scripts/proc`) | a label shortcut if the drawn shares are asymmetric → audits (11 §1.1) | held-out family miss; T3 gap; audit I1c |
| **C2 VITS-style coverage** (ko-synth ×2/×3 draw weight, or new MMS-TTS/Piper/Melo EN+KO) | E3 regression | E3 | draw weight: 0; synthesis: hours | — | run 3a (due ≈ 19:00): melo / mms miss |
| **C3 in-the-wild / channel real** (more Emilia, phone-band real, broadcast) | H2 | CV-ko false alarm still .02–.11 | hours | — | real false alarm |
| **C4 music** (more fake-music families, real music mixed at realistic ratios) | H3; music head reads BEATs only (`c_run2.yaml` branches) | E4 flat | hours–day | — | music EER, cell 4/6 |
| **D1 init from T7** where modules match | E2 | E2 | 0 | carries run-2 biases | T4-style control |
| **D2 SAM** (sharpness-aware minimisation; reported to help cross-dataset anti-spoofing) | Flatter minima transfer better under domain shift | H1/H2 | ~2× step | time | same slices |
| **D3 weight soup of fine-tunes from one init** | Unlike run 1's soup (54k steps apart), short fine-tunes from the same T7 init stay in one basin; averaging them is a free ensemble at no inference cost | E5 says prediction ensembles of *different* runs add nothing; a same-basin soup is a different claim — test it | 0 train (reuses ablation runs), 0 inference | may equal the best single model | PROBE, VAL |
| **F1 per-file TTA** (average over K crops/shifts within a file) | Reduces variance on long files; uses the runtime headroom; legal under rule 2.4 (per file) | runtime 3 m / 60 m | K× inference | little | VAL/PROBE with vs without |
| **F2 file-head mode max3** (11 §2) | Label-consistent OR of the present heads | run 1: 0.060 vs 0.064 file EER | 0 | small | fold VAL |

## 8 · Other methods considered (not planned unless evidence says so)

- **AASIST-style graph back-end** on the SSL features. A strong published back-end, but our heads
  are SED heads with presence masking; swapping them costs a re-validation we cannot afford. *Later.*
- **Label smoothing / mixup between real and fake.** Mild calibration gains; weak evidence for
  unseen generators.
- **Knowledge distillation from a larger teacher** (e.g. XLS-R-2B). Needs the teacher trained
  first; no time.
- **Pseudo-labelling the test.** Impossible: the test is hidden (code submission).
- **Test-set statistics / rank normalisation.** Forbidden (rule 2.4, per file only).
- **Frequency-band / sub-band models** (high-band artifacts). Covered in part by BEATs and the
  multi-layer fusion; revisit only if diag shows the file head is the problem.
- **Longer training at the same data.** Measured flat after ~36k (10 F7); no.

## 9 · Ablation plan

**Principle.** Ablate on the **cheap proxy** (today's architecture, XLS-R-300M@12, init from run 2's
fold models, 10k steps, ~3.5–4 h per round on 8 GPUs). Transfer the winners into the **1B main
run**. The 1B run itself gets one component check (A2) through its fold-validation tasks.

**Folds used for ablations: fold 1 and fold 2**, each task paired over the two.
- Fold 1 holds out XTTS (unseen cloner) and maskgct, and has musan and CV-ko real (H2).
- Fold 2 holds out melo (VITS-style, E3) and seedvc (voice conversion).
- Fold 0's LJ test stays in the final check.

**Metrics for every task** (on run 1's exact v3 VAL specs *and* v4 VAL, via `post_run2.sbatch`-
style scoring + `run_breakdown.py`):
- M1 held-out-family miss rate (xtts, melo, maskgct, seedvc) at the pooled threshold;
- M2 real false alarm on unseen real corpora (musan, CV-ko, Emilia VAL);
- M3 T3 matched-pair gap (fold 0 only, final check);
- M4 pooled score (guard: must not drop > 0.005).

**Decision rule:** a component is kept if it improves **M1 or M2 on both folds** without hurting
the other by more than its gain, and M4 holds.

| round | when (KST, ESTIMATE) | tasks (8 GPUs = 4 variants × folds 1, 2) | question |
|---|---|---|---|
| R0 | running → 19:00 | run 3a: ko-synth ×2 / ×3, folds 2, 3 | C2: does up-weighting recover melo/mms? |
| R1 | Sat 19:30 → Sun 00:00 | (a) baseline: continue on v4; (b) + B1 one-class; (c) + B2 paired contrastive; (d) + B3 corpus-adversarial | which objective terms help unseen families / unseen real |
| R2 | Sun 00:30 → 05:00 | (a) v5 data (+ C1 copy-synthesis, + C2 fix); (b) v5 + R1 winner; (c) v5 + winner + D2 SAM *or* B4 MoLE; (d) A3 300M@24 + fusion | does copy-synthesis help; does the stack compose |
| — | Sun 05:00 → | eval-only: D3 soups of the R1/R2 fine-tunes; F1 TTA K ∈ {1, 3, 5}; F2 max3 | free gains at inference |
| main | Sun midday → Mon early | **1B + A2 fusion + winners + v5**, all-data (DDP), plus a small fold check if GPUs allow | the submission |

**Budget check.**
- R1 and R2 need the objective code (B1–B3) behind flags by ≈ 19:30 and the v5 data by ≈ 00:30.
  - B1–B3: the lead writes them tonight; they are small loss additions.
  - C1: generated tonight on GPUs 4–7 *until R1 starts*, then on CPU where possible.
- The data side: v5 = v4 + C1 + C2. It is built with the pinned folds (the new families are placed
  like run 2's), and the audits must pass.
- If the objective code is not ready by 19:30, R1 runs the data questions first and the rounds
  swap order.

## 10 · Double-check log
(filled after review)
