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
