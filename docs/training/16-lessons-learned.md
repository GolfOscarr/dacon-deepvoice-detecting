# 16 · What we have learned (runs 1–3, strategy-v5, the 1B model)

*2026-09-27 ~10:00 KST, lead session. Every number below was measured in this project; the source
(a submission, a job, a file) is named. **UNVERIFIED** marks the few inferences that are not
measurements. Plan documents: 11 (run 2), 13 (run 3), 14 (DDP), 15 (data plan).*

## 1 · The one-paragraph summary

- **The test set is harder than every local measure.** Each head sits at about **0.2 EER on the
  real test**, against 0.02–0.15 on our validation (§2).
- **So the gap is generalization, across the board, not one broken head.**
- **Data has moved the leaderboard; objectives have not.** Run 2's gain came from data, while the
  one-class loss and draw re-weighting measured flat or worse (§3–4).
- **Many of the traps were shortcut cues in the data, not model problems** (§5).
- The response is:
  - **strategy-v5**: 12 new generator families, singing voice, in-the-wild Korean, phone
    channels, and those cues removed;
  - **XLS-R-1B** on DDP;
  - the 1B run trains on v5 now, and the leaderboard judges v5 tonight.

## 2 · What the leaderboard tells us

| submission | LB | ADS | CPS | note |
|---|---|---|---|---|
| run 1 (`first-v3-seed0`) | 0.79188 | 0.77288 | 0.96287 | |
| run 2 T7 (`run2-T7`) | **0.81144** | 0.79302 | 0.97730 | best so far |
| T7 diagA (file head + music presence live, others 0.5) | 0.65797 | 0.64914 | 0.73744 | per-head split |
| leader | 0.89871 | 0.88756 | 0.99907 | 97.5 % of our gap is ADS |

**Per-head split of T7 on the real test** (diagA + T7, by the formulas in HANDOFF_RUN3 §4):

| quantity | test | our VAL (v4, run 2 fold models) | reading |
|---|---|---|---|
| file EER | **0.202** | ~0.03–0.08 | 3–6× worse on test; costs 0.10 of ADS by itself |
| voice + music EER | 0.2·(1−v) + 0.3·(1−m) = 0.394 → **≈ 0.21 each if equal** | 0.05–0.15 / 0.003–0.04 | not separable without diagB (range: music 0.02–0.35) |
| music-presence AUC | **0.975** | 0.99999 | presence also generalizes worse |
| voice-presence AUC | **0.980** | 0.99999 | likely part singing (§5.2), UNVERIFIED |

**Lessons:**
- **L1. Local numbers over-read the leaderboard.** v3 VAL 0.956, v4 VAL 0.944 and PROBE 0.910
  sit against an LB of 0.811. PROBE over-read the run-1 → run-2 LB gain by about 3×. Rank with
  local numbers; decide with per-slice VAL plus the LB.
- **L2. The file head is the single biggest lever.** At 0.5 weight and 0.20 EER it loses as much
  ADS as voice and music together. Tonight's max3 A/B on the v5 model tests one cheap fix.
- **L3. diagB was not worth a slot on Sunday.** It separates voice from music, but nothing before
  Tuesday would change on the answer.

## 3 · Data: what moved the needle and what did not

| experiment | result | decision |
|---|---|---|
| run 2 vs run 1 on run 1's exact v3 VAL | 0.950 → 0.956. Real LJ called fake .52 → .27; Korean XTTS missed .31 → .09; real MUSAN called fake .49 → .25. LB 0.792 → 0.811 | **data is the lever**. Control T5 (same data change on v3) says the gain came from the data; T4 says init from run 1 beats scratch |
| run 2 regression | melo missed .05 → .19, mms .003 → .17 (older VITS-style Korean TTS) | investigated in run 3a |
| run 3a: ko-synth ×2 / ×3 in the draw | fold 2 melo EER .088 (run 2) → .086 (×2) / .105 (×3); fold 3 mms .071 → .076 / .067; scores flat | **re-weighting does not fix a family**; not adopted |
| R1 (c): WaveFake + MLAAD-LJ ×0 | fold 1 0.9491 → 0.9482, fold 2 0.9771 → 0.9744 on v3 VAL; melo +.042 missed, mixed cells +.02 | **keep WaveFake at ×0.25** |
| held-out families (v4 VAL) | maskgct missed .41 en / .27 ko, seedvc .22, melo .21, mms .22 | **unseen generators are the failure** → v5 adds 12 new families |
| real in-the-wild Korean (v4 VAL) | Emilia-ko real called fake .15–.18 | → v5 +40 h YODAS-ko |

- **L4. Generator diversity, not hours per generator.** Every held-out family fails, and adding
  weight to a seen family does not transfer to its unseen neighbours.
- **L5. Real-side diversity matters as much.** Most of run 2's gain was fewer false alarms on real
  speech (LJ, MUSAN) from new real corpora.

## 4 · Objective and model

| experiment | result | decision |
|---|---|---|
| R1 (b): one-class loss (OC-Softmax, 0.02) | v3 VAL fold 1 0.9491 → 0.9498, fold 2 0.9771 → 0.9752; no slice gains ≥ 2 pts; melo slightly worse | **dropped** (fails the §5 rule of doc 13) |
| O5 file mode max3 | +0.002 file EER on 8/8 folds (VAL) | untested on LB → tonight's A/B |
| PROBE ensemble T6 + T7 | 0.9107 vs 0.9103 alone | no gain from same-recipe ensembles |
| XLS-R-1B trial (2 GPUs, fold 1, 2k steps × 32 = 64k samples, cold speech trunk) | fold-1 VAL 0.9235 (eval-n 2000) vs run-2 T1 0.9334 (320k samples on top of run 1) | the 1B path works end to end; the main run sees 20–30× more samples |
| 1B memory (2×H200) | per-rank 8: 1.18 s/step, 76 GiB; 12: 1.70 s, 112 GiB; 16: OOM; 16 + grad ckpt: 2.34 s, 48 GiB | main run: per-rank 16 with checkpointing |

- **L6. Objectives measured flat; data measured positive.** Spend the remaining time on data and
  scale, not loss tweaks.
- **L7. Warm init is worth a lot.** Scratch lost to init-from-run-1 (T4), and the 1B partial init
  keeps BEATs and the music/presence heads from T7.
  - A key-by-key partial load would have silently put 300M's CNN under 1B's transformer; the
    shapes are identical. Load whole modules or nothing.

## 5 · Shortcut cues we found (the most reusable lessons)

Each of these would let a model score well locally while learning something other than "is this
generated".

| # | cue | how found | fix |
|---|---|---|---|
| 5.1 | **Exact digital zeros.** Some TTS (CosyVoice 2/3, Spark, StyleTTS2) insert pauses of exact-zero samples; real recordings almost never do (Zeroth 0 %, Emilia/YODAS 1.5 % with a run > 20 ms) | agent QC on Spark-TTS, then calibrated on 200 real + fake files per source | drop files with an exact-zero run > 20 ms or > 5 % zeros inside the speech span, on **old and new** data (v4: 1,380 rows / 4.2 h dropped; en-synth2 CosyVoice 22 %). A stricter "> 1 % zeros" rule was **wrong**: it hit 23.5 % of real Zeroth (quiet samples that are exactly zero) |
| 5.2 | **Singing labelled "no voice".** 1,694 real "instrumental" clips (FMA 12.9 h, MUSAN 9.2 h) contain singing; the EDA speech detector misses singing. Fake instrumentals: 745 clips (2.1 h) | the singing agent's htdemucs vocal screen over all 41,447 music clips (78 % recall vs MUSAN's own annotation, 1.4 % false alarms) | dropped from both music pools in v5 (C −24.3 h, D −2.1 h); singing voice added as its own voice component (26 h fake, 17 h real) |
| 5.3 | **Asymmetric processing.** Real songs are 44.1 kHz stereo while SONICS is 16 kHz mono, so separating at native rates would leave a > 8 kHz band only on real stems | the singing agent's design | canonicalise both sides to 16 kHz mono **before** separation; identical screens both sides |
| 5.4 | **Language-only-on-one-side.** Korean AI singing without any real Korean singing makes "Korean singing" a fake cue | v5 filter | the Korean ACE-Step vocal stems are left out of v5 |
| 5.5 | **Repetition asymmetry.** Tiny speaker buckets make voice slots repeat audio, more on the fake side. v5: repeats fake .441 vs real .36; draw-feature AUC for voice_fake 0.597 (gate 0.60) | draw_balance / audit I1c | v4 was tuned to pass; v5 passes narrowly. **Watch**; the next rebuild should tune it down |
| 5.6 | **Shared speaker buckets across corpora.** MLAAD-LJ shares WaveFake's bucket, so zeroing WaveFake left a 0.75 % residue | R1 (c) config | zero every family in the bucket |
| 5.7 | **Codec round-trips are REAL** (DACON A1): post-processing that creates no new component | talkboard ruling | never label codec/vocoder copy-synthesis of real speech as fake |

- **L8. Every new data source needs a symmetry check before it is trusted.** The audits (I1c, I3,
  I5) catch draw-level cues; the audio-level cues (5.1–5.4) were caught only by looking at the
  audio.

## 6 · Process and infrastructure

- **L9. Judge by per-slice EER, not pooled.** Pooled VAL looked fine while held-out families
  failed (maskgct .41).
- **L10. Pin folds.** Re-rotating folds when data is added breaks fold-k init, PROBE, and
  like-for-like comparison (`scripts/build_folds_pinned.py`).
- **L11. DDP is exact when normalised globally.**
  - A per-rank masked mean silently down-weights rare components. The fix is `GlobalNorm`:
    global present counts, times world.
  - Tests prove gradient equality with the single process, and bitwise resume on CPU.
  - On GPU two fresh identical runs already differ (max |Δw| 3e-4 at 150 steps), so GPU resume
    cannot be bitwise; judge it by the loss (it agreed to 6e-7).
- **L12. A module outside the known list never trains** unless the trainable-parameter collector
  knows it. `trainable_parameters` now takes every top-level module; a test guards it.
- **L13. Capacity:**
  - The node exposes **64 usable CPUs** (not 96): 8 per GPU.
  - Idle-GPU slots went unused when agents were blocked on waits. Reassign after ~30 min.
- **L14. Shared-corpus hygiene.** Someone flattened `emilia-ko/fleurs/<split>/<split>/`, breaking
  2,460 paths; it was repaired with self-symlinks. Rule: nobody moves files under `dacon-corpus/`
  outside their own new directory.
- **L15. Licence surprises:**
  - OuteTTS and Higgs (Llama 3.2 / Boson) carry a naming clause (owner to decide).
  - The Orpheus Korean weights are gated.
  - Higgs v3 forbids voice cloning without consent.
  - Commercial APIs are excluded by ToS.

## 7 · Where this leaves us (10:00 Sunday)

- **strategy-v5:** 590,316 rows, 3,324 h. Audits pass on folds 0 and 1; the all-data audit is
  finishing.
  - +12 voice families (Korean: CosyVoice3, Qwen3-TTS, OuteTTS, F5-TTS-ko, Higgs v2, Supertonic,
    RVC; English: F5-TTS, Zonos, IndexTTS2, Spark-TTS, StyleTTS2).
  - +maskgct/seedvc hours.
  - Singing voice, AI songs with Korean/English lyrics, real songs, +40 h YODAS-ko.
  - Phone-channel augmentation (GSM, Opus, packet loss, compression).
  - The §5 fixes.
- **Running:**
  - 1B main (7 GPUs, DDP, all data, ~24 passes, to ~Mon 05:00; job 221491).
  - 300M T7 fine-tune on v5 (1 GPU, ~3 h; job 221492), to submit tonight with learned and max3
    file modes.
- **Open questions:**
  - Does v5 raise the LB (tonight)?
  - Does max3 beat the learned file head on the LB?
  - Does 1B beat 300M (Monday)?
  - Voice vs music split (diagB, not planned).
  - The OuteTTS/Higgs licence decision.
