# 17 · Final runs (Mon 2026-09-28 → Tue 09-29): 1B on v6b/v6c, soups, ensembles

The closing 30 hours of the competition: the XLS-R-1B main run finished, three new data
versions (v6, v6b, v6c) were built, five fine-tune runs were planned (four started), and the final packages are
soups and a two-model ensemble. Read with [16-lessons-learned.md](16-lessons-learned.md) (runs 1–3,
strategy-v5) and [../HANDOFF_RUN4.md](../HANDOFF_RUN4.md) (the 1B run's operations).

## 1 · The one-paragraph summary

- **The v6b music fix was the biggest late gain:** LB 0.83046 → **0.85441** (+0.024, almost all ADS,
  0.812 → 0.838), from real music diversity (MTG-Jamendo), a fair real/fake music draw (10 s cut)
  and newer fake music (ACE-Step 1.5), on the 1B model.
- **Every model peaks mid-run.** The 1B main run peaked at pass 15 of 24, run A at pass 8 of 15, run C
  at pass 5 of 11. Training loss keeps falling while PROBE falls back, so **soups of the mid-run passes**
  are the final models.
- **max3 beats the learned file head on every model** (+0.005 to +0.012 PROBE).
- **Averaging with weaker (v5) models hurts;** averaging two strong models from different data does not
  hurt and helps voice (see §6).
- **Continuing a converged model on shifted data at a low LR hurts** (run D).

## 2 · Leaderboard

| submission | LB | ADS | CPS | PROBE (max3) |
|---|---|---|---|---|
| run5-T7v5-max3 (300M, v5) | 0.83046 | 0.81196 | 0.99698 | 0.9209 |
| **1b-v6bA-p2-max3** (run A pass 2) | **0.85441** | **0.83845** | **0.99806** | 0.9627 |
| ens-1b-v6bA8-v6cC2-max3 (Mon slot 3) | *pending* | | | 0.9639 |
| ens-soupA58-soupC567-max3 (Tue, planned slot 1) | *pending* | | | 0.9658 |
| 1b-soupC567-max3 (Tue, planned slot 2) | *pending* | | | 0.9652 |
| diagB (T7, voice columns only) | 0.57479 | 0.55644 | 0.73985 | – |

- **PROBE vs LB:** PROBE predicted +0.042 for v6b over 300M-v5; the LB gave +0.024 (over-read about
  1.75×, down from ~3× in runs 1–2). PROBE differences above ~0.005 are trustworthy in sign.
- **diagB** (Mon 03:30) completed T7's per-head test split: **file 0.202, voice 0.218, music 0.209 EER**,
  voice-presence AUC 0.980. Music costs 0.063 of ADS vs voice 0.044 (weights .3 vs .2), and music had
  the biggest VAL→test gap (~0.02 → 0.21). That result chose the music-first data plan below.

## 3 · Runs

All runs: XLS-R-1B@24 + BEATs, per-branch layer fusion, LoRA, SED heads (`configs/c_1b_fusion.yaml`),
DDP on 8×H200, per-rank batch 14 (global 112), 128,000 draws per pass (1,142 steps), bf16, EMA 0.999,
cosine to 5 % of peak, frontend LR ×0.5, loss weights file .45 / music .27 / voice .18 / presence .05.
Code: worktree `dacon-ddp-1b`, branch `feat/ddp-1b`. Launcher: `_ops/train_resumable.sbatch`
(requeue-safe: resumes from the newest complete checkpoint).

| run | job | data | init | LR peak | passes | node | status | best PROBE (max3) |
|---|---|---|---|---|---|---|---|---|
| main 1B | 221709 | v5 | pretrained | 3e-4 | 24 | 42 | done (20 h 47 m) | pass 15: 0.9510 |
| **A** | 222290 | **v6b** | main pass 15 | 1.5e-4 | 15 (stopped after 12) | 68 | stopped 22:40 (past its best) | **soup 5+8: 0.9649** |
| B | – | v6b | main pass 15 | 1e-4 | 12 | 42 | cancelled before start (owner: node 42 for the v6c ablation) | – |
| **C** | 222620 | **v6c** + voice fixes | main pass 15 | 1.5e-4 | 11 | 42 | done | **soup 5–7: 0.9652, 5–8: 0.9654** |
| D | 222733 | v6c + voice fixes | A pass 8 | 5e-5 | 6 (stopped after 3) | 68 | stopped 04:10 (got worse) | – |

Configs (worktree): `processing_1b_ft.yaml` (A: run 5 draw, hold 0, warmup 300),
`processing_1b_ft6c.yaml` (C, D: + voice fixes), `train_1b_ft.yaml` (1.5e-4),
`train_1b_ft_lr1e4.yaml` (B), `train_1b_ft_lr5e5.yaml` (D).

### 3.1 PROBE by checkpoint (v3 sealed PROBE, eval-n 3000)

| run · pass | learned | max3 | file | voice | music |
|---|---|---|---|---|---|
| main · 12 | 0.9411 | | 0.078 | 0.038 | 0.063 |
| main · 15 | 0.9452 | 0.9510 | 0.070 | 0.050 | 0.053 |
| main · 18 | 0.9354 | | 0.084 | 0.050 | 0.066 |
| main · 21 | 0.9367 | | 0.082 | 0.049 | 0.065 |
| main · final (23) | 0.9354 | 0.9471 | 0.085 | 0.049 | 0.065 |
| A · 2 | 0.9578 | 0.9627 | 0.055 | 0.045 | 0.035 |
| A · 4 | | 0.9626 | | 0.046 | 0.035 |
| A · 5 | 0.9586 | | 0.055 | 0.048 | 0.030 |
| A · 8 | 0.9590 | 0.9636 | 0.054 | 0.049 | 0.029 |
| A · 11 | 0.9577 | | 0.052 | 0.049 | 0.037 |
| C · 2 | 0.9574 | 0.9611 | 0.056 | 0.044 | 0.036 |
| C · 5 | 0.9595 | 0.9644 | 0.054 | 0.047 | 0.029 |
| C · 8 | 0.9575 | | 0.060 | 0.045 | 0.027 |
| D · 1 | 0.9558 | | 0.059 | 0.051 | 0.032 |
| D · 3 | 0.9530 | | 0.064 | 0.054 | 0.032 |

EER columns are the learned-head row where both exist. With max3 the file EER of the v6 models is
0.042–0.047.

### 3.2 Readings
- **Mid-run peak, every run.** Train loss for A fell 0.027 → 0.018 across passes 1–12 while PROBE
  peaked at pass 8; C peaked at pass 5 (its 11-pass cosine decays faster). The late low-LR passes fit
  the seen files, not the held-out ones. Next time: shorter schedules, or keep the mid-run checkpoints
  and soup them (both done here).
- **v6c + voice fixes vs v6b (C vs A):** a PROBE tie at matched passes (C best learned 0.9595 vs A
  0.9590). C has the best voice (0.044–0.045) and music (0.027) of any model. The voice fixes target
  the real test's channel and noise-floor domain, which PROBE only partly contains; the LB result of
  the ensemble is the only read of that.
- **Run D:** continuing A pass 8 (0.9590) on v6c at 5e-5 fell to 0.9558 by pass 1 and 0.9530 by pass 3, on every head. Shifted data
  (new music + heavier noise/MP3 augmentation) on a converged model moved it off its optimum and the
  low LR could not re-converge in 3 passes. Starting from the earlier checkpoint (C: pass 15, 1.5e-4)
  was the right way to add data.
- **LR 1.5e-4 from a trained model:** voice train loss rose for ~1 pass (0.043 → 0.057) and recovered;
  PROBE showed no damage. Untested alternative: 1e-4 (run B, cancelled).

## 4 · Data: strategy-v6 → v6b → v6c

Built in the main repo (`feat/training`), all on top of strategy-v5 (never written). Folds pinned to
the base; PROBE membership never changes; audits at **n=80000** on fold 0, fold 1 and all-data.

### 4.1 What was added

| version | rows | change | build |
|---|---|---|---|
| v6 | 635,560 | + MTG-Jamendo real instrumentals 23.7 h (2,843 clips, 1,476 artists); all music rows cut to 10 s; music exact-zero rule; sonics-sep subsampled to 1/3 of fake music | `scripts/build_strategy_v6.sh` |
| v6b | 641,220 | + ACE-Step 1.5 instrumentals 15.7 h (1,888 of 2,400 clips kept) | `scripts/build_strategy_v6b.sh` |
| v6c | 652,165 | + Jamendo batch 2 20.8 h (2,502 clips) + ACE-Step batch 2 9.6 h (1,154 of 1,504) | `scripts/build_strategy_v6c.sh` |

### 4.2 How the music mixture works (and the trap)
- **Domain weights cannot steer music.** Music has one bucket per side (`processing/sampler.py`
  `bucket_keys` → `"*"`) and tile files are drawn uniformly among the rows that fit (`_tiles`), so a
  corpus's share of music tiles equals its **row count**. A Jamendo weight of ×54 left it at .229.
- **Fix: one cut rule, both sides** (`scripts/strategy/cut_music_rows.py`): every pool C/D music row
  ≥ 15 s becomes consecutive 10 s rows (last piece kept if ≥ 5 s). Tiles are 1.5–2.5 s takes, so any
  row ≥ 3.5 s can serve them; share now follows **hours**. Pieces inherit labels, fold, slice and
  groups (`cut_map.parquet` maps piece → original file, offset, licence).
- **Tile shares (all-data):**

| side | v5 | v6b | v6c |
|---|---|---|---|
| fake | FakeMusicCaps .85 / Suno-Udio (sonics-sep) .15 | FMC .49 / sonics .35 / ACE-Step .16 | FMC .44 / sonics .32 / ACE-Step .24 |
| real | FMA .57 / realmusic-sep .40 / MUSAN .04 | FMA .34 / realmusic-sep .30 / Jamendo .18 / MUSAN .18 | FMA .29 / realmusic-sep .26 / Jamendo .30 / MUSAN .15 |

### 4.3 Cues found and fixed (the audit found them before training did)
1. **Exact-zero gaps in fake music:** 12.3 % of FakeMusicCaps rows had a zero run > 20 ms (real ≤ 0.4 %).
   The §5.1 rule of doc 16 now applies to every music piece, both sides (3,192 FMC rows dropped).
2. **sonics-sep spectral tilt** (95 % rolloff median 445 Hz): after the cut raised its share to .52,
   its file-level AUC rose 0.51 → 0.61. Subsampled deterministically by parent to 1/3 of fake music.
3. **Level:** ACE-Step peaks 0.54–0.81 vs Jamendo ~0.95 — neutralised by the render's −23 dBFS tile
   RMS normalisation (`processing/render.py`).
4. **Format symmetry:** ACE-Step clips are encoded like Jamendo's source (44.1 kHz mono MP3 ~96 kbps),
   middle 30 s excerpts, same 10 s cut, same AST vocal screen (vox_max < 0.03).

### 4.4 Screens
- **Vocal screen (CPU):** AST AudioSet tagger, `vox_max < 0.03` over every 10 s window; calibrated
  against htdemucs on 384 clips (1/214 misses) and MUSAN's labels (≤ 4.6 % contamination). Independent
  check on 40 kept clips with htdemucs + Silero + Whisper: 0/40 with vocals. ACE-Step still sings in
  21–23 % of "[Instrumental]" generations; those are dropped.
- **Songs with vocals are not in training:** whole-file rows (cells 5 and 8) are never drawn while
  `f8 = 1`, so only instrumentals were added.

### 4.5 Voice-side audit (run C's config)
Tile-level features that survive the ship chain separate real from fake voice at AUC 0.62 on held-out
corpora (0.77 in-corpus). Real Emilia (the largest real source) reads fake-like. Fixed in
`processing_1b_ft6c.yaml`, all label-independent draws:
- pause noise floor / exact zeros: gaussian and pink noise p .3 → .5, SNR 10–30 → 10–55 dB;
- crest factor: compression p .2 → .4;
- MP3-only reals (Emilia, Common Voice): container menu wav+flac .42 → .18.

Not fixed (needs code): low-frequency rumble in pauses (random high-pass), edge-silence trim.
Evidence: `/data/project/private/dacon-runs/_voice_audit/`.

### 4.6 Audits (v6c, verbatim numbers)
I3 fold 0 0.517, fold 1 0.513, all-data **0.505** (gate ≥ 0.5 — n=40000 fails it because 10 s pieces
are drawn only ~twice; n=80000 is the standard now); I1c music_fake 0.530, voice_fake 0.592–0.597;
I1b/I1bp (run C's augmentation) 0.49–0.51. All `audit ok: True`.

## 5 · Inference ablations (PROBE, max3 unless noted)

| candidate | PROBE | note |
|---|---|---|
| file mode on A pass 2: learned / max3 / max | 0.9578 / 0.9627 / 0.9635 | max ≈ max3; max3 kept (LB-proven) |
| A pass 2 + main pass 15 (prob mean) | 0.9588 | worse than A alone: weaker v5 member dilutes |
| A pass 2 + 300M v5 | 0.9534 | worse |
| A pass 8 + C pass 2 | 0.9639 | = A pass 8 (0.9636); voice 0.049 → 0.043 |
| soup A 5+8 / 4+5+8 | 0.9649 / 0.9642 | |
| soup C 5+8 / 5–7 / 5–8 | 0.9650 / 0.9652 / 0.9654 | |
| weight soup of soup-A and soup-C (one model) | 0.9652 | A and C share an init, so weights average cleanly |
| **ensemble soup-A 5+8 + soup-C 5+8** | **0.9657** | |
| **ensemble soup-A 5+8 + soup-C 5–7 (shipped)** | **0.9658** | prob mean; logit mean 0.9653 |

Server runtime: single 1B ≈ a few minutes for 1,200 files; the two-model package took 1.7× a single
model on the CPU mirror, far inside the 60-minute limit.

## 6 · Final packages (S3 `dacon-deepfake-detection/submissions/`, all server-mirror checked)

| package | contents | PROBE | size |
|---|---|---|---|
| **ens-soupA58-soupC567-max3.zip** | run A passes 5+8 soup + run C passes 5–7 soup, per-file prob mean | **0.9658** | 6.1 GB |
| **1b-soupC567-max3.zip** | run C passes 5–7 soup | 0.9652 | 4.1 GB |
| ens-1b-v6bA8-v6cC2-max3.zip | A pass 8 + C pass 2 (Mon slot 3) | 0.9639 | 6.1 GB |
| 1b-v6bA-p8-max3.zip, 1b-v6cC-p2-max3.zip, 1b-v6bA-p2-max3.zip (LB 0.85441), 1b-v5-p15-max3.zip | singles | 0.9636 / 0.9611 / 0.9627 / 0.9510 | 4.1 GB |

The packages are built by `scripts/package_submission.py` (`--member SCORED::WEIGHTS` for the
ensemble) and `_ops/pkg_run.sh`. The soup checkpoints are in `/data/project/private/dacon-runs/_ablate/`
(`soupA_58.pt`, `soupC_567.pt`, `soupC_5678.pt`, `xsoup_A58_C567.pt`).

## 7 · Process lessons from these two days
- **The shell is the training node** (Slurm-GPU-Node-42): 32 CPU htdemucs shards slowed DDP training
  by 60 %. Side work is capped at 24 threads under `nice -n 19`, or runs as a Slurm job.
- **Owner cap: node 42 + one normal node, nothing else.** Subagents spread jobs over six other nodes
  before the rule was stated; every job is now pinned with `--nodelist`.
- **Another user's CPU load on node 42** (a VS Code search, ~91 cores) halved run C's speed for ~30 min;
  not ours to stop.
- **`--workers 4 --threads 4` means one process** in the ACE-Step finalize; the AST screen then ran 6×
  slower until relaunched with 12 processes.
- **Checkpoint writes are not atomic;** the requeue wrapper skips a checkpoint that is not a complete
  zip.
- **A soup or a mid-run checkpoint beats the last one**, so every run now saves every 400 steps and
  every pass (≈ 4.9 GB each; the directories are large, prune after the competition).
- **Licences:** the owner chose to comply with the Llama 3.2 naming clause (OuteTTS, Higgs are in v5+),
  so models trained on v5+ data carry a "Llama" name prefix and "Built with Llama" attribution. ND music (FMA, Jamendo) is
  shipped as originals + deterministic code (talkboard A5), never as the cut pieces.

## 8 · Open at the time of writing
- The LB results of the Monday-night ensemble and of Tuesday's submissions: record them in §2.
- Voice fixes needing code (high-pass rumble augmentation, edge-silence trim) and a new fake-music
  family (ACE-Step v1 was prepared and put on hold by the owner) were not done.
- `processing/*` ingesters that look up parents (`proc`, `sing`, `realmusic-sep`) fail on a manifest
  whose music parents were cut; rebuilds must start from v5.
