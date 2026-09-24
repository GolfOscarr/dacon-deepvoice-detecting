# 07 — The first real training run: plan

*Written 2026-09-24 on `feat/training` (from `main` at `3d8e1b3`). Owner's instruction: "put all
things in the first run", on the Slurm `debug` node (`Slurm-GPU-Node-42`, 8× H200, 64 CPUs
allocatable), all eight GPUs used. The leaderboard closes **2026-09-29 10:00 KST**; 3
submissions per day.*

Owner decisions of 2026-09-24:

- **Compute**: the Slurm debug node, all 8 GPUs, through Slurm.
- **Scope**: everything in the first run. The model revisions, the speech branch, the data
  mixture fixes and the new data land before the first training job, not in later rounds.
- **Korean fake voice**: "we need to synthesize this set" — we synthesise it ourselves.
- **Fake music**: left to the implementer. The choice, Demucs-separated SONICS instrumentals
  applied symmetrically to real music, is §1 D-b.

## 0 · What the corpus lacks (measured on strategy-v2)

| gap | measurement | consequence |
|---|---|---|
| Korean fake voice | real 52.9 h (Zeroth) vs fake 0.9 h (MLAAD `ko`) | "Korean ⇒ real" is learnable |
| English fake voice is one speaker | 181 h of 288 h fake voice is WaveFake LJ | old vocoders, one voice |
| Japanese only fake | JSUT 13.4 h fake, 0 h real | "Japanese ⇒ fake" |
| Modern fake music unused | SONICS 1,971 h never drawn under f8 = 1; pool D = FakeMusicCaps only (5 families, 10 s clips) | Suno / Udio unseen in training |
| Multi-speaker English real | LibriTTS-R, Common Voice acquired to S3, absent locally | English real is LJ + MUSAN only |

## 1 · Data — strategy-v3 (a new manifest directory; strategy-v2 is not touched)

**D-a · Korean fake voice, synthesised (new corpus `ko-synth`, pool B).**
- **Families.** At least six generator families, each a separate `artifact_family`, so the
  folds can rotate them. MIT / Apache first; non-commercial licences are acceptable under rule
  2.1 but each is recorded. Candidates, verified by the synthesis track:
  - MMS-TTS `kor` (VITS)
  - MeloTTS-KR
  - XTTS-v2 zero-shot cloning
  - OpenVoice-V2 tone-colour conversion
  - kNN-VC (WavLM)
  - CosyVoice2 or Fish-Speech zero-shot, whichever supports Korean
  - Bark with Korean presets
- **Text and speakers.** Texts come from Zeroth transcripts. Cloning prompts come from Zeroth
  speakers. A cloned file's `speaker_ref_id` is its prompt speaker, so a clone and its source
  speaker share a fold atom. `pair_id` is set where text and speaker equal a real utterance.
- **Target.** ≥ 40 h, every file ≥ 3 s, 16 kHz or higher.
- **Reproducibility.** A `metadata.csv` per family records text, prompt file, model name and
  revision, seed and licence (stage-2 rule: self-generated data ships with its recipe).

**D-b · Fake music: SONICS instrumentals by Demucs, applied to both sides.** `htdemucs` (MIT)
`no_vocals` stems:
- **Fake side.** A SONICS subset (~60 h, spread over its generator families) becomes pool D,
  corpus `sonics-sep`.
- **Real side.** A pool-C subset of the same order (fma + MUSAN music) passes through the same
  model and becomes corpus `realmusic-sep`. Separation is a transform both labels carry, never a
  cue. DACON #417333 A1: separation output keeps its label.

**D-c · Real voice.** LibriTTS-R (CC BY 4.0) — a ~50 h subset of many speakers — and Common
Voice `ko` (CC0), from S3 via `scripts/fetch_from_s3.py`, into pool A.

**D-d · Language balance in the draw.**
- **The `lang` column.** The manifest gains `lang` (ko / en / zh / ja / other), derived per
  corpus.
- **Balanced shares.** Within the voice pools the draw weights `(pool, lang)` so each
  language's share is the same on the real side (A) and the fake side (B). A language present on
  one side only gets a small fixed share (`other`) or 0 (`ja`, i.e. JSUT out).
- **Where it applies.** On top of the DOSS cap. The audit gains a `lang` feature in I1b so the
  balance is measured, not assumed.

Then, in order: manifest `strategy-v3`, folds, cache extension, gates and the fold-0 audit
(04's commands with `strategy-v3`).

## 2 · Model — `configs/c_first_run.yaml`

- **The audio trunk.** BEATs, all 12 layers (truncation was unmeasured and the L4 budget
  allows it). LoRA r16 on `q_proj k_proj v_proj out_proj fc1 fc2`, GeM frequency pooling.
- **The speech trunk.** XLS-R-300M (Apache-2.0, `facebook/wav2vec2-xls-r-300m`), 12 of 24
  layers, LoRA r16 on the same projection names, fps 50. It is a new `XLSRFrontend` on
  `transformers==4.57.6`, the version the test server preinstalls.
- **Branches.**
  - voice and file read `[speech, audio]` aligned to `audio`.
  - music, v_pres and m_pres read `audio`.
- **Everything else.** Heads, top-k pooling, softsign output and metric-proportional loss
  weights are unchanged from candidate A. No distillation (no teacher is loadable).

## 3 · Training

- **One joint stage.**
  - S1 is dropped: it runs 5× the passes, one branch group at a time.
  - S3's four-way codec expansion is dropped: the processing draw's `normalize_menu` already
    applies mp3 / flac / telephone per sample, and S3 would apply codecs twice.
- **Parallel rendering.** A pool of worker processes renders batches ahead of the GPU. A
  render is a pure function of its spec, so parallelism changes no sample.
- **Optimiser.** AdamW with two groups: the heads, and LoRA plus GeM at a lower rate. Linear
  warmup then cosine decay, bf16, EMA 0.999, grad clip 5, a checkpoint every N steps.
- **Eight runs on eight GPUs**, one Slurm job step each:
  - folds 0–3, each training on its fold's TRAIN and validating on its VAL — the CV estimate
    and the step count;
  - four all-data runs, TRAIN + VAL of every fold, seeds 0–3 — the models that ship, as a soup
    or ensemble, picked at the step the fold runs point to.
- **CPU budget.** 64 allocatable CPUs means about 7 render workers per run. Measured render
  cost is ~3 samples/s per core, so the run is CPU-bound near 20 samples/s. The step count is
  set from that, not from epochs.

## 4 · Before the first submission

- **L4 runtime.** Measure on an H200 and scale; the budget is 60 min for 1,200 files up to
  60 s.
- **`script.py`** loads both frontends from `model/`.
- **PROBE** is checked once, on the chosen models.

## 5 · Tracks, run in parallel

| track | owner | GPUs | output |
|---|---|---|---|
| K · Korean synthesis (D-a) | agent | 5 | `interim/ko-synth/<family>/`, `metadata.csv` per family, scripts under `scripts/synth/` |
| M · Music separation + real-voice sync (D-b, D-c) | agent | 2 | `interim/sonics-sep/`, `interim/realmusic-sep/`, `interim/libritts-r/`, `interim/common-voice-ko/` |
| X · XLS-R frontend + model config (§2) | agent, own worktree | 1 (smoke) | `XLSRFrontend`, `configs/c_first_run.yaml`, tests |
| T · Training loop, draw weights, launcher (§1 D-d, §3) | main session | — | parallel render, schedule, `lang` weights, `scripts/train_first_run.sbatch` |
| I · Integration | main session | 8 | strategy-v3, cache, audit, the eight runs |

All GPU work goes through `sbatch` / `srun -p debug`. Tracks K and M write data only under
`/data/project/private/dacon-corpus/`. They do not commit; the main session reviews and
commits their scripts.

## 6 · As built and measured (2026-09-25 00:30 KST)

The run `first-v3` is Slurm array 218976: tasks 0-3 are folds 0-3, tasks 4-7 are
all-data seeds 0-3. Outputs go to `/data/project/private/dacon-runs/first-v3/` (symlinked as
`runs/first-v3`). It runs 18 passes × 48,000 draws at batch 16, about 54,000 steps per task.
Measured at 1.24–1.40 s/step on all eight tasks at once, so about 20 h.

**Data (strategy-v3, 400,944 rows)**

| corpus | rows | h | note |
|---|---|---|---|
| ko-synth | 19,432 | 39.6 | 7 families (mms, melo, knnvc, openvoice, xtts, cosyvoice, bark); cosyvoice and bark stopped early to free GPUs |
| sonics-sep | 3,458 | 62.9 | 5 generator versions; vocal-bleed screen, same rule both sides |
| realmusic-sep | 3,853 | 41.8 | fma + MUSAN music through the same htdemucs |
| libritts-r | 24,206 | 49.6 | 243 speakers; 9,026 clips under the 2.5 s floor dropped |
| common-voice-ko | 1,697 | 2.7 | validated.tsv only |

- **Folds.** VG1 passes. Every VAL fold has ≥ 2 fake-music families, so the one-family
  music caveat is gone.
- **Fake voice VAL hours.** 182 / 22 / 28 / 28 h across folds 0–3 (the LJ atom).
- **Cache.** 49,638 files added, 0 failures.

**The draw.** The audit passes 25/25 at n = 40,000.

- **Language shares.** Real voice draws ko 0.536 / en 0.464; fake voice draws ko 0.497 /
  en 0.442, plus MLAAD's long tail.
- **CFAD sits in PROBE.** Its 59.4 h of Chinese fakes are sealed in PROBE with their pair
  atoms, so Chinese is dropped from every train and VAL view. It had been Chinese on the
  real side only: 20 % of real draws against 0 % of fake.
- **I3 threshold.** I3 reads ≈ 0.5 at n = 20,000 by construction (2p(1−p) = 0.48 for a
  component drawn twice), so audit at n ≥ 40,000.

**Model (`configs/c_first_run.yaml`).**

- **Trunks.** BEATs has 12 layers, LoRA on six projections, and encodes in 62-column
  (9.92 s) windows. XLS-R-300M uses 12 of its 24 layers. Of 264M parameters, 9.7M are
  trainable.
- **Alignment.** XLS-R's 50 fps is pooled onto BEATs' 6.25 fps grid.

**Training.**

- **Settings.** One joint stage, batch 16, lr 2e-4 with a 0.2× frontend rate, warmup 1,000
  steps, then cosine to 5 %. EMA 0.999, layer checkpointing, 7 render workers per task.
  Steps take 1.24–1.40 s.

What stood between the first launch and that speed, in order:

| symptom | cause | fix |
|---|---|---|
| OOM at batch 32, then at 8 | both trunks stored every layer's attention maps | layer checkpointing: 100 → 22 GiB at 8 × 60 s |
| memory +0.75 GB / step, then cuFFT alloc failure | cuFFT cached a plan per transform length, outside PyTorch's allocator | `bandpass` caps the plan cache at 8 (also an L4 inference leak) |
| 3.9 s/step, 3.7 s in BEATs | one 3,000-token attention per 60 s row | 62-column windows: 0.79 s |
| 2.7 s/step on real batches | a separate encoder call per partial-window length (launch-bound) | pad windows and use the encoder's padding mask: 1.24 s |
| 22 s/step with 8 tasks, GPUs idle | CUDA JIT cache `~/.nv` on EFS, every task blocked on its index over NFS | `CUDA_CACHE_PATH` on node-local /tmp |

**Submission path.** `scripts/package_submission.py` has been verified on the 100-step smoke
model.

- **Offline load.** The zip is 1.66 GB. `script.py`, run in a venv that mirrors the server
  (`/data/project/private/dacon-venvs/server-mirror`: torch 2.7.1+cu128, numpy 1.26.4,
  pandas 2.0.3, transformers 4.57.6), loaded the model offline from `model/weights/`.
- **Result.** It scored WAV / FLAC / MP3 files with no fallback row.
- **Not measured.** L4 wall time. The pre-window estimate was 29 min fp32 worst case.

**Open.**

- **L4 wall time.**
- **Choosing between the four all-data runs.** They share their initialisation (the model
  seed is fixed), so the plan is a uniform soup of their EMA weights, checked on PROBE
  against the single runs.
- **The remaining synthesis.** CosyVoice and Bark can resume if a second run is wanted
  (`scripts/synth/run_family.sbatch`).
