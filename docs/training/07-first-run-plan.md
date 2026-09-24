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
