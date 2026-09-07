# 02 — What We Submit (Code Submission Guide)

This is a **코드 제출 대회 (code-submission competition)**. We do *not* upload a predictions CSV.
We upload a zip containing weights + inference code; DACON runs it on their machines against
the hidden 1,200-file test set.

## Required zip structure

Directory and file names must match **exactly**. No extra top-level folder.

```
submit.zip
├── model/            # directory holding model weight files
│      └── (e.g. model.pt)
├── script.py         # the inference entry point, executed automatically
└── requirements.txt  # packages + versions, installable via `pip install -r`
```

At evaluation time the server adds two directories:

```
submit.zip
├── model/                    # ours
├── script.py                 # ours
├── requirements.txt          # ours
├── data/                     # auto-created: the real evaluation data (READ-ONLY)
└── output/submission.csv     # auto-created dir; our predictions must land here
```

- `data/` is **read-only** — no writing or modifying. It contains `data/test/` with the real
  1,200 files at the same path/structure as the dummy release.
- `output/` must end up containing a file named exactly **`submission.csv`**.
- `submission.csv` must have the **same rows as the 1,200 test IDs** and the **same columns as
  `sample_submission.csv`** (`ID` + the 5 probability columns). UTF-8 encoding.

> ⚠️ The official guide text contains a typo in one bullet — it says load data from `open/`
> where every other reference (including the structure diagram) says `data/`. The structure
> diagram is authoritative: read from `data/test/`, write to `output/submission.csv`.
> Defensive move: resolve paths relative to `script.py`'s own directory and fall back across
> `data/test` → `open/test` if the former is missing.

## Hard limits

| Limit | Value |
|---|---|
| Zip file size | **≤ 10 GB** (≤ 32 GB uncompressed) |
| Package install time | **≤ 10 min** (exceeding = install error) |
| Inference runtime | **≤ 60 min** for all 1,200 samples (exceeding = submission error) |
| Submissions per day | **3** |
| Language | Python only |
| Internet at runtime | ❌ **disabled** (except the pip install step) |

60 min / 1,200 files = **3.0 s per file average** on one L4. Files are 4 s–60 s. That is a
generous but not unlimited budget — a large ensemble with per-model resampling and source
separation (e.g. Demucs) can realistically blow it. Budget this explicitly and measure.

## Evaluation server specification

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA **L4**, 22.4 GiB VRAM |
| CPU | 6 vCPU |
| RAM | 28 GB |
| Python | **3.11.15** |
| CUDA | **12.8** |
| Internet | ❌ disabled (no external download after install) |

Because the runtime is offline, **all weights must be inside `model/`**. Anything that
lazily downloads at runtime (`from_pretrained("...")` hitting the hub, `torch.hub`,
`panns_inference` fetching checkpoints, Demucs downloading its models) **will fail**. Every
such library needs its cache pre-populated inside the zip and pointed at via env vars
(`HF_HOME`, `TORCH_HOME`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).

## Preinstalled packages (do NOT pin different versions)

DACON recommends using these and **not** listing them in `requirements.txt`; a conflicting
version can cause an install error.

```
torch==2.7.1+cu128        torchaudio==2.7.1+cu128
pandas==2.0.3             numpy==1.26.4
scipy==1.15.3             scikit-learn==1.8.0
joblib==1.5.3             threadpoolctl==3.6.0
transformers==4.57.6      accelerate==1.9.0
huggingface-hub==0.34.4   safetensors==0.6.2
sentencepiece==0.1.99     regex==2023.12.25
einops==0.8.1
librosa==0.10.2.post1     soundfile==0.12.1
soxr==0.5.0.post1
demucs==4.0.1             panns-inference==0.1.1
torchlibrosa==0.1.0       julius==0.2.7
tqdm==4.66.4              loguru==0.7.2
pyyaml==6.0.1             rich==13.7.1
matplotlib==3.10.8        diffq==0.2.4
```

**The preinstalled list is a strong hint about the intended solution shape.** `demucs`
(source separation — split voice/music), `panns-inference` + `torchlibrosa` (audio tagging —
voice/music presence), `librosa`/`soxr` (resampling). The organizers clearly anticipate a
separate-then-classify pipeline.

Preinstalled system packages:

```
git  ca-certificates  build-essential
python3.11  python3.11-dev  python3.11-venv  python3-pip
ffmpeg  libsndfile1  libsndfile1-dev  libffi-dev
libblas3  liblapack3  libomp-dev  libatlas-base-dev  gfortran
cmake  pkg-config  ninja-build  tzdata  unzip  p7zip-full
libgl1  libglib2.0-0
```

`ffmpeg` and `libsndfile1` are present → MP3/WAV/FLAC decoding via `soundfile`/`librosa`/
`torchaudio` will work.

## Two error classes (matters for the 3/day budget)

| Error | Cause | Counts against daily quota? |
|---|---|---|
| **설치 오류 (install error)** | wrong zip structure, package install failure | ❌ **No** |
| **제출 오류 (submission error)** | any error raised after `script.py` starts executing | ✅ **Yes** |

So a runtime crash burns one of only 3 daily attempts. `script.py` must be defensive:
never let a single corrupt/odd file abort the run — wrap per-file inference in try/except and
emit a fallback prediction (e.g. 0.5) so a full-length `submission.csv` is always written.

## Practical submission checklist

- [ ] Glob `data/test/*` for **any** extension, not `*.wav`.
- [ ] Handle **stereo** (downmix to mono) and confirm 16 kHz assumption, resample defensively.
- [ ] Handle lengths 4 s – 60 s (segmenting + pooling is allowed and encouraged).
- [ ] All weights local under `model/`; offline env vars set; no network calls.
- [ ] Wall-clock measured locally on ~1,200 representative files; keep margin under 60 min.
- [ ] `output/` created if absent; `submission.csv` written UTF-8 with exact columns.
- [ ] Per-file try/except with fallback row so the CSV is always complete.
- [ ] 🔴 **Startup assertions so a silent model failure cannot masquerade as a valid run** — see below.
- [ ] Zip has no wrapping top-level directory (`zip -r submit.zip model script.py requirements.txt`
      from inside the folder).
- [ ] `requirements.txt` only lists packages *not* already preinstalled.

## Other submission notes (from the official FAQ)

- Baseline code is reference only; we write our own code and reflect it in `requirements.txt`.
- Individual submissions **cannot be deleted**.
- **No submission is possible after the competition ends.**
- Best score shows on the leaderboard; per-submission scores are in the 제출 tab.
- Public Score updates in real time to the best score.
- **Ties are broken by who reached the score first.**

## 🔴 Guard against a silently-broken submission

A participant reported (2026-09-06, talkboard #417136) that **two submissions with different model
weights scored identically**. Whatever the cause there, it names a hazard our own design creates:
the per-file try/except fallback above will write a complete, well-formed `submission.csv` full of
0.5s even if the model never loaded — and that scores **exactly 0.5000** without raising an error.
With Private = Public and 3 submissions/day, a wasted day is expensive.

Required guards in `script.py`, all before inference starts:

```python
# 1. weights actually exist and are the expected size
assert os.path.getsize(WEIGHTS) > MIN_BYTES, "model weights missing/truncated"
# 2. checkpoint loaded with no unexpected/missing keys
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing and not unexpected, (missing, unexpected)
# 3. offline env is set, so nothing tries (and fails) to download
assert os.environ.get("HF_HUB_OFFLINE") == "1"
# 4. a fixed canned input produces a known-good fingerprint
assert abs(model(CANNED).item() - EXPECTED) < 1e-3, "model output drifted"
```

And after inference, before writing:

```python
# 5. predictions must not be degenerate
assert preds.std(axis=0).min() > 1e-6, "constant column -> model did not run"
# 6. fallback rows must be rare
assert n_fallback / len(ids) < 0.01, f"{n_fallback} files fell back"
```

Fail loudly. A **submission error** costs one of 3 daily attempts; a silent 0.5000 costs a day
*and* misleads us about whether the model works.
