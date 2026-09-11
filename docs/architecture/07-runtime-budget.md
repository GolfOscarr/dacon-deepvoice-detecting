# 07 — The L4 Inference Budget

🔴 **Every number in this file is an estimate until measured.** They are parameter-count
extrapolations, not benchmarks. They are written down so they can be *checked* — the protocol in
§4 is the part that matters, and it should replace §2 with real numbers as soon as it runs.

---

## 1. The budget, stated correctly

`3.0 s/file` is the wrong unit. Files are **4–60 s**, so a per-file average hides a 15× spread.

Think in **aggregate real-time factor**:

| | |
|---|---|
| Files | 1,200 |
| Duration each | 4–60 s |
| ⚠️ Mean duration | **unknown** — the test set is not released |
| If mean ≈ 30 s | ~10 h of audio |
| Wall clock | 60 min |
| **Required throughput** | **≈ 10× real-time**, decode included, on one L4 |

⚠️ The mean-duration assumption is doing real work here. If the test set skews long (mean 45 s),
the requirement becomes ~15× real-time and every margin below shrinks by a third. 🔷 Because we
cannot measure it, **design for the worst case: assume every file is 60 s.** That is 20 h of audio
and **20× real-time**, and it is the number to hold ourselves to.

Hardware: **L4, 22.4 GiB VRAM · 6 vCPU · 28 GB RAM** ([competition/02](../competition/02-submission.md)).

---

## 2. Estimated line items 🔷

Derived from parameter counts and typical SSL inference throughput. **Not measured.**

🔴 **Define the unit first.** An earlier version of this table multiplied a coverage factor into a
throughput that already accounted for coverage, and understated the margin by 3×. Throughout:

> **real-time factor = (seconds of *file duration* handled) / (wall-clock seconds)**

The key consequence: **full non-overlapping tiling processes exactly 1.0× the file duration**, so
it does *not* reduce this figure. Tiling is 3× the cost of a 4-crop sample only because that sample
covers just 33% of the file — the ×3 is relative to a baseline we rejected, not a charge against
the budget.

| Component | Est. RTF (per file duration) | Note |
|---|---|---|
| XLS-R-300M, full 24 layers, fp16 batched | ~50–150× | Commonly cited range for wav2vec2-class encoders on a mid GPU |
| Same, truncated to ~8–9 layers ([06](06-compression.md)) | ~150–400× | ~3× fewer transformer blocks |
| General-audio frontend, truncated | ~150–400× | Similar scale |
| **Two truncated frontends, full tiling** | **~75–200×** | ⭐ The shipped configuration |
| Same, whole-file single pass instead of tiling | ~47–125× | ~1.6× tiled ([04 §6](04-heads-and-pooling.md#6-temporal-coverage-tiling-not-sampling)) |
| 3 branch heads + 2 presence heads | negligible | Conv1d + attention over pooled features |
| Decode, 6 vCPU, **no resampling needed** | 🔷 ~500×+ | See §3 — much less of a threat than we first claimed |

🔷 **Against the 20× worst case, the margin is ~3.7–10× tiled, or ~2.3–6.3× whole-file.** That is
more headroom than the earlier arithmetic suggested — comfortable for candidate B, and enough that
a score-level ensemble member is plausibly affordable rather than obviously not. ⚠️ It is still an
extrapolation; [§4](#4-the-measurement-protocol-) decides, and
[05 §7](05-multi-model.md#7-the-decision-rule) still gates on measurement rather than on this
table.

★ It is also broadly consistent with the independent assessment in
[survey/05](../survey/05-models.md): 1× XLS-R 300M + 1× BEATs at 4 crops "comfortable"; three-model
XLS-R fusion "likely over budget"; adding HT-Demucs "tight — measure before committing."

---

## 3. The CPU side

⚠️ **We previously predicted this would be the binding constraint. That was almost certainly
wrong**, and it is recorded here rather than deleted because it was stated in three places and
would have misdirected effort away from the GPU path.

🔷 The arithmetic: the worst case is 20 h of audio; `ffmpeg` decodes MP3 at order 10²× real-time
*per core* across 6 cores; and **no resampling is needed at all**, because the test set is already
standardized to 16 kHz ([competition/01](../competition/01-overview.md)). That is single-digit
minutes against a 60-minute budget.

So decode is a thing to **measure and not to plan around**. The hazards below are still real —
they are ways to turn a 2-minute job into a 20-minute one — but they are implementation bugs, not
a structural constraint.

The work, per file: probe the container, decode MP3 / WAV / FLAC, downmix stereo, confirm or force
16 kHz, hand a tensor to the GPU. Across 1,200 files on **6 vCPU**.

⚠️ Specific hazards:

- `librosa.load` defaults to a high-quality resampler that is slow, and it is single-threaded per
  call. `soxr` (preinstalled, `soxr==0.5.0.post1`) is much faster.
- ❌ **Do not glob `*.wav`.** The dummy files are WAV; the real set is MP3/WAV/FLAC
  ([competition/01](../competition/01-overview.md)). Glob everything.
- If audio is already 16 kHz, **skip resampling entirely** rather than round-tripping through it.
- Decode must overlap GPU compute — a serial decode-then-infer loop wastes whichever resource is
  idle. A `DataLoader` with `num_workers` sized to 6 vCPU is the obvious form.
- ⚠️ 28 GB RAM with several worker processes each holding 60 s float32 buffers is not unlimited;
  prefer int16 until the last moment ([kaggle/06 §11](../kaggle/06-notebook-code.md)).

★ `ffmpeg` and `libsndfile1` are present, so decoding will *work*. The open question is only
whether it is fast enough, and it is answerable today — it does not need the model.

---

## 4. The measurement protocol 🔴

This is the actionable part of this file. Run it before committing to any configuration.

### 4a — First partial measurement, 2026-09-11 ★

**TRAINING only, and on one H200 rather than the L4** — so it does not touch the inference budget in
§2, which is still an extrapolation. What it does settle is the *binding constraint* question, and
the answer is **stage-dependent**:

| Stage | render | GPU | throughput | bound by |
|---|---|---|---|---|
| S2 `joint` (1-way codec) | 24.0% | 76.0% | 47.7 samp/s | **GPU** |
| S3 `codec_aware` (4-way, incl. mp3) | 75.5% | 24.5% | 14.4 samp/s | **decode** |

So this file's prediction — "the binding constraint is the 6 vCPU decode path, not the GPU" — is
**right for S3 and wrong for S2**. GPU time per sample is essentially equal across the two (0.38 s vs
0.41 s per 24 samples); the entire 3.3× gap is rendering, in `ffmpeg` subprocesses, and S3 also
expands 48 specs to 192 by design.

Rendering is inline and single-threaded on purpose (`training/loop.py:193`, which names
`DataLoader(dataset, batch_sampler=plan, collate_fn=collate)` as the alternative). 🔷 A worker pool
should recover most of S3's gap — **an inference from this measurement, not a measured speedup**, and
bounded by CPU count and process-spawn cost.

⚠️ Measured at 4–10 s only. The smoke corpus cannot compose longer samples (pool D is fixed 4.00 s
clips), so the 60 s end — where `whole_file` costs most — is still unmeasured.

**Build a proxy test set that matches the stated contract**, not our training data: 1,200 files,
durations sampled across 4–60 s, mixed MP3 / WAV / FLAC, mono **and** stereo, 16 kHz, with a
telephone-band subset. The dummy-file forensics in [data/07](../data/07-eda-plan.md) should
constrain the encoder settings.

Then measure, in this order:

| # | Measure | Why it comes first |
|---|---|---|
| 1 | **Decode + resample alone**, no model | Establishes the floor. If this alone is 30 min, no model choice saves us |
| 2 | Decode + model, **serial** | The naive implementation |
| 3 | Decode + model, **overlapped** | The real number |
| 4 | The same, with **every file forced to 60 s** | The worst case we actually design for |
| 5 | Cold start: import, weight load, `torch.compile`, startup assertions | ⚠️ Charged to the 60 min. Compile may not pay |

⚠️ Measure on hardware as close to an L4 as available, and **state the mapping** if it is not an
L4. An H200 measurement is not evidence about an L4; the ratio is neither constant nor small.

**Hold ≥30% margin.** A runtime overrun is a 제출 오류, which **consumes one of only 3 daily
submissions** — unlike an install error, which does not
([competition/02](../competition/02-submission.md)). With Private = Public, a wasted day near the
deadline is expensive.

---

## 5. How the budget gets spent

In priority order, from [05 §7](05-multi-model.md#7-the-decision-rule):

1. **Full temporal coverage.** A label-semantics requirement, not an optimization — a model that
   examines 33% of a 60 s file cannot see a fake segment in the other 67%.
2. **Two specialist frontends**, truncated. The core of candidate B.
3. Free combining — checkpoint soup, distillation. Costs nothing, so never traded.
4. **Measured** leftovers → overlap in the tiling, then score-level ensemble members, most-different
   first ([05 §5](05-multi-model.md#5-diversity-is-what-pays-not-count)).

⚠️ And the thing that is *not* on the list: **source separation in the inference path.** It is
rejected on accuracy grounds by four independent results ([03 G](03-candidates.md#g--frozen-separator--per-stem-detectors--rejected));
the runtime cost is merely the second reason.

---

## 6. Failure modes that are not about speed

The budget is not the only way inference goes wrong, and the others are cheaper to prevent than
to diagnose from a leaderboard score.

| Risk | Guard |
|---|---|
| 🔴 Silent 0.5000 — the per-file fallback writes a complete, well-formed CSV even if the model never loaded | Startup assertions: weight size, `load_state_dict` with no missing/unexpected keys, `HF_HUB_OFFLINE=1`, and a **canned-input fingerprint** |
| Degenerate output column | Post-inference assertion `preds.std(axis=0).min() > 1e-6` |
| Too many fallback rows | `n_fallback / len(ids) < 0.01` |
| One corrupt file aborting the run | Per-file try/except with a fallback row, so the CSV is always complete |
| ❌ Batch-dependent output (rule 2.4) | Assert a file's score alone vs inside a batch agrees to a tight tolerance **and preserves ranking** over a canned set — ⚠️ *not* bitwise equality, which fails spuriously (see [01 §1.4](01-design-envelope.md#14-rule-24--per-file-independence-)) |

All but the last are already specified in
[competition/02](../competition/02-submission.md#-guard-against-a-silently-broken-submission); the
last follows from [01 §1.4](01-design-envelope.md#14-rule-24--per-file-independence-).

⚠️ Note the interaction that makes this worth repeating here: **the fallback that protects us from
a 제출 오류 is the same mechanism that can hide a total model failure behind a valid-looking
0.5000.** A participant reported two different-weight submissions scoring identically
(talkboard #417136). Fail loudly at startup; fall back only per file, only after the model has
proven it loaded.
