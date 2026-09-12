# 00 — The Shared Harness

What every per-pool script emits, how it decodes, and how much of the corpus it touches.
Written before the pool plans because all five of them produce **one table with one schema**.

---

## 1 — 🔴 The output is a superset of the manifest, not a separate artifact

The temptation is to write EDA notebooks that print plots and then, separately, write a manifest
builder that re-crawls the corpus. That is two crawls of 291 GiB, two decode paths, and two chances
to disagree about what `duration_s` means.

**Contract:** the EDA pass emits `files.parquet`, and `training.manifest.REQUIRED_COLUMNS` is a
**projection** of its columns. The manifest builder becomes a column selection plus a label join —
it decodes nothing.

⚠️ `scripts/build_test_corpus.py` already computes a sha256 per file while hard-coding
`dup_group = None` — the leak that put 88 byte-identical recordings on both sides of a fold was a
value that was being computed and thrown away. **Emit every column you compute**, even where no
consumer exists yet; a column with no consumer costs a few bytes, and a consumer with no column
costs a corpus rebuild.

### 🔴 Three tables, not one — the tiers have different populations *and* different shapes

The M tier runs on **100%** of the corpus (~1.7 M files, Common Voice English alone) and emits
scalars. The S tier runs on a **stratified sample** (~2,000 per source) and emits `ltas[128]`,
`mel_band_skew[128]` and `mel_band_kurt[128]` per plane — 768 floats a row. Putting them in one
table forces a choice between 1.7 M rows of mostly-null vector columns and silently sampling the
thing the shortcut audit needs the *population* of ([06 X1](06-cross-pool.md)). Keep them apart:

```
eda/out/
  <pool>/files.parquet      M tier · 100% · scalars only   <- the manifest projects from THIS
  <pool>/signal.parquet     S tier · sampled · scalars     <- joins on file_id
  <pool>/vectors.npz        S tier · the [128]-arrays      <- keyed by file_id
  <pool>/segments.parquet   F tier · 50 ms frames
  <pool>/sample.json        the seeded draw (00 section 3)
  <pool>/parts/             per-shard outputs + DONE markers
  _shared/duplicates.parquet  the cross-pool hash + fingerprint matrix
  _shared/roles.md            E-S4: every column assigned feature / split key / leakage risk
  _shared/signal_chain.yaml   E-S1's output; the only thing the normalizer reads
  <pool>/report.md            the findings, with the numbers inline
```

Joining is on `file_id`, which is why `file_id` is assigned in the **M** tier and never recomputed:
a sampled row that cannot be joined back to its metadata is a row that cannot enter the manifest.

⚠️ **Parquet cannot be appended**, so per-shard parts plus a consolidation step are structural
rather than a convenience — and they are what makes an interrupted 1.7 M-file pass resumable. Write
a `DONE` marker per shard *after* the part, never before, which is the idiom
`scripts/fetch_to_s3.py` already uses and already learned the hard way
([data/12](../data/12-acquisition-status.md)).

⚠️ **The M tier's cost is process spawn, not decode.** `ffprobe` is ~20 ms per file: 1.7 M files is
~8 hours single-threaded and ~16 minutes across 32 workers. Size the pool before starting, not
after.

## 2 — 🔴 The two measurement planes (R1)

Every spectral and level statistic is computed twice.

| Plane | What it is | What it answers |
|---|---|---|
| **`native`** | the file exactly as the publisher shipped it — its own sample rate, channel count, container | *What is this file?* Corpus identity, provenance, generator fingerprints above 8 kHz |
| **`chain`** | after `decode → resample 16 kHz → channel policy → DC removal → (normalize)`, the identical code path the model and the eval server use | *What will the model see?* Everything that actually reaches a head |

Columns are suffixed `_native` / `_chain`. The derived column that matters most is the
**difference**: `bw_loss = effective_bandwidth_native − effective_bandwidth_chain`, and the
per-band energy deltas. A statistic that separates classes at `native` and stops separating them at
`chain` is a fingerprint that **does not survive the competition's own 16 kHz standardization** —
which is the premise the entire music-head strategy rests on and which no published work has
tested ([survey/10](../survey/10-open-questions.md)).

🔴 **Decode through `training.render.load_audio`, never through librosa defaults.**
`librosa.load` resamples with `soxr_hq` and normalizes; the training path does neither by default,
and `models.audio.prepare_waveform` is what applies the channel and level policy afterwards. An EDA
that measures a different signal than the model receives is measuring nothing. The `native` plane
bypasses both and uses `soundfile`/`ffmpeg` with **no** resampling and **no** normalization.

## 3 — Sampling: what is measured on 100% and what is not

291 GiB decoded once is a multi-day job on 6 vCPUs; the repo already measured render throughput at
14.4–47.7 samples/s ([PROGRESS](../../PROGRESS.md)). Split the work by cost:

| Tier | Cost | Coverage | Contents |
|---|---|---|---|
| **M — metadata** | `ffprobe` only, no decode, ~1 ms/file | **100% of every pool** | container, codec, native sr, channels, bitrate, encoder tag, duration, stream count |
| **S — signal** | full decode + STFT, ~50–200 ms/file | **stratified sample**, `min(2000, N)` per `source_name × generator/speaker`, seeded | levels, silence, bands, LTAS, VAD, statistics-T |
| **F — frame** | decode + 50 ms framing | S-sample only, plus **100% of any pool where a validity mask will be built** | `segments.parquet` |

🔴 **The sample is part of the corpus definition, not a convenience.** Seed it, write the drawn ids
to `eda/<pool>/sample.json`, and never redraw silently — the MLAAD cap is already precedent
([data/12](../data/12-acquisition-status.md)): a selection recorded with its seed is reproducible
from originals + code, which is what #417333 A6 makes the 2nd-stage submission rest on.

⚠️ Metadata tier being 100% is not optional. The shortcut audit ([06](06-cross-pool.md)) is a
*population* statement — "does container predict the label" cannot be answered on a sample
stratified by source, because stratifying by source is precisely what destroys the effect being
measured.

## 4 — The statistic set

Computed per file, at both planes unless marked.

**Level & dynamics**
`peak_dbfs`, `rms_dbfs`, `lufs_integrated`, `crest_factor`, `dc_offset`, `clipping_ratio`
(fraction of samples within 1 LSB of full scale), `statistics_T = std + var + rms + pwr`
☆ `[BirdCLEF 2024, 1st]` — validate the 0.8-quantile threshold on our own data rather than
inheriting it.

**Time structure**
`duration_s`, `silence_ratio` and `lead_silence_s` / `tail_silence_s` (energy segmentation:
0.05 s chunks, power in dB, −50 dB crossings ★ `[BC25 separation notebook]`),
`n_valid_spans_ge_4s`, `longest_valid_span_s`.

**Spectral**
Log-mel at 16 kHz: `n_fft=1024, hop=256, n_mels=128, fmin=20, fmax=8000` — the
`[BirdCLEF'25 B0]` grid (`32000/1024/512/128`) halved in hop to hold time resolution at our rate,
with `fmax` capped by Nyquist ([kaggle/06](../kaggle/06-notebook-code.md)). Native plane uses the
file's own rate with `fmax = sr/2`.
`ltas[128]` (long-term average spectrum, the mel means), `band_energy[8]` over 8 × 1 kHz bands,
`effective_bandwidth_hz` (highest frequency holding −60 dB relative to peak band),
`near_nyquist_ratio` = energy in `[0.94, 0.99]·Nyquist` over energy in `[0.80, 0.94]·Nyquist`,
`spectral_flatness`, `mel_band_skew[128]` and `mel_band_kurt[128]` (TISMIR's most-indicative music
features — ⚠️ recorded *because* they are expected to die at 16 kHz, see
[survey/02](../survey/02-sota-music.md)).

**Content**
`vad_speech_ratio` at Silero thresholds **0.5 and 0.4** (the community finding is that 0.4 recalls
better ★ `[BC25 separation notebook]`) and `vad_spans`; PANNs top-10 tags with scores.
⚠️ Both must be vendored, not `torch.hub.load`'d — the eval server is offline
([competition/02](../competition/02-submission.md)) — and PANNs conflates singing with Music, which
is an open item in [PROGRESS](../../PROGRESS.md), so the tag is evidence, never a label.

**Identity**
`sha256`, and a **content fingerprint** robust to re-encoding: the 128-bin mel mean quantized to
8 levels, hashed. Byte-identity caught 88 of the RIRS/MUSAN pairs; it would not have caught the
same recording re-encoded at a different bitrate, which is what redistribution across music
corpora actually looks like.

## 5 — Cost estimate

| Pool | Files (est.) | M tier | S tier (sampled) |
|---|---|---|---|
| A | ~1.5 M (CV-en dominates) | ~25 min | ~2,000 × 5 sources |
| B | ~16 k (mlaad) + ~117 k (wavefake) | ~3 min | ~2,000 × 2 |
| C | ~8 k (fma) + ~55 k (jamendo shard) | ~2 min | ~2,000 × 2 |
| D | ~27.6 k | ~1 min | **100%** — only 5 families, and it carries 0.27 |
| E | ~2 k | <1 min | **100%** |

🔴 **Pool D and pool E are measured in full.** D because five generator families is too few to
sample from and it carries more metric weight than voice; E because it is small and its last
surprise cost a corpus rebuild.

⚠️ **Unpack before you measure, and unpack once.** Nothing in S3 is extracted — all 291 GiB are
publisher archives. `scripts/fetch_from_s3.py --extract` is the path; egress is billed per read,
so pull to local disk and keep it. Cache decoded audio as `.pt` int16 tensors ★ `[BC2026
Distilled-SED]` — at 16 kHz mono that is 32 KB/s and both compact and re-readable by training.
