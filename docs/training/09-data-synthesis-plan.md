# 09 — Filling the data gaps by synthesis: priorities, methods, plan

*Written 2026-09-25 on `feat/training`, while run `first-v3` trains
([07 §6](07-first-run-plan.md#6--as-built-and-measured-2026-09-25-0030-kst)). Owner's brief:
prioritise the gaps, pick models or processing that can fill the urgent ones with 8 H200s,
and plan how to get good data fast. Facts marked **UNVERIFIED** were not measured when this was
written.*

## 0 · Constraints

**Deadline.** The leaderboard closes **2026-09-29 10:00 KST**. A full run is ~20 h on 8 GPUs
(07 §6). Working back from that:

| milestone | latest | why |
|---|---|---|
| run 2 starts | 2026-09-26 ~14:00 | ends 09-27 ~10:00, leaves room for a run 3 |
| synthesis finished and ingested | 2026-09-26 ~12:00 | strategy-v4 manifest, folds, cache and audit take ~1 h |
| GPU synthesis window | 09-25 ~21:00 → 09-26 ~10:00 | run 1's tasks free their GPUs one by one. At the rates measured 09-25 10:23: all-data seeds and folds 0 / 2 at ~20:00–21:30, fold 3 ~23:30, fold 1 ~03:00 (it restarted, 07 §6). ≈ **90 GPU-h** |

CPU-only work — downloads, installs, DSP processing, ingestion — can start now.

**Rules and verdicts that shape the choices.**

- **DACON #417333 A1.** Post-processing that generates no new voice or music component is
  **REAL**: neural codec encode/decode, enhancement, separation
  ([competition/05](../competition/05-talkboard-qa.md)). By our reading of the same rule, a
  DSP voice changer applied to a real voice (pitch or formant shift) is REAL. DACON has not
  ruled on that case specifically.
- **Licences.** NC and NC-SA licences are allowed (#417212). Every self-generated file
  ships with its model name, revision, seed and prompt (#417333 A6). Commercial TTS APIs and
  our own scraping are out ([data/01](../data/01-rules-check.md)).

**The test set.** 1,200 files of 4–60 s, standardised to 16 kHz, mono and stereo, in mp3, wav
and flac, some of them telephone audio. The competition is Korean voice-phishing.

## 1 · Which gaps are critical

Scored on three things: the loss a gap can cause on the test set, the evidence it exists, and
the cost to fill it before 09-26 12:00.

| rank | gap | evidence | what goes wrong on the test | cost |
|---|---|---|---|---|
| **S1** | **Processed REAL audio (hard negatives)** | DACON defines codec round-trip, enhancement, separation and voice-changer audio as REAL. The corpus has none, while most modern fakes pass through a neural codec (codec-LM TTS) | Any codec- or enhancement-processed real file reads FAKE. File-head false positives cost 0.45 of the score | **low**: processing only, GPU codecs run ≫100× real time |
| **S2** | **Modern Korean fakes, zero-shot and voice-conversion** | ko-synth has 7 families (07 §6), but only 2 are VC and most are 2023-era. Phishing fakes are cloned voices | Newer Korean clones (codec-LM, flow matching) look unlike anything in training | **medium**: autoregressive models run ~2× real time per GPU (below) |
| **S3** | **Conversational / in-the-wild Korean, real and fake alike** | Korean real is Zeroth (read, studio) plus 2.7 h of Common Voice. Korean fakes are cloned from Zeroth prompts, so both sides are read speech | Test calls are conversational and noisy. The model has never seen real Korean in that style, and "unlike Zeroth" may read as fake | **low–medium**: a dataset plus reuse of its speakers as S2 prompts. Blocked on one decision (§4 Q1) |
| S4 | Chinese fakes in the train views | CFAD's 59.4 h of zh fakes sit in PROBE, so zh is dropped from training (07 §6) | Chinese test audio is unmodelled. Probably rare in a Korean test | low: CosyVoice2 is installed |
| S5 | Fake singing voice | SONICS vocals were stripped (sonics-sep), so there is no sung fake voice. CtrSVDD is in S3 but not local | AI songs with fake vocals (a likely test item) are judged by a voice head that never heard singing | low if CtrSVDD's real singing half exists. **UNVERIFIED**: CtrSVDD distributes the fakes, while its bona fide half is rebuilt from source corpora |
| — | partial fakes, modern English TTS, more noise, more fake music families | 03 / 07 | smaller or already covered | deferred to after run 2 |

**The urgent set is S1, S2 and S3.** S4 and S5 go in only if they are cheap (§3.5).

## 2 · Methods

### S1 — processed REAL, and the same processing on FAKE

Every transform is applied to a real subset (label REAL) **and** to an equally sized fake
subset (label unchanged, FAKE). Otherwise "processed" becomes a cue in the other direction —
the same rule 07 D-b used for Demucs.

| family | tool | licence | what it models |
|---|---|---|---|
| `proc-encodec` | EnCodec 24 kHz at 1.5 / 3 / 6 / 12 kbps | MIT (code + weights, **UNVERIFIED**: the weights' licence field is empty on HF) | neural codec round-trip |
| `proc-dac` | DAC 16 / 44 kHz | MIT | neural codec round-trip |
| `proc-xcodec2` / `proc-mimi` | XCodec2 (Llasa's codec), Mimi | **UNVERIFIED** (check the model cards) | the codecs modern TTS generates through |
| `proc-enhance` | DeepFilterNet3 | MIT / Apache | speech enhancement / denoise |
| `proc-dsp` | pitch shift ±1–4 semitones, formant shift, time-stretch 0.9–1.1 (librosa / pyrubberband) | ISC / GPL tool, our code | voice changer / modulation. REAL by #417333 A1 |

Sources:
- **REAL.** ~40 h drawn from pool A across languages, Korean-weighted (Zeroth, Common Voice ko,
  LibriTTS-R, S3's new real).
- **FAKE.** ~40 h drawn from pool B with the same language mix.

One transform per file, family-tagged. The folds treat `proc-*` as a transform, not a
generator: a processed file inherits its SOURCE row's speaker and source atoms, as
realmusic-sep does, so it never straddles a fold with its source.

### S2 — modern Korean clones

The seven families from round 1 stay. The new ones, all Korean-capable and checked on
Hugging Face 2026-09-25:

| family | model | licence | architecture | est. speed / GPU |
|---|---|---|---|---|
| `chatterbox` | ResembleAI/chatterbox (multilingual) | MIT | LLM tokens + flow matching, zero-shot | **UNVERIFIED** ~2–5× RT |
| `maskgct` | amphion/MaskGCT | CC-BY-NC-4.0 | non-autoregressive masked codec model | **UNVERIFIED** ~2× |
| `llasa` | HKUSTAudio/Llasa-1B-Multilingual | CC-BY-NC-4.0 | LLM over XCodec2 tokens (vLLM-servable) | **UNVERIFIED** ~2–5× |
| `fishspeech` | fishaudio/fish-speech-1.5 | CC-BY-NC-SA-4.0 | dual AR codec LM | **UNVERIFIED** ~3× |
| `cosyvoice` | FunAudioLLM/CosyVoice2-0.5B (resume) | Apache-2.0 | installed; 1.65× RT measured | measured |
| `seedvc` | Plachta/Seed-VC | GPL-3.0 code (weights: repo) | zero-shot voice conversion | **UNVERIFIED** ≫ RT |
| `supertonic` | Supertone/supertonic-2 | OpenRAIL | fast Korean TTS by a Korean vendor, preset voices | **UNVERIFIED** fast |

Measured round-1 speeds, in hours of audio per GPU-hour:

| model | speed |
|---|---|
| MMS-VITS | ≈ 330× |
| kNN-VC | ≈ 165× |
| MeloTTS, OpenVoice | ≈ 17× |
| XTTS | ≈ 2.9× |
| CosyVoice2, Bark | ≈ 1.7× |

Autoregressive TTS is therefore the budget item.

- **Seed-VC** converts real Korean utterances (S3's conversational real among them) onto
  other speakers. This is the closest analogue to a phishing voice changer that *generates*
  a new voice, so its output is **FAKE**.
- **Pitch/formant DSP** does not generate a new voice, so it goes to S1, where it is REAL.

### S3 — conversational real Korean

| option | licence | content | status |
|---|---|---|---|
| **Emilia (ko part)** | HF card `amphion/Emilia-Dataset`: CC-BY-4.0 (dataset level). Whether the original `Emilia/` part and `Emilia-YODAS/` carry different terms is **UNVERIFIED**; read both READMEs before download | in-the-wild talk shows / podcasts / video, many speakers, with transcripts | gated (`auto`); an HF token is on this machine. **Decision §4 Q1**. Size of the ko part **UNVERIFIED** |
| FLEURS ko | CC-BY-4.0 | read Wikipedia sentences, ~10 h | open; fallback if Emilia is refused, but it is read speech |

S3 does two jobs. It adds conversational REAL Korean, and it supplies the **prompts and
texts** for S2, so the fakes are conversational too and style cannot separate the labels.

## 3 · The synthesis plan

### 3.1 Targets (hours after QC)

| set | real | fake | GPU-h | when |
|---|---|---|---|---|
| S1 processed | ~40 h over the 5 families (~8 h each) | ~40 h over the same 5 families | ~4 (codecs) + CPU (DSP) | CPU now; GPU codecs in the window |
| S2 Korean clones | — | 6 families × 8 h = ~50 h | ~40 | window |
| S3 conversational ko | 50–80 h | (feeds S2) | 0 | download now |
| S4 Chinese fakes | — | 15 h (CosyVoice2 + Chatterbox zh, CFAD-real / aishell prompts) | ~10 | window, if S2 is on schedule |
| S5 singing | CtrSVDD bona fide (if present) | CtrSVDD fakes, capped at 40 h | 0 | ingest only if its real half exists |

**Budget.** ~55 GPU-h of the ~90, which leaves ~35 GPU-h of slack for slow models and re-runs.

### 3.2 Getting good data fast

1. **Install and smoke-test on CPU now, and on one GPU as soon as a run-1 task ends.** Each
   family must produce 3 files that pass QC (below) before it gets GPUs. Round 1 lost most of
   its time to installs (ko-synth REPORT: torch/torchaudio/onnx traps), so installs go
   first, one venv per family under `/data/project/private/dacon-venvs/synth-<family>`.
2. **Fast families first, slow ones sharded wide.**
   - The window opens with S1's GPU codecs and Seed-VC: minutes each, done early.
   - The AR models get 2–3 GPUs each, via `--shard i/n` (round 1's `synth_common.py` supports it).
   - Llasa is served with vLLM if its smoke test shows the gain.
3. **Resumable by construction.** Per-file metadata appends under a lock (round 1's
   contract), so a cancelled or pre-empted job loses nothing and the ingest can cut at any
   time.
4. **Quality control, applied to every family the same way.**
   - **Intelligibility.** Whisper large-v3 (MIT) Korean CER ≤ 0.35 against the input text.
   - **Duration, level and silence.** Duration 3–20 s, RMS ≥ −45 dBFS, clipped ≤ 0.1 %,
     silence ≤ 50 %.
   - **Reject rate.** Measured per family on its smoke output. A family over 30 % rejects is
     dropped, not tuned.
   - QC runs on the same GPUs between synthesis shards; Whisper large-v3 is ≫ real time on an H200.
5. **Diversity over volume.** Round 1 showed the draw caps each generator domain (DOSS,
   `domain_cap` 500), so **8 h per family is enough**, and a new family is worth more than more
   hours of an old one.
   - **Prompts** spread over ≥ 100 speakers per family.
   - **Texts** come from S3's conversational transcripts, capped at 1–2 sentences, 3–15 s.
6. **Keys for the folds**, as in round 1 (processing/extend.py):
   - A clone's speaker key is `<family>/<prompt speaker>`.
   - A processed file inherits its source row's atoms.
   - Each family is its own `artifact_family`, so the folds rotate them.

### 3.3 Integration

`processing/extend.py` gains readers for the new corpora. It builds strategy-v4 = v3 +
S1–S3 (+ S4/S5), and v3 is not overwritten. Then, in order:

1. folds
2. cache extension
3. `scripts/strategy/audit_fold.py` at n = 40,000, fold 0 and all-data. **I1 must pass on the
   `proc-*` families**: processing may not predict the label.
4. the 100-step smoke run, then run 2 with the first-run configs

### 3.4 Owners (proposed)

| track | what | GPUs in the window |
|---|---|---|
| A | S1 processing: CPU DSP now; GPU codecs + DeepFilterNet in the window | 1 |
| B | S3 download + text/prompt index → feeds C | 0 |
| C | S2 installs + smoke now; synthesis in the window | 6 |
| D | S4 (CosyVoice zh) and S5 (CtrSVDD fetch + check) | 1 |
| main | extend.py readers, strategy-v4, audit, run 2 | — |

### 3.5 Risks

- **AR model speeds are UNVERIFIED.**
  - If Chatterbox, MaskGCT or Llasa run < 1× RT, cut that family to 4 h rather than extend
    the window.
  - Measure each speed in the smoke test and re-plan GPU shares at window open.
- **CtrSVDD's bona fide half may be absent.** Its fakes alone would teach "singing ⇒ fake",
  so S5 is ingested only with a real-singing counterpart.
- **Processed-FAKE symmetry costs hours.** If S1's fake side lags, ship S1's real side only
  after the audit shows `proc-*` does not predict the label. Otherwise hold both.
- **Licences.**
  - NC and NC-SA are allowed. Seed-VC's GPL covers its code, not our output audio.
  - Each family's `licence` column is filled from the model card, as in round 1.

## 4 · Decisions for the owner

1. **Emilia for S3.**
   - Emilia is a published dataset; its HF card says CC-BY-4.0 at dataset level (per-part
     terms **UNVERIFIED**).
   - Its audio is in-the-wild media, and the YODAS half is YouTube-derived. data/01 bars *our
     own* scraping, not licensed datasets built from web audio, but the YODAS half is close
     to that line.
   - Options: (a) Emilia excluding YODAS, (b) both, (c) neither, use FLEURS (read speech).
     Recommendation: (a).
2. **GPUs before run 1 ends (~20:00 → 03:00).**
   - Option (a): wait. The synthesis window opens ~22:00.
   - Option (b): stop the four fold runs now. They have given their mid-run CV reads (07 §6:
     0.949–0.985); stopping frees 4 GPUs ~12 h earlier and loses the end-of-run fold scores.
   - Recommendation: (b) if you want run 2 a day earlier; otherwise (a).
3. **Texts for conversational Korean.**
   - Emilia's own transcripts, or scam-style scripts written by an open-weight LLM
     (e.g. Qwen2.5-7B-Instruct, Apache-2.0) run locally.
   - Recommendation: transcripts first; scripts only if time allows.
