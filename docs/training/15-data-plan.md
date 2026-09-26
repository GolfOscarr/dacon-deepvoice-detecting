# 15 · Data plan for strategy-v5: new generators, singing, channel augmentation, mixture

*2026-09-27 ~00:30 KST. Owner decision: **data first**, then the 1B run on the new data. This is
the plan for approval: nothing here is generated until the owner says go. Every hour figure for
the current corpus was measured on `strategy-v4/manifest.parquet`. **UNVERIFIED** marks licence
and language-support claims that must be checked, per model, before any GPU time is spent.*

## 0 · Brief

Run 2 showed that the data moves the leaderboard (0.792 → 0.811, and the T5 control says the gain
came from the data). What still fails on VAL: **held-out modern cloners** (maskgct missed
.41 en / .27 ko, seedvc .22), **real in-the-wild Korean called fake** (.15–.18), and the
**mixed cells** (fold 3: cells 4–6 EER .10–.14). Five additions, ranked by expected leaderboard
value per GPU-hour:

| # | addition | targets | GPU cost (ESTIMATE) | new hours |
|---|---|---|---|---|
| **N1** | **new voice-cloning families**: Korean 4, English 4 (voice fakes) | the unseen-generator gap (voice + file EER, 0.7 of the score weight) | ~45 GPU-h | ~50 h fake |
| **N2** | **singing voice**: vocal stems from real and from AI songs, plus AI songs with Korean lyrics | the song/mixed cells, a test type with no singing component in training today | ~8 GPU-h | ~25 h real + ~25 h fake |
| **N3** | **more real in-the-wild Korean** (YODAS-ko; 8 of 208 tars used) | the Emilia-ko false alarms | CPU only | +40 h real |
| **N4** | **channel augmentation**: phone codecs, packet loss, compression | "실제 통신환경" (the competition background names it) | CPU only | — (on the fly) |
| **N5** | more hours of maskgct + seedvc | the two worst held-out families | ~12 GPU-h | +16 h fake |

Mixture changes (§5):
- **N1/N2/N5 enter as new domains** at weights that give each modern family an equal share.
- **The Korean share goes 0.55 → 0.60.**
- **WaveFake stays at ×0.25.** R1 measured that removing it costs 0.001–0.003.
- **The one-class loss is dropped.** R1 found no gain.

**Budget:** 6 GPUs from ~01:00 to ~13:00 Sunday (72 GPU-h), with 2 GPUs left for the 1B wiring and
probe.
- **strategy-v5** is built by ~15:00.
- **A quick 300M T7 fine-tune on v5** (1 GPU, all data) is submitted Sunday night. It is the
  leaderboard test of the data before 1B finishes.
- **The 1B main run** trains on v5 from ~15:00 Sunday.

## 1 · Current inventory (strategy-v4, hours)

| pool | what | hours | notes |
|---|---|---|---|
| A real voice | Emilia-ko 88.1 (Emilia 40, YODAS 40, FLEURS 8.4), Emilia-en 80.0, MUSAN speech 60.4, Zeroth-ko 52.9, LibriTTS-R 49.6, LJSpeech 23.6, proc-real 30.8, CommonVoice-ko 2.7, CFAD-real zh 55.2 (drawn at 0) | 443 | Korean ≈ 164 h, English ≈ 224 h |
| B fake voice | WaveFake en 181.5 (×0.25 in the draw), ko-synth 39.6 (mms, openvoice, knnvc, melo, xtts, cosyvoice, bark), ko-synth2 40.8, en-synth2 36.1 (5 modern families each), proc-fake 30.8, MLAAD en 8.5, CFAD zh 59.4 and WaveFake ja 13.5 (drawn at 0) | 436 | only **10 modern cloner families** (5 ko + 5 en, the same 5 models) |
| C real music | FMA 57.8, realmusic-sep 41.8, MUSAN music 38.8 | 138 | instrumental components |
| D fake music | FakeMusicCaps 74.5 (MusicGen, AudioLDM2, MusicLDM, Mustango, Stable Audio Open), SONICS-sep 62.9 (Suno chirp v2/v3/v3.5, Udio; **instrumental stems only**) | 137 | |
| whole-file songs | SONICS AI songs 1,970.6 (cell 8); real songs FMA 8.6 + MUSAN 3.8 (cell 5) | — | **real songs: 12 h** |
| E noise | CompSpoof env 14.6, MUSAN noise 6.2, RIR noise 0.8 | 22 | |

**Gaps this shows:**
- **G1:** 5 distinct modern cloner *models*, each in two languages.
- **G2:** **no singing voice as a voice component, real or fake.** The separation kept the Suno/Udio
  instrumentals and discarded the vocals. Real songs are 12 h against 1,971 h of AI songs.
- **G3:** real Korean is read or clean-ish speech plus 80 h of Emilia/YODAS.
- **G4:** the telephone leg is μ-law / A-law / plain 8 kHz only, on 20 % of files.

## 2 · N1 — new voice-cloning families (the largest expected gain)

**Why:**
- Unseen-generator transfer is the measured weak point.
- A hidden test built by a forensic institute will use generators we do not have.
- Diversity of *generator families* is what transfers; hours of the same five do not.

**Rules for every family:**
- The licence is checked and recorded before synthesis: model weights licence, output-use terms,
  and no commercial API (rules check: API ToS ✗).
- The synthesis pipeline is `scripts/synth2/ko_*` (with `SYNTH2_LANG=en`):
  - **prompts:** Emilia speakers ~75 %, the rest from Zeroth / LibriTTS-R train_val speakers only;
  - **transcripts:** dataset transcripts;
  - **levelling:** prompts are levelled (the CosyVoice clipping fix);
  - **QC:** Whisper;
  - the same metadata schema.
- **Targets:** 8 h kept per Korean family and 5 h per English family.
- **Smoke test first:** 10 files, keep rate ≥ 80 %, or the family is dropped (as Llasa was).

**Candidates** (pick 4 per language after the licence and language checks; **all UNVERIFIED**):

| family | kind | Korean? | licence (to verify) | why |
|---|---|---|---|---|
| GPT-SoVITS v2+ | few-shot TTS + VC | yes (claimed) | MIT | very common in real Korean misuse |
| RVC (retrieval VC) | voice conversion | language-agnostic | MIT | the most common real-world VC; needs a small per-target index (minutes per speaker) |
| F5-TTS | flow-matching TTS | community Korean fine-tunes | code MIT, base weights CC-BY-NC | a distinct architecture (DiT flow) |
| CosyVoice 3 / Fun-CosyVoice | LLM + flow | to verify | Apache-2.0 | newer than our CosyVoice2 |
| Orpheus multilingual | LLM TTS (SNAC codec) | Korean release (claimed) | Apache-2.0 | a new codec-LM family |
| Higgs Audio v2 | LLM TTS | to verify | to verify | already a tiny slice in MLAAD (0.16 h) |
| IndexTTS2 | AR TTS | no (zh/en) | to verify | English only |
| Spark-TTS | BiCodec LM | no (zh/en) | CC-BY-NC-SA (to verify) | English only |
| Zonos v0.1 | hybrid | no | Apache-2.0 | English only |
| VibeVoice / Dia | long-form / dialogue | no | MIT / Apache-2.0 | English only; conversational speech |

**Placement in folds:**
- Pinned folds keep every v4 row where it is.
- Each new family is **one atom assigned to one fold**, extending `FAMILY_FOLD` in
  `scripts/build_folds_pinned.py`, so that each fold holds out at least one new family. That keeps
  the S1 (held-out family) reading available for the 1B fold check.
- Nothing new goes into PROBE.

## 3 · N2 — singing voice (cheap, and a whole test type)

**Test fact** (competition overview): a song with vocals is *mixed*, and the vocals are the
**voice** component.
- A real song → voice REAL. A Suno song → voice FAKE.
- Our voice head has never seen singing on either side, so on songs it is extrapolating.

- **N2a · Vocal stems from AI songs → fake singing voice (pool B).**
  - Re-run the source separation (htdemucs, as in `_music_sep_work`) on the SONICS songs we
    already hold, and keep the **vocal** stem this time.
  - Target: ~20 h, balanced over chirp v2/v3/v3.5 and Udio 30 s/120 s.
  - GPU: ~2–3 GPU-h.
- **N2b · Vocal stems from real songs → real singing voice (pool A).**
  - Same separation on the real songs with vocals: FMA (CC; fma_small plus the parts of
    fma_medium that have vocals, screened by speech/vocal activity) and MUSAN music with vocals.
  - Target: ~20 h. Separation is post-processing, so under A1 the stem stays REAL, and **both
    sides are separated**, so separation is not a shortcut.
  - **Risk:** real songs with vocals under a clean licence are scarce, and Korean ones scarcer. Jamendo
    (CC) is the fallback source (UNVERIFIED availability).
- **N2c · AI songs with Korean and English lyrics → cell 8 whole files + vocal stems.**
  - SONICS is almost all English. Generate ~10 h with an open song generator: ACE-Step (Apache-2.0,
    multilingual lyrics incl. Korean, UNVERIFIED) or YuE (Apache-2.0, ko listed, UNVERIFIED;
    slower).
  - Lyrics: from public-domain / CC text or generated; recorded per file.
  - GPU: ~5 GPU-h.
- **Balance:** singing is ~10 % of voice components on each side, the same share on both sides, and
  checked by the audits (I1c).

## 4 · N3, N4, N5

- **N3 · Real in-the-wild Korean (CPU).**
  - Sample 40 more hours of YODAS-ko from the 200 unused tars, spread evenly by video.
  - Same screens as S3 (speech ratio, loudness, duration), speaker key per video.
  - Download and index on CPU; ~3–4 h wall. No GPU.
- **N4 · Channel augmentation (CPU, on the fly; `processing/render.py` normalize menu).**
  - **New telephone legs**, each with its encoder delay measured and cancelled, the rule
    `render.CODEC_CONTAINERS` states:
    - AMR-NB (4.75–12.2 kbps, 8 kHz);
    - AMR-WB (6.6–23.85 kbps);
    - Opus (8–24 kbps, VoIP mode);
    - GSM-FR.
  - **Packet loss:** drop 20 ms frames at 1–5 % with zero-fill or repeat concealment, p 0.1.
  - **Dynamic-range compression / loudness normalisation** (a common post-process; REAL under A1),
    p 0.2.
  - **Telephone share** 0.20 → 0.30; low-bitrate MP3 (32/48 kbps) added to the container menu.
  - **RIR** p 0.2 → 0.3.
  - Every leg is applied **identically to both labels** (the draw decides by spec key, not label),
    and the audits must pass on folds 0 and 1.
  - **Cost:** codec availability in the venv's ffmpeg (UNVERIFIED for AMR-WB and Opus
    encoders), plus ~2–3 h of code with delay tests.
- **N5 · maskgct + seedvc, +4 h each per language** (16 h), with new prompt speakers. Modest,
  because diversity (N1) matters more than hours.

## 5 · Mixture (the draw, `configs/processing_run5.yaml`)

| knob | v4 (run 2) | v5 | why |
|---|---|---|---|
| `lang_shares` | ko .55 / en .45 | **ko .60 / en .40** | Korean forensic context; ko false alarms |
| `domain_weights` new families | — | each new family weighted so that **every modern cloner family has an equal share** of the voice-fake draw (about 1/20 each with 10 new + 10 existing) | no family > ~8 % of voice fakes |
| WaveFake / MLAAD-LJ | ×0.25 | ×0.25 | R1 (c): removing it costs 0.001–0.003 |
| ko-synth (older TTS) | ×1 | ×1 | run 3a: ×2/×3 did not fix melo/mms |
| singing (N2a/b) | — | ~10 % of voice components on each side | new test type, symmetric |
| AI songs (cell 8 whole files) | SONICS only | SONICS + N2c Korean/English lyrics | language match |
| real songs (cell 5 whole files) | 12 h | + real songs from N2b (whole) | the cell-5 natural fraction must match the cell-8 policy (`f8`, data/02 §composition) |
| `cell_mix` | unchanged | unchanged | no evidence to move it |
| augments / normalize menu | §run 2 | + N4 | |

Tuning procedure (as for v4):
1. `scripts/strategy/draw_balance.py` to tune the domain weights until repetition share, processed
   share and language share match per label.
2. `audit_fold.py` on folds 0 and 1 (I1c < 0.60, I3, I5).
3. The all-data audit.

## 6 · Schedule (KST) and who does what

| when | GPUs | work |
|---|---|---|
| Sun 00:30–01:00 | — | owner approves; licence and language check per N1 candidate (a subagent, web + model cards) |
| 01:00–13:00 | 6 | N1 Korean (4 families, 2–3 GPUs), N1 English (4 families, 1–2 GPUs), N2 separation + ACE-Step (1 GPU), N5 (fills gaps) |
| 01:00–05:00 | CPU | N3 YODAS-ko; N4 augment code + delay tests |
| alongside | 2 | 1B wiring + memory probe (subagent already running) |
| 13:00–15:00 | CPU | strategy-v5: extend manifests, pinned folds, draw tuning, audits (the `build_strategy_v4.sh` pattern) |
| 15:00 | 1 + 7 | **v5 test**: T7 all-data fine-tune on v5, 300M, 1 GPU, ~5 h → Sunday-night submission (with max3). **1B main** on 7 GPUs, DDP, all data, v5 |
| Mon | — | 1B → package (max3) → server-mirror timing → submit; Tue AM final pick |

**Staffing:** four synthesis agents, as for Korean S2/S3 (one each for Korean N1, English N1, N2
and N3/N4). The lead integrates v5 and runs the audits. Every job is `-J eval-hyeonseop`, with
dispatchers that stop at 13:00.

## 7 · Decision rules and risks

- **Keep a family only if** its smoke keep rate is ≥ 80 %, Whisper QC reject is ≤ 15 %, and its
  licence is clean. Otherwise drop it; a dropped family costs its GPU slot and nothing else.
- **v5 is accepted only if** its audits pass. The Sunday-night LB test (300M on v5 vs T7's 0.81144)
  is the judge of the data as a whole. PROBE over-reads by about 3×.
- **Risks:**
  - Licences: some candidates are non-commercial. This competition is non-commercial research,
    but each must be recorded in the licence log.
  - Korean support claims are unverified.
  - N2b real vocal songs may be scarce.
  - N4 codecs may be missing from ffmpeg.
  - 12 h of generation is tight: families run in parallel, the dispatchers stop at 13:00, and we
    use what is kept by then.
- **Not in scope:**
  - Chinese and Japanese (owner rule).
  - Commercial API voices (ToS).
  - Copy-synthesis labelled fake (A1).

## 8 · Open question that changes the plan

**The per-head leaderboard split** (the `run2-T7-diagA/B.zip` submissions, in S3):
- **If music is most of the gap:** shift GPU from N1 to music generators (ACE-Step instrumental,
  newer MusicGen / Stable Audio) and weight D higher.
- **If voice/file is most of the gap:** this plan as written.

The submissions cost 2 of today's 3 slots.
