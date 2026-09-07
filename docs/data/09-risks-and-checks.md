# 09 — Self-Review: Where This Plan Could Be Wrong

Written adversarially against [02](02-label-taxonomy.md)–[08](08-build-plan.md). Each risk has a
**pre-committed check** so we detect it early rather than at the deadline.

## Correctness risks in the plan itself

| # | Risk | Why it's plausible | Check |
|---|---|---|---|
| R1 | 🔴 **"Artificially mixed → fake" shortcut** | Cells 6/7 can *only* be our own mixes; if cell 5 is all natural songs, mixing perfectly predicts the label | Metadata-only logistic regression must give AUC < 0.6 ([07](07-eda-plan.md#tier-s)). Also: hold out a validation slice of **natural** AI songs (cell 8) and natural real songs (cell 5) and confirm the model still separates them |
| R2 | 🔴 **Corpus-identity shortcut** | Real voice from LibriTTS, fake voice from MLAAD — different recording chains entirely. Model learns "LibriTTS-ness" | T3 matched pairs (same utterance, vocoder-resynthesized) share a chain by construction — measure EER **restricted to T3 pairs**. If pooled EER ≪ T3-pair EER, we have corpus leakage |
| R3 | **Test chain mis-specified** | Our whole normalization rests on 3 dummy files, which may not represent all 1,200 | Randomize container/channel/bandwidth *widely* rather than pinning to exactly what the dummies show; treat the spec as a centre, not a boundary |
| R4 | **Held-out generators aren't actually held out** | Many open TTS models share vocoders/codecs (HiFi-GAN, EnCodec appear everywhere) | Hold out by **artifact family** (vocoder / codec / diffusion / AR-LM), not just by model name |
| R5 | **Sung-voice gap** | If Pool A sung hours are thin, every song's vocal reads as fake | Track `VOICE_FAKE` EER on the sung subset separately; it must not be ≫ spoken subset |
| R6 | **Instrumental pool contaminated with vocals** | Jamendo's "instrumental" tag is uploader-supplied | Run a separation/vocal-energy pass over Pool C; quarantine anything with vocal energy above threshold |
| R7 | **Cells 6/7 under-represented** | They're the expensive ones and easy to skimp on | Fixed floor: ≥15% of composed samples in each of cells 6 and 7 |
| R8 | **Pitch-shift label noise** ([06](06-augmentation-spec.md)) | Phase-vocoder artifacts resemble generative artifacts | Ablate: train with and without pitch shift, compare on the held-out probe |
| R9 | **On-the-fly composition isn't reproducible** | DACON requires reproduction from code + config + seed | CI test: same seed → byte-identical rendered sample. Verify before Phase 2 ends |
| R10 | **License verdict wrong on a large source** | Verdicts in [04](04-sources.md) are unverified | Phase 0 audit is blocking; record `verified_by`/`verified_date` in the ledger |

## Where I am least confident

| Item | Uncertainty | Mitigation |
|---|---|---|
| 🔴 **What survives 16 kHz for AI-music detection** | [G1](../survey/10-open-questions.md) — no published work evaluates this. Our music-head plan rests on an untested premise | Make it Phase 1's first experiment: train a small music-fake model on 48 kHz vs 16 kHz data and measure the gap. Cheap, and it either confirms the strategy or forces a rethink early |
| **Whether MUSDB18-HQ (150 tracks) is enough stem diversity** | Small corpus; genre-narrow (mostly Western pop/rock) | Supplement by **separating** Jamendo/FMA tracks with HT-Demucs to manufacture more pseudo-stems. ⚠️ separation artifacts then enter both real and fake sides — apply label-independently ([06](06-augmentation-spec.md)) |
| **CtrSVDD's ND license** | [V2](../survey/10-open-questions.md). It's the best sung-fake asset | Plan assumes it is **unavailable**; self-generated SVS covers the gap. Treat clearance as upside |
| **Test-set composition** | Unknown ratios across cells | EER/AUC are prevalence-insensitive, so this matters less than it feels ([02](02-label-taxonomy.md)) |
| **Whether commercial-generator coverage matters** | We can't include Suno/Udio | 16 kHz plausibly strips their distinctiveness ([01](01-rules-check.md)). Unfalsifiable until we probe the LB |

## Things this plan deliberately does not do

- **No leaderboard-driven data selection beyond Phase 3.** With Private = Public, 3 submissions/day
  over 24 days is enough to overfit 1,200 samples if we let per-submission feedback steer the corpus.
- **No hand-engineered features from EDA.** EDA decides what *data* to add ([07](07-eda-plan.md)).
- **No pre-rendered augmented corpus.** On-the-fly + seeds, for both training variety and
  shipping volume.

## Open items requiring your decision

1. **Ask DACON** the four questions in [survey 10](../survey/10-open-questions.md) — especially
   whether vocoder/codec resynthesis and AI denoising count as FAKE. This changes what Pool B and
   the REAL-processed slice mean. Turnaround is a few days, so post early.
2. **Korean slice size** — how much to invest given unknown test-set language composition.
3. **CtrSVDD legal call** — accept the ND risk, or self-generate only.
