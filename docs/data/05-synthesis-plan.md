# 05 — Synthesis Method Catalog (tiered)

Self-generation is the **only** lever that expands the pool without touching the license gate:
we own the files, we know the labels exactly, and we control every confound.
Tier legend in [07](07-eda-plan.md#tier-legend-used-in-05-06-07).

**Governing principle** — optimize for **number of distinct generator families**, not hours per
family. ASVspoof 5 baselines sat above **29% EER** against unseen generators while top systems
reached ~4–5%; the difference was data and augmentation breadth, not architecture
([survey/01](../survey/01-sota-speech.md)).

---

## Tier S

| # | Method | What it buys / why | Output artifact |
|---|---|---|---|
| **S-S1** | 🔴 **T3 resynthesis twins** — run every Pool A/C file through vocoders (HiFi-GAN, BigVGAN, Vocos, MultiBandMelGAN) and neural codecs (EnCodec, DAC, SpeechTokenizer) | **Perfectly matched real/fake pairs**: same speaker, words, prosody, room, channel — the *only* difference is the synthesis artifact. Nearly free, and it is also the leak detector for [09 R2](09-risks-and-checks.md). Independently validated by the SETI on/off-target cadence design ([kaggle/03](../kaggle/03-weak-signal-anomaly.md)) | Paired corpus with `pair_id`; a **T3-restricted EER** metric reported alongside pooled EER |
| **S-S2** | **Generator-family breadth** — ≥20 fake-voice families, ≥5 fake-music families, **assigned to train/val/probe before generation** | Cross-generator generalization is the binding constraint everywhere in the literature. Assignment must precede generation or the held-out split is fiction | `generator_registry.csv`: family, version, artifact class, split assignment |
| **S-S3** | 🔴 **The 4-way voice×music grid** — cells 6 (real voice + fake music) and 7 (fake voice + real music) built by mixing stems | These two cells are **why the competition has two separate fake heads**. Without them the heads collapse into one entangled "fakeness" axis. Unobtainable by any other means ([02](02-label-taxonomy.md)) | Composed cells 5–8, with cell 5 containing artificial mixes at the same rate (anti-shortcut) |
| **S-S4** | **REAL-processed slice** — Pool A/C through neural denoise/enhancement, dereverb, loudness normalization, EQ, DRC, lossy round-trips — **labeled REAL** | The rules make post-processing that does not regenerate a component **REAL**. Modern denoisers are themselves generative and leave similar traces. Without this slice we systematically false-positive on cleaned-up real audio — a trap the test set very likely contains | ~10% of the REAL class, tagged `processing_type` for per-type error analysis |

---

## Tier A

| # | Method | What it buys / why | Output artifact |
|---|---|---|---|
| **S-A1** | **T2 speaker + text matched cloning** — zero-shot clone the *same* speaker saying the *same* sentence | Removes content, speaker, language and accent as confounds. One tier weaker than T3 but covers real TTS pipelines rather than pure resynthesis | Paired corpus, `pair_id` shared with the T3 twin of the same utterance |
| **S-A2** | 🔴 **Generator conditioning on our own domain** — fine-tune / prompt-condition generators on our real corpus before generating | ★ `[LLM-Detect-AI 2024, 1st]` CLM-tuned 9 LLM families on the in-domain corpus so fakes matched the *domain*, not just the task. That was the winning move in the closest structural analogue to our competition | `conditioning_recipe.md` per family; conditioned checkpoints logged in the ledger |
| **S-A3** | 🔴 **Adversarial / hard-negative generation** — highest-quality settings, artifact-suppressing post-processing, denoised fakes, low-CFG "natural" samples | ★ `[LLM-Detect-AI 2024, 1st]` used instruction-tuned LLMs to generate *adversarial* essays. Our easy fakes teach little; the decision boundary lives at the hard end | `hard_fakes/` slice, tagged and over-weighted in Phase 3 |
| **S-A4** | **Tag/caption-matched music generation** — use MTG-Jamendo genre/instrument/mood tags as TTM prompts | Removes genre and instrumentation as confounds — the FakeMusicCaps construction. Otherwise "fake music" collapses into "the genres our generators like" | Prompt manifest linking each generated track to the real track whose tags produced it |
| **S-A5** | **Sampling-parameter sweep** — temperature, CFG/guidance, seed, decoding strategy, speaker reference | ★ `[G2Net 2021, 3rd]` varied the physical parameters that *shape artifact morphology*, not just sample count — worth 2–8 bps. Default settings produce a narrow artifact distribution | Per-file sampling params in the ledger; coverage matrix per family |
| **S-A6** | **Korean slice** — CosyVoice 3 / VibeVoice / MeloTTS / XTTS-v2 over Zeroth-Korean references | Korean-hosted competition; the test set plausibly contains Korean, possibly telephone-channel Korean. Treat as required, not optional | ≥5 h real + ≥5 h fake Korean, incl. a telephone-simulated subset |
| **S-A7** | **Codec-LM TTS coverage** — the Codecfake families (SoundStream, SpeechTokenizer, FunCodec, EnCodec, AudioDec, AcademicCodec, DAC) | Modern TTS generates **directly from discrete codecs, skipping the vocoder**. Codec-trained detectors show a **41.4% relative EER reduction** on codec fakes; vocoder-trained ones are structurally blind ([survey/01](../survey/01-sota-speech.md)) | Codec-resynthesis slice, tagged by codec family |

---

## Tier B

| # | Method | What it buys / why | Output artifact |
|---|---|---|---|
| **S-B1** | **SVS / SVC for sung fake voice** — DiffSinger, VISinger2, NNSVS; RVC, Soft-VITS-SVC variants, SeedVC | Competition labels vocals as **voice**, so every song's vocal is in scope for `VOICE_FAKE`. Spoken-only TTS coverage leaves this blind. Hedges the CtrSVDD ND-license risk ([survey V2](../survey/10-open-questions.md)) | Sung-fake slice; ≥2 h, ≥5 methods |
| **S-B2** | **Pseudo-stem manufacture** — separate Jamendo/FMA with HT-Demucs to expand the stem pool beyond MUSDB18-HQ's 150 tracks | MUSDB18-HQ is small and genre-narrow (Western pop/rock). ⚠️ Separation artifacts then enter the corpus — **must be applied label-independently** ([06](06-augmentation-spec.md#the-governing-rule)) | Pseudo-stem pool tagged `stem_origin=separated`; ablation vs true-stem-only training |
| **S-B3** | **Low-quality generation** — small models, low bitrates, short context, older architectures | Covers the "bad fake" end of the distribution. Test data from real-world voice phishing is unlikely to be uniformly state-of-the-art | `low_quality_fakes/` slice |
| **S-B4** | **Cross-lingual breadth** beyond ko/en — MLAAD already spans 54 languages | Cheap generalization; the artifact should be language-invariant, and proving that is report material | Language coverage table |

---

## Tier C

| # | Method | Note |
|---|---|---|
| **S-C1** | Anti-forensic / laundering attacks on our own fakes | The SAFE challenge tests "audio laundering" ([kaggle/01](../kaggle/01-ai-content-detection.md)). Realistic but probably beyond a 24-day budget |
| **S-C2** | Full-song generation (YuE, needs ≥24 GB) | ACE-Step covers the same cell more cheaply |
| **S-C3** | Environmental-sound generation (AudioGen, AudioLDM) | Only useful as cell-9 negatives; MUSAN is cheaper |

---

## Tier X — rejected

| Method | Why not |
|---|---|
| **Suno / Udio / ElevenLabs APIs** | ToS typically forbids using outputs to build detection or competing models, and we could not ship the files ([01](01-rules-check.md)). 16 kHz strips their distinctive fingerprints anyway ([survey/02](../survey/02-sota-music.md)) |
| **Crawled "AI-generated" web audio** | Blocked by the rules, and it cannot supply component-level voice/music fake labels even in principle ([01](01-rules-check.md)) |
| **Pseudo-labeling DACON's evaluation set** | Explicitly prohibited by rule 2.3 — the single most repeated Kaggle audio technique is illegal here ([kaggle/README](../kaggle/README.md)) |

---

## Using the pairs in training

Not DPO — that aligns *generative* policies to preferences. We train a discriminator and our
metric is pure ranking, so:

| Technique | Why |
|---|---|
| **Pairwise margin-ranking loss** on (real, fake) twins | Directly optimizes the ordering EER measures |
| **Supervised contrastive** with pair-aware batching | Independently arrived at by ★ `[LLM-Detect-AI 2024, 1st]`, who trained a DeBERTa **ranking** model *and* a supervised-contrastive embedding model |
| **Twins in the same batch** | Forces the network onto the only difference — the artifact |
| **Short rank-loss fine-tune at low LR at the end** | ★ `[G2Net 2021, 3rd]`, ~1 bps, metric-aligned and nearly free |

Log `pair_id` per sample from day one so pair-aware batching stays possible even if we start with
plain BCE.

## Recording requirements (mandatory, per DACON #417198)

Every generated file needs: `model_name · version/revision · checkpoint hash · seed · prompt/text ·
speaker_ref_id · sampling params · post-processing params · output sample rate · timestamp`.
DACON requires this verbatim alongside the files. Emit it at generation time — retrofitting is
impossible.
