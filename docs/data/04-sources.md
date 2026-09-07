# 04 — Source Inventory (the four pools)

Every row needs a verified redistribution verdict **before download**
([01](01-rules-check.md), [survey 06](../survey/06-datasets.md)). Hours are initial targets, not
commitments.

## POOL A — Real voice (target ~70 h)

| Source | Type | Hours | License (verify) | Why |
|---|---|---|---|---|
| **LibriTTS-R** | spoken EN, transcripts | 25 | CC BY 4.0 | ⭐ transcripts → paired TTS design |
| **Common Voice** (ko, en, +5 langs) | spoken, transcripts | 20 | CC0 | Language + channel diversity |
| **Zeroth-Korean** | spoken KO, transcripts | 5 | ★ CC BY 4.0 | ⭐ Korean coverage; Korean-hosted competition |
| **VCTK** | spoken EN, multi-accent | 5 | CC BY 4.0 | Speaker diversity, clean studio |
| **M-AILABS** | spoken, multi-lang | 5 | verify | Real counterpart to MLAAD |
| **MUSDB18-HQ vocal stems** | **sung** | 3 | CC BY-NC-SA | ⭐ isolated sung vocals for cells 5–8 |
| **Opencpop / M4Singer / ACE-KiSing** | **sung** ZH | 7 | CC-BY-NC 4.0 | Sung-voice coverage |
| *(optional)* ASVspoof19/21 bonafide | spoken | – | ODC-BY / verify | Also gives us channel-degraded reals |

⚠️ Without sung real voice, the model will call every sung vocal fake.

## POOL B — Fake voice (target ~70 h, **maximize generator count**)

| Source | Generators | Hours | License (verify) | Why |
|---|---|---|---|---|
| **MLAAD v9** | **175 TTS models, 54 languages** | 30 | ★ CC-BY-NC 4.0 | ⭐⭐ By far the best generator-diversity asset available |
| **Codecfake** | 7 neural codecs (SoundStream, SpeechTokenizer, FunCodec, EnCodec, AudioDec, AcademicCodec, DAC) | 10 | verify | ⭐ Covers codec-LM TTS, which vocoder-trained detectors are blind to |
| **ASVspoof 2019 LA / 2021 LA+DF** | classic TTS/VC + telephony transmission | 10 | ODC-BY / verify | ⭐ The telephone-channel asset |
| **Self-generated TTS** | 8–12 open models → [05](05-synthesis-plan.md) | 15 | ours | ⭐ Paired-script design; Korean coverage |
| **Self-generated VC / SVC** | RVC, Soft-VITS-SVC variants, SeedVC, kNN-VC | 3 | ours | Conversion ≠ synthesis; distinct artifact family |
| **Self-generated SVS** (sung) | DiffSinger, VISinger2, NNSVS | 2 | ours | Sung fake voice |
| **Self vocoder/codec resynthesis** of Pool A | HiFi-GAN, BigVGAN, Vocos, EnCodec, DAC | – | ours | ⭐ Cheapest family multiplier; ⭐ gives **perfectly matched** real/fake pairs |
| ⚠️ **CtrSVDD** | 14 SVS+SVC methods, 260 h fake singing @16 kHz | (up to 20) | ★ **CC BY-NC-ND** | Best sung-fake asset, but **ND blocks it pending legal review** ([V2](../survey/10-open-questions.md)) |

## POOL C — Real instrumental (target ~45 h)

| Source | Hours | License (verify) | Why |
|---|---|---|---|
| **MUSDB18-HQ accompaniment stems** | 3 | CC BY-NC-SA | ⭐ Vocal-free by construction — the cleanest instrumental-only source |
| **MTG-Jamendo** (instrumental tag) | 25 | Apache-2.0 dataset / per-track CC | ⭐ Scale + **genre tags usable as TTM prompts** |
| **FMA** | 15 | per-track CC | Genre breadth; used as OOD real music in AT-ADD |
| *(optional)* MedleyDB | 2 | CC BY-NC-SA | More stems |

⚠️ "Instrumental" must genuinely contain no vocals. Jamendo's instrumental tag is uploader-
supplied — **verify with a vocal-detection pass** (e.g. separation energy ratio) before trusting.

## POOL D — Fake instrumental (target ~45 h, all self-generated)

| Generator | License | Notes |
|---|---|---|
| **ACE-Step** (3.5B) | ☆ Apache 2.0 | ⭐ Permissive, fast, vocals *and* instrumentals |
| **MusicGen** (+ stereo, melody) | verify | AR over EnCodec tokens; melody conditioning |
| **Stable Audio Open 1.5** | verify (Stability community) | Latent diffusion + T5 |
| **AudioLDM 2 / MusicLDM / Mustango** | verify | FakeMusicCaps + AT-ADD generator families |
| **YuE** (7B) | verify | Full songs; needs ≥24 GB GPU |
| **DiffRhythm 2** | verify | Block flow matching |

Instrumental-only generation: prompt for "instrumental", "no vocals", or take the instrumental
stem of a generated song. Verify vocal-free with the same detection pass as Pool C.

## POOL E — Non-musical sound (target ~10 h)

MUSAN (noise + speech partitions), RIRS_NOISES, and an environmental set (ESC-50 / FSD50K,
verify license). Uses: background-noise augmentation, cell 9 negatives, applause/crowd
false-positive inoculation for the music-presence head.

## Aggregate budget & shippability

At 16 kHz mono 16-bit PCM: **115.2 MB/hour**. FLAC ≈ 50–60% of that.

| | Hours | WAV | FLAC |
|---|---|---|---|
| Pools A+B+C+D+E | ~240 | ~27.6 GB | ~14–17 GB |

Deliverable to DACON is roughly **15 GB of FLAC** — comfortable for Google Drive, which DACON
explicitly sanctioned (#417198). ⚠️ We ship the **pools**, not the composed/augmented output:
DACON confirmed (#417280) that augmentation intermediates need not be submitted if we provide
originals + code + config + seed. That is why [06](06-augmentation-spec.md) makes composition
**on-the-fly and seeded** rather than pre-rendered.

## Priority order (if time runs short)

1. MLAAD + LibriTTS-R + Zeroth-Korean → Pool A/B minimum viable
2. MUSDB18-HQ (all stems) → unlocks cells 5–8 with a single 4 GB download
3. ACE-Step self-generation → Pool D
4. MTG-Jamendo instrumental → Pool C scale
5. Codecfake + ASVspoof21 LA → codec + telephone families
6. Everything else
