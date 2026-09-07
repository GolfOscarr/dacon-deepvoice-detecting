# 07 — Generators for Building Our Own Fake Data

We must build most of the training set ourselves ([06](06-datasets.md)). Data we generate is
data we can ship. **Generator diversity is the single strongest predictor of generalization.**

⚠️ Licenses unverified. Two separate checks needed per model:
(a) may we use the weights? (b) **may we redistribute the audio it produces to DACON?**
Commercial APIs (ElevenLabs, Suno, Udio) typically restrict output use for building detection
or competing models — **check ToS before using any hosted service.**

## Speech TTS (→ fake voice)

| Model | Notes |
|---|---|
| **VibeVoice** (Microsoft Research) | ☆ **MIT**. Long-form, expressive, multi-speaker; up to ~90 min, 4 speakers. Permissive → low legal risk. |
| **Kokoro** | 82M params, 54 preset voices / 8 languages, most-downloaded open TTS. No cloning, no emotion control. Cheap to run at scale. Used in AT-ADD. |
| **CosyVoice 3** (Alibaba FunAudioLLM) | Multilingual zero-shot cloning, 9 languages + 18 Chinese dialects, ~150 ms streaming. CosyVoice used in AT-ADD. |
| **F5-TTS** | Flow-matching; strong long-input prosody. |
| **Higgs Audio V3** | Zero-shot multi-speaker dialogue, prosody adaptation, humming, **simultaneous speech + background music** ← directly useful for mixed-audio generation. |
| **IndexTTS 2** | Named in AT-ADD (Index-TTS). |
| **XTTS-v2** | ⚠️ Coqui Public Model License — non-commercial; verify redistribution of outputs. |
| **MeloTTS, Parler-TTS, Tortoise-TTS, Llasa 1B** | All named in AT-ADD's generator list. |
| **Classic/legacy** (ProDiff, PortaSpeech, DiffSpeech, FastSpeech2, FastDiff, GradTTS, FastPitch, Tacotron2, Glow-TTS, WaveNet, WaveGlow) | Covered by ASVspoof/MLAAD; cheap family diversity |

**Korean coverage matters** — the competition is Korean-hosted and the test set likely contains
Korean speech. CosyVoice 3, VibeVoice, MeloTTS and XTTS-v2 have Korean support; Zeroth-Korean
([06](06-datasets.md)) provides real Korean reference audio for cloning.

## Voice conversion / singing (→ fake voice, sung and spoken)

| Model | Notes |
|---|---|
| **Soft-VITS-SVC** (+ WavLM / ContentVec / MR-HuBERT / WavLabLM / Chinese HuBERT / SF-HiFiGAN variants) | 6 variants used in CtrSVDD — cheap diversity from one codebase |
| **RVC** | The most widely deployed SVC in the wild → high ecological validity |
| **SeedVC, NeuCoSVC, NU-SVC** | Named in AT-ADD / CtrSVDD |
| **StarGANv2-VC, kNN-VC** | Classic VC families |
| **SVS**: XiaoiceSing, VISinger / VISinger2, NNSVS, DiffSinger, Naive RNN, ACESinger | CtrSVDD's 7 SVS methods |

## Vocoders & neural codecs (→ resynthesis fakes, cheap and high-value)

Resynthesizing **real** audio through a vocoder or codec produces a FAKE-labeled sample without
any TTS pipeline. This is the cheapest way to multiply generator families.

- **Vocoders**: HiFi-GAN, BigVGAN, Vocos, MultiBandMelGAN, WaveGlow
- **Neural codecs** (Codecfake's seven): SoundStream, SpeechTokenizer, FunCodec, **EnCodec**,
  AudioDec, AcademicCodec, **DAC**

⚠️ **This is also where the REAL/FAKE boundary gets dangerous.** The rules say post-processing
that does not regenerate the component (denoise, enhance, volume) stays **REAL**, while
generation is FAKE. Codec *resynthesis* is generation → FAKE. But modern **neural denoisers**
are themselves generative and will leave similar traces on REAL audio. Both sides of that line
must be represented in training or we produce systematic false positives. See [10](10-open-questions.md).

## Text-to-music / song generation (→ fake music)

| Model | Notes |
|---|---|
| **ACE-Step** (3.5B) | ☆ **Apache 2.0** — the permissive pick. Vocals **and** instrumentals, up to ~4 min, fast on consumer GPUs. [ACE-Step 1.5](https://arxiv.org/pdf/2602.00744) · [paper](https://arxiv.org/pdf/2506.00045) |
| **YuE** (7B) | Full-length songs with coherent lyrics + melody; closest open analogue to Suno. Needs 24 GB+ GPU. |
| **Stable Audio Open 1.5** | Latent diffusion + T5 conditioning; longer high-quality audio. Verify license (Stability community license). |
| **MusicGen (+ stereo, melody)** | Single-stage AR transformer over EnCodec tokens; melody conditioning. Used in FakeMusicCaps + AT-ADD. |
| **AudioLDM 2 / MusicLDM / Mustango** | FakeMusicCaps + AT-ADD generators |
| **DiffRhythm 2** | Block flow matching (non-AR within blocks, AR across) |
| **AudioGen / AudioLDM** | Environmental sound (AT-ADD's sound track) — useful as *negative* material |

⚠️ **Suno / Udio outputs cannot be self-generated under permissive terms** and their public
datasets (SONICS) aren't redistributable. Since the test set may well contain Suno/Udio-class
audio, and since 16 kHz strips their production fingerprints anyway ([02](02-sota-music.md)),
our best proxy is **maximum breadth across open TTM models** rather than trying to match a
specific commercial generator.

## Generation strategy checklist

- [ ] **Held-out generator families** for validation — never a random split ([README](README.md) #6)
- [ ] Cover **both** vocoder-based and codec-LM-based synthesis ([01](01-sota-speech.md))
- [ ] Cover **sung** as well as spoken fake voice (competition labels vocals as voice)
- [ ] Generate the **4-way voice×music grid** with varied mixing gains ([03](03-sota-singing-mixed.md))
- [ ] Include **REAL-but-processed** audio (denoised, enhanced, normalized) as REAL
- [ ] Push everything through the **identical 16 kHz + container chain** as the test set
- [ ] Record per file: model, version/revision, **seed**, prompt, post-processing params
      (required verbatim by DACON — see [05-talkboard-qa](../competition/05-talkboard-qa.md))
