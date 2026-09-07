# 11 — Dataset Source Inventory

Candidate data sources with descriptions and download links. **Nothing downloaded** — this is a
review list. Compiled 2026-09-05.

## How to read this

| Verdict | Meaning |
|---|---|
| ✅ | License appears to permit training **and** provision to DACON. Verify at source, then take |
| ⚠️ | Usable but a specific clause needs a decision (**gate G2** in [10](10-preprocessing-and-filtering.md)) |
| ❓ | License not established — must be checked before download |
| ❌ | Blocked: cannot be provided to a third party ⇒ cannot be used at all ([01](01-rules-check.md)) |

🔴 **The gate that governs everything here**: DACON requires submitting the actual training files
and stated *"라이선스상 제3자 제공 자체가 제한되는 데이터라면 대회 학습 데이터로 사용할 수 없습니다"*
(#417280). **Every license below is reported from secondary sources and must be verified on the
dataset's own page before download.**

---

## 🔴 1. AI-Hub — blocked by default, but worth one inquiry

**Finding**: AI-Hub's 이용정책 states that data *"제공자의 승인 없이 제3자에게 제공, 양도, 대여,
판매 또는 사용 허가를 할 수 없다"*, and that **원본 데이터는 제3자와 공유할 수 없으며, 다만 데이터를
활용한 학습 결과물(모델)은 자유롭게 활용 가능**. Non-commercial research use only; phone identity
verification required; overseas export needs separate NIA agreement.

Under DACON's rule that is **❌ — we cannot ship the files, so we cannot train on them.**

🔴 **But there is a genuine angle worth pursuing.** This competition's 주최 is
**행정안전부 + 한국지능정보사회진흥원 (NIA)** — and **NIA operates AI-Hub**. Providing AI-Hub data to
a competition co-hosted by AI-Hub's own operator is not obviously "제3자 제공". Two cheap actions
with a potentially large payoff:

1. Post a `[DACON 답변 요청]` asking whether AI-Hub datasets may be used, given NIA is a 주최기관,
   and whether submission to the 2차 평가 counts as 제3자 제공.
2. Contact AI-Hub directly (safezone1@aihub.kr / 02-525-7708) describing the competition.

If granted, this unlocks a very large Korean speech pool — **4,000+ hours of 자유대화 음성 (일반)**
and **3,000+ hours (노인)**, plus KsponSpeech's 969 h — matched to the competition's own language.
Until then, treat all AI-Hub data as ❌.

| AI-Hub dataset | Content | Link |
|---|---|---|
| 자유대화 음성(일반남녀) | 2,000+ speakers, ~4,000 h | [dataSetSn=109](https://aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=&aihubDataSe=data&dataSetSn=109) |
| 한국어 음성 (KsponSpeech) | 969 h, 2,000 speakers, 622,545 utterances | [aidata/105](https://aihub.or.kr/aidata/105) |
| 음성 및 모션 합성 데이터 | Synthesis data | [dataSetSn=539](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=539) |
| 감정 음성 / 아동 음성 / 외국인 한국어 발화 | Speaker & style diversity | [637](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=637) · [540](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=540) · [505](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=505) |
| 이용정책 (read this first) | — | [usagepolicy](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) |

---

## ⭐ 2. Crown jewels — get these five first

| Rank | Dataset | Why it is the single most valuable item in its category | Link | Verdict |
|---|---|---|---|---|
| **1** | **CompSpoof V2** | 🔴 **The only public dataset with our exact label structure.** Five classes = every combination of bona fide / spoofed **speech × environmental sound**: `original`, `bonafide_bonafide`, `spoof_bonafide`, `bonafide_spoof`, `spoof_spoof`. 250k+ samples, ~283 h, 4 s clips, **multiple sampling rates**, `mix_target_snr` in metadata. This is cells 6/7 of [02](02-label-taxonomy.md) handed to us | [HF](https://huggingface.co/datasets/XuepingZhang/ESDD2-CompSpoof-V2) · [V1](https://huggingface.co/datasets/XuepingZhang/CompSpoof) · [site](https://xuepingzhang.github.io/CompSpoof-V2-Dataset/) · [paper](https://arxiv.org/abs/2509.15804) | ⚠️ CC BY-NC 4.0 (NC is fine per DACON #417212) |
| **2** | **MLAAD** v9 | Largest generator-diversity asset anywhere: **175 TTS models, 1002.9 h, 54 languages**. Generalization to unseen generators is the binding constraint; nothing else comes close | [deepfake-total.com/mlaad](https://deepfake-total.com/mlaad) · [HF](https://huggingface.co/datasets/mueller91/MLAAD) | ⚠️ CC-BY-NC 4.0 |
| **3** | **MUSDB18-HQ** | 150 tracks with **isolated vocal + accompaniment stems** — the keystone for building the 4-way voice×music grid ourselves | [sigsep](https://sigsep.github.io/datasets/musdb.html) | ⚠️ CC BY-NC-SA; **academic use only, access request required** |
| **4** | **Codecfake** (Zenodo) | 132,277 real / **925,939 fake** across neural codecs. Modern TTS generates directly from codecs, skipping the vocoder — vocoder-trained detectors are structurally blind ([survey/01](../survey/01-sota-speech.md)) | [Zenodo 13838106](https://zenodo.org/records/13838106) · [codecfake.github.io](https://codecfake.github.io/) · [ST-Codecfake](https://zenodo.org/records/14631091) | ❓ verify |
| **5** | **ASVspoof 2021 LA** | The best public source of genuine **telephony / VoIP transmission** degradation — the test set explicitly contains 전화채널 audio | [asvspoof.org](https://www.asvspoof.org/index2021.html) | ❓ check LICENSE.txt |

---

## 3. Pool B — fake voice

Sorted by value. Sample counts as reported by the [media-sec-lab aggregator](https://github.com/media-sec-lab/Audio-Deepfake-Detection).

| Dataset | Year | Lang | Real / Fake | Note | Link | Verdict |
|---|---|---|---|---|---|---|
| **MLAAD** | 2024 | 54 langs | – / 76k (v-dependent) | ⭐ 175 TTS models | [link](https://deepfake-total.com/mlaad) | ⚠️ NC |
| **Codecfake** | 2024 | en, zh | 132,277 / 925,939 | ⭐ codec-LM family | [Zenodo](https://zenodo.org/records/13838106) | ❓ |
| **ASVspoof 5** | 2024 | en | 188,819 / 815,262 | Crowdsourced, adversarial attacks | [Zenodo 14498691](https://zenodo.org/records/14498691) | ❓ |
| **ASVspoof 2021** | 2021 | en | LA 18,452 / 163,114 · DF · PA | ⭐ telephony/codec | [asvspoof.org](https://www.asvspoof.org/index2021.html) | ❓ |
| **ASVspoof 2019** | 2019 | en | LA 12,483 / 108,978 | Classic TTS/VC baseline | [DataShare](https://datashare.ed.ac.uk/handle/10283/3336) | ☆ ODC-BY |
| **ShiftySpeech** | 2025 | en, zh, ja | 3,000+ hours | Very large; distribution-shift focused | [HF](https://huggingface.co/datasets/ash56/ShiftySpeech) | ❓ |
| **SpoofCeleb** | 2024 | en | 250k+ | Built from in-the-wild speech | [jungjee.com](http://www.jungjee.com/spoofceleb/) | ❓ |
| **DFADD** | 2024 | en | 44,455 / 163,500 | ⭐ **Diffusion + flow-matching TTS** — the current generation, absent from older sets | [GitHub](https://github.com/isjwdu/DFADD) | ❓ |
| **WaveFake** | 2021 | en, ja | 16,283 / 117,985 | 7 vocoder networks over LJSpeech + JSUT | [Zenodo](https://zenodo.org/record/5642694) | ❓ |
| **FoR (Fake-or-Real)** | 2019 | en | 111k / 87k | 33 synthetic voices. ⭐ Ships `for-norm` (**already 16 kHz, mono, volume-normalized, silence-trimmed**) and `for-rerecorded` (real replay channel) | [York U](http://bil.eecs.yorku.ca/datasets) · [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) | ❓ |
| **PartialSpoof** | 2023 | en | 12,483 / 108,978 | ⭐ **Segment-level labels, 20–640 ms** — trains the frame-max head | [Zenodo](https://zenodo.org/record/4817532) | ❓ |
| **LlamaPartialSpoof** | 2024 | en | 10,573 / 33,479 | LLM-era partial spoofing | [GitHub](https://github.com/hieuthi/LlamaPartialSpoof) | ❓ |
| **DiffSSD** | 2024 | en | 24,226 / 70,000 | Diffusion-based synthetic speech | [HF](https://huggingface.co/datasets/purdueviperlab/diffssd) | ❓ |
| **LibriSeVoc** | 2023 | en | 13,201 / 79,206 | Vocoder-artifact focused | [GitHub](https://github.com/csun22/SyntheticVoice-Detection-Vocoder-Artifacts) | ❓ |
| **Voc.v1–v4** | 2023 | en | 2,580 / 10,320 | Vocoded-data training sets from NII | [GitHub](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/09-asvspoof-vocoded-trn) | ❓ |
| **CVoiceFake** | 2024 | multi | 23,544 / 91,700 | Multilingual vocoder set | [SafeEar](https://safeearweb.github.io/Project/) | ❓ |
| **MLADDC** | 2024 | multi | 80k / 160k | Multilingual | [site](https://speech007.github.io/MLADDC_Nips/) | ❓ |
| **XMAD-Bench** | 2025 | multi | 414,858 | Cross-lingual benchmark | [GitHub](https://github.com/ristea/xmad-bench/) | ❓ |
| **In-the-Wild** | 2022 | en | 19,963 / 11,816 | 58 public figures, scraped from social media | [Fraunhofer](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) | ⚠️ scraped → redistribution doubtful |
| **CFAD** | 2022 | zh | 38,600 / 77,200 | Chinese | [Zenodo](https://zenodo.org/record/8122764) | ❓ |
| **DECRO** | 2023 | en, zh | 21,218/41,880 (zh); 12,484/42,799 (en) | ⭐ Built for **channel/codec robustness** | [GitHub](https://github.com/petrichorwq/DECRO-dataset) | ❓ |
| **SceneFake** | 2022 | en | 19,838 / 64,642 | **Acoustic-scene manipulation** — background tampered, speech genuine | [Zenodo](https://zenodo.org/record/7663324) · [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/scenefake) | ❓ |
| **TIMIT-TTS** | 2022 | en | – / 79,120 | | [Zenodo](https://zenodo.org/records/6560159) | ❓ |
| **ODSS** | 2023 | en, de, es | 11,032 / 18,993 | | [Zenodo](https://zenodo.org/records/8370669) | ❓ |
| **RFP** | 2024 | en | 28,115 / 74,199 | | [Zenodo](https://zenodo.org/records/10202142) | ❓ |
| **HABLA** | 2023 | es | 22,000 / 58,000 | Spanish | [GitHub](https://github.com/Ruframapi/HABLA) | ❓ |
| **BanglaFake** | 2025 | bn | 12,260 / 13,260 | | [HF](https://huggingface.co/datasets/sifat1221/banglaFake) | ❓ |
| **AVspoof / ASVspoof 2015** | 2015 | en | 15.5k/120.5k · 16,651/246,500 | Historical baselines | [Zenodo](https://zenodo.org/record/4081040) · [DOI](http://dx.doi.org/10.7488/ds/298) | ❓ |

---

## 4. Pool D + sung voice — fake music & singing

| Dataset | Year | Content | Note | Link | Verdict |
|---|---|---|---|---|---|
| **SingNet** | 2025 | **2,963.4 hours** | ⭐ By far the largest singing resource found | [site](https://singnet-dataset.github.io/) | ❓ |
| **CtrSVDD** | 2024 | 32,312 real / 188,486 fake singing, 307.98 h **@16 kHz** | ⭐ Already at our sample rate; 14 SVS+SVC methods | [Zenodo train/dev](https://zenodo.org/records/10467648) · [eval](https://zenodo.org/records/10742049) · [GitHub](https://github.com/SVDDChallenge/) | ⚠️ **CC BY-NC-ND** — the **ND** term may bar augmented copies. Legal review ([V2](../survey/10-open-questions.md)) |
| **FakeMusicCaps** | 2024 | 5.5k real / 27,605 fake, 5 TTM models | Fake side is self-contained; real side is MusicCaps (YouTube) | [Zenodo](https://zenodo.org/records/13732524) | ⚠️ fake ✅ / real ❌ |
| **SONICS** | 2025 | 48,090 real / 49,074 fake (Suno, Udio) | Full songs. **Reals scraped from YouTube** | [Kaggle](https://www.kaggle.com/datasets/awsaf49/sonics-dataset) · [GitHub](https://github.com/awsaf49/sonics) | ⚠️ fake ❓ / real ❌ |
| **SingFake** | 2024 | 634 real / 671 fake | Small; UGC-sourced | [GitHub](https://github.com/yongyizang/SingFake) | ❌ scraped |
| **FSD** | 2024 | 200 real / 500 fake (zh) | Small fake-song set | [GitHub](https://github.com/xieyuankun/FSD-Dataset) | ❓ |
| **M6** | 2024 | Multi-generator/domain/lingual/genre/instrument, WAV | Claims free availability | [arXiv](https://arxiv.org/abs/2412.06001) · [Sci Rep](https://www.nature.com/articles/s41598-026-36044-w) | ❓ |
| **FakeSound / FakeSound2** | 2024–25 | 3,798 fake environmental | Non-speech, non-music generated audio | [GitHub](https://github.com/FakeSoundData/FakeSound) | ❓ |
| **6KSFx** | 2025 | 6,000 fake SFX | | [GitHub](https://github.com/nellyngz95/6KSFX) | ❓ |

---

## 5. Pool A — real voice

| Dataset | Content | Link | Verdict |
|---|---|---|---|
| **Zeroth-Korean** | 51.6 h, 105 speakers, transcripts | [OpenSLR 40](https://openslr.org/40/) | ✅ **CC BY 4.0** — our best clean Korean source |
| **KSS** | 12 h Korean single female speaker | [Kaggle](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset) | ❓ verify |
| **ClovaCall** | Korean **call-based** speech, 11,000+ people | [GitHub](https://github.com/clovaai/ClovaCall) | ❓ ⭐ telephone-channel Korean if permitted |
| **Common Voice** | Multilingual incl. Korean, transcripts | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/) · [Kaggle mirror](https://www.kaggle.com/datasets/mozillaorg/common-voice) | ☆ CC0 |
| **LibriTTS-R / LibriSpeech** | English, TTS-grade, transcripts | [OpenSLR](https://openslr.org/) | ☆ CC BY 4.0 |
| **LJSpeech** | 24 h single English speaker | [keithito.com](https://keithito.com/LJ-Speech-Dataset/) | ☆ public domain |
| **VCTK** | 109 English accents | [DataShare](https://datashare.ed.ac.uk/handle/10283/3443) | ☆ CC BY 4.0 |
| **M-AILABS** | Multilingual; the real counterpart to MLAAD | [GitHub](https://github.com/imdatceleste/m-ailabs-dataset) | ❓ |
| **MLS** (Multilingual LibriSpeech) | 50k+ h, 8 languages | [OpenSLR 94](https://openslr.org/94/) | ☆ CC BY 4.0 |
| **English multi-speaker for voice cloning** | Cloning reference pool | [Kaggle](https://www.kaggle.com/datasets/mfekadu/english-multispeaker-corpus-for-voice-cloning) | ❓ |
| **Korean open-corpus index** | Maintained list of accessible Korean corpora | [GitHub](https://github.com/indra622/Korean-open-speech-corpora) | — |
| **TIMIT** | Phonetic English | [Kaggle mirror](https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech) | ❌ LDC-licensed; the Kaggle mirror is almost certainly unauthorized |

**Sung real voice** (needed or every song's vocal reads as fake): Opencpop, M4Singer, ACE-KiSing
(☆ CC-BY-NC-4.0), Ofuton-P, Oniku Kurumi, Kiritan, JVS-MuSiC — all enumerated in the
[CtrSVDD paper](https://arxiv.org/html/2406.02438); plus **MUSDB18-HQ vocal stems**.

---

## 6. Pool C — real instrumental / music

| Dataset | Content | Link | Verdict |
|---|---|---|---|
| **MTG-Jamendo** | 55,000+ full tracks, 3,777 h, **195 tags** (incl. instrumental), 320 kbps MP3 | [site](https://mtg.github.io/mtg-jamendo-dataset/) | ☆ dataset Apache-2.0, audio per-track CC. ⭐ **Tags double as TTM prompts** ([05 S-A4](05-synthesis-plan.md)) |
| **FMA** | Free Music Archive dump; "each track is legally free to download" | [GitHub](https://github.com/mdeff/fma) · [paper](https://arxiv.org/pdf/1612.01840) | ☆ per-track CC |
| **MUSDB18-HQ** | ⭐ 150 tracks, isolated stems | [sigsep](https://sigsep.github.io/datasets/musdb.html) | ⚠️ CC BY-NC-SA, academic only, access request |
| **MedleyDB** | 122 → 196 songs, hierarchical multitrack stems | [medleydb.weebly.com](https://medleydb.weebly.com/) | ⚠️ CC BY-NC-SA |
| **MoisesDB** | Source separation **beyond 4 stems** | [paper](https://archives.ismir.net/ismir2023/paper/000073.pdf) | ❓ |
| **Slakh2100** | Synthesized multitrack from MIDI — **unlimited clean instrumental stems** | [slakh.com](http://www.slakh.com/) | ❓ ⭐ worth checking; synthetic-but-not-AI |
| **MagnaTagATune** | Tagged music clips | [Kaggle mirror](https://www.kaggle.com/datasets/stefanneacsu/magnatagatune-pcm8-top50) | ❓ |
| **GTZAN** | 1,000 × 30 s, 10 genres | [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | ⚠️ Known label errors and duplicates; provenance murky |
| **Million Song Dataset** | Features + links only, **no audio** | — | ❌ no audio |

---

## 7. Pool E — noise, environment, impulse responses

| Dataset | Content | Link | Verdict |
|---|---|---|---|
| **MUSAN** | Noise / music / speech partitions for augmentation | [OpenSLR 17](https://openslr.org/17/) | ☆ CC BY 4.0 ⭐ ⚠️ never use the *music* partition as "noise" ([06](06-augmentation-spec.md)) |
| **RIRS_NOISES** | Room impulse responses | [OpenSLR 28](https://openslr.org/28/) | ☆ Apache-2.0 |
| **ESC-50** | 2,000 environmental clips, 50 classes | [GitHub](https://github.com/karolpiczak/ESC-50) | ☆ CC BY-NC |
| **FSD50K** | 51k Freesound clips, 200 classes | [Zenodo](https://zenodo.org/records/4060432) | ☆ CC (per-clip) |
| **DEMAND** | Real multichannel noise environments | [Zenodo](https://zenodo.org/records/1227121) | ☆ CC BY-SA |
| **WHAM!** | Real ambient noise for speech separation | [wham.whisper.ai](http://wham.whisper.ai/) | ❓ |
| **AudioCaps / VGGSound / AudioSet** | Used by CompSpoof for the environmental component | — | ❌ YouTube-linked, no audio |

---

## 8. Kaggle mirrors — quick download index

With the API configured, these pull with `source .env && .venv/bin/kaggle datasets download <ref>`.
⚠️ **Kaggle mirrors frequently carry wrong license metadata — always verify at the original source
before use** ([kaggle/04](../kaggle/04-datasets.md)).

| Ref | Title | Size |
|---|---|---|
| `mohammedabdeldayem/the-fake-or-real-dataset` | FoR (deepfake audio) | 17.2 GB |
| `walimuhammadahmad/fakeaudio` | WaveFake | 28.9 GB |
| `mohammedabdeldayem/avsspoof-2021` | ASVspoof 2021 | 62.3 GB |
| `awsaf49/asvpoof-2019-dataset` | ASVspoof 2019 | 25.3 GB |
| `anishsarkar22/asvpoof-2019-dataset-la` | ASVspoof 2019 LA only | 7.6 GB |
| `mohammedabdeldayem/scenefake` | SceneFake | 5.8 GB |
| `awsaf49/sonics-dataset` | SONICS real vs fake songs | 32.3 GB |
| `birdy654/deep-voice-deepfake-voice-recognition` | DEEP-VOICE | 4.0 GB |
| `bhaveshkumars/release-in-the-wild` | In-the-Wild | 8.1 GB |
| `adarshsingh0903/audio-deepfake-detection-dataset` | Audio Deepfake Detection | 1.2 GB |
| `jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech` | Fake vs Real Speech (2026) | 4.4 GB |
| `unidpro/real-vs-fake-human-voice-deepfake-audio` | Real vs Fake Human Voice | 88 MB |
| `ameythakur20/deepfakeaudio` | Neural Voice Cloning | 577 MB |
| `vladimirkeller/generated-russian-phrases` | Russian deepfake audio | 7.9 GB |
| `bryanpark/korean-single-speaker-speech-dataset` | KSS Korean | 3.1 GB |
| `mozillaorg/common-voice` | Common Voice (2017 snapshot) | 12.9 GB |
| `stefanneacsu/magnatagatune-pcm8-top50` | MagnaTagATune | 5.8 GB |

---

## 9. Aggregators — where to look for more

| Resource | What it gives |
|---|---|
| ⭐ [media-sec-lab/Audio-Deepfake-Detection](https://github.com/media-sec-lab/Audio-Deepfake-Detection) | **~60 datasets with links**, grouped by attack type (TTS, VC, replay, vocoder, codec, TTM, impersonation). The single best index found; §3–4 above are largely drawn from it |
| [deepfake-total.com](https://deepfake-total.com/related_work/) · [Fraunhofer AISEC](https://deepfake-demo.aisec.fraunhofer.de/related_work/) | Curated related-work + dataset pages |
| [AUDDT toolkit](https://arxiv.org/pdf/2509.21597) | Benchmark harness that already wires up many of these datasets |
| [Korean-open-speech-corpora](https://github.com/indra622/Korean-open-speech-corpora) | Maintained list of accessible Korean corpora |
| [OpenSLR](https://openslr.org/) | Speech and noise resources, mostly permissive |
| [Zenodo](https://zenodo.org/) | Hosts most academic audio datasets; license stated per record |

---

## 10. Recommended review order

1. **CompSpoof V2** — closest thing to our exact task that exists. Check the NC terms and take it
2. **MLAAD** — generator diversity, nothing else comes close
3. **MUSDB18-HQ** — request academic access early; it gates cells 6/7
4. **Codecfake + ASVspoof 2021 LA** — codec-LM and telephony coverage
5. **Zeroth-Korean + MUSAN + RIRS** — clean permissive foundations
6. **MTG-Jamendo + FMA** — real music at scale, tags reusable as TTM prompts
7. **PartialSpoof** — if we adopt segment-level supervision
8. **DFADD** — diffusion/flow-matching TTS, the current generation
9. **CtrSVDD** — only if the **ND** clause clears legal review
10. **AI-Hub** — only if the NIA inquiry (§1) succeeds

## Before anything is downloaded

Every row gets a provenance-ledger entry with an explicit **redistribution verdict, the license
URL, who verified it and when** ([08](08-build-plan.md#provenance-ledger)). Contested cases go to
**gate G2** ([10 §5](10-preprocessing-and-filtering.md)) — that call is yours, not mine.
