# 06 — Dataset Catalog & License Gate

## 🔴 The gate that decides everything

DACON requires submitting **the actual training data files** (URL/script/checksum manifests
explicitly rejected), and states:

> 라이선스상 **제3자 제공 자체가 제한되는 데이터라면 대회 학습 데이터로 사용할 수 없습니다.**

**Therefore: if we cannot hand the audio to DACON, we cannot train on it.** This eliminates the
majority of this field's benchmark datasets, which distribute *YouTube links*, not audio.

Confirmed usable license classes (official DACON answers): **CC-BY-NC ✅**, **CC-BY-NC-SA ✅**
(with compliance). ⚠️ **-ND variants are legally unclear** for us — NoDerivatives may bar
augmented copies and possibly trained models.

⚠️ **Every license below is unverified.** Verify on the source page before download.

## Verdict table

| Dataset | Content | License (unverified) | Verdict |
|---|---|---|---|
| **MUSDB18-HQ** | 150 tracks, **isolated vocal + accompaniment stems** | ☆ CC BY-NC-SA (46 tracks from MedleyDB, BY-NC-SA 4.0) | ⭐ **Best single asset.** Stems enable the 4-way voice×music grid. [source](https://sigsep.github.io/datasets/musdb.html) |
| **MTG-Jamendo** | Large CC-licensed music, 320 kbps MP3 | ☆ dataset v1.0.0 Apache-2.0; audio CC (per-track) | ✅ likely — real music at scale. [site](https://mtg.github.io/mtg-jamendo-dataset/) |
| **FMA** (Free Music Archive) | Real music, CC-licensed | ☆ per-track CC | ✅ likely — used as OOD real music in AT-ADD |
| **Zeroth-Korean** | Korean read speech | ★ CC BY 4.0 | ✅ [openslr/40](https://openslr.org/40/) |
| **LibriTTS-R / LJSpeech / VCTK** | English TTS-grade real speech | ☆ CC BY 4.0 / public domain | ✅ likely — AT-ADD reals |
| **Common Voice** | Multilingual real speech | ☆ CC0 | ✅ likely |
| **MLAAD** v9 | **175 TTS models**, 1002.9 h synthetic, **54 languages** | ★ CC-BY-NC 4.0 | ⭐ ✅ Largest generator-diversity asset available. Pairs with M-AILABS for reals. [hf](https://huggingface.co/datasets/mueller91/MLAAD) |
| **M-AILABS** | Real speech counterpart to MLAAD | ☆ verify | ✅ likely [repo](https://github.com/imdatceleste/m-ailabs-dataset) |
| **Codecfake** | 1,058,216 clips; 7 neural codecs; reals from LibriTTS+VCTK | ☆ verify | ✅ likely — **essential** for codec-LM TTS coverage [hf/arxiv](https://arxiv.org/abs/2406.08112) |
| **ASVspoof 2019 LA** | Classic TTS/VC spoofing | ☆ Open Data Commons Attribution | ✅ likely |
| **ASVspoof 2021 LA/DF** | + telephony/codec transmission | ☆ check LICENSE.txt | ✅ likely — **the telephone-channel asset** |
| **ASVspoof 5** | Crowdsourced, many speakers, adversarial | ☆ check README/LICENSE on Zenodo | ✅ likely |
| **CtrSVDD** | 220,798 clips / 307.98 h @16 kHz singing, 14 SVS+SVC methods | ★ **CC BY-NC-ND 4.0** | ⚠️ **ND risk — legal review required.** [train/dev](https://zenodo.org/records/10467648) · [eval](https://zenodo.org/records/10742049) |
| **Opencpop / ACE-KiSing** | Real Mandarin singing | ☆ CC-BY-NC 4.0 | ✅ likely |
| **M4Singer** | Real Mandarin singing, multi-singer | ☆ verify | ✅ likely |
| **FakeMusicCaps** | ~27.6k clips, 5 open TTM models | ☆ verify | ⚠️ fake side likely OK; **real side is MusicCaps = YouTube** |
| **M6** | Multi-generator/domain/lingual/genre/instrument, WAV | ☆ "freely available" | ✅ likely — verify. [arxiv](https://arxiv.org/abs/2412.06001) · [Sci Rep](https://www.nature.com/articles/s41598-026-36044-w) |
| **MUSAN** | Noise / music / speech for augmentation | ☆ CC BY 4.0 | ✅ likely |
| **RIRS_NOISES** | Room impulse responses | ☆ Apache-2.0 | ✅ likely |
| **SONICS** | 97k songs (48,090 real **YouTube** + 49,074 Suno/Udio) | — | ❌ **reals not redistributable** |
| **SingFake / WildSVDD** | In-the-wild deepfake songs from UGC sites | — | ❌ **YouTube-sourced** |
| **MusicCaps / AudioCaps** | Real music / sound captions | — | ❌ **YouTube-linked, links not audio** |
| **Million Song Dataset** | Real music | — | ❌ features/links, not audio |
| **In-the-Wild** (Fraunhofer) | 20.8 h bona fide + 17.2 h spoofed, 58 public figures, scraped | ☆ verify | ⚠️ scraped from social media/streaming → redistribution doubtful. Excellent **eval-only** proxy if we can't ship it (but then we can't train on it). [page](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) |
| **KsponSpeech** (AI Hub) | 969 h Korean spontaneous speech, 2,000 speakers | ☆ AI Hub terms | ⚠️ AI Hub terms typically restrict redistribution → **likely unusable**. Verify. |

## Practical consequences

1. **Self-generation is not optional, it's the core strategy.** The redistributable public pool
   is thin, especially for AI *music*. We generate → we own the files → we can ship them.
   See [07-generators.md](07-generators.md).
2. **MUSDB18-HQ is the keystone** for the mixed case: stems let us build the four-way
   voice×music real/fake grid at controlled mixing gains, exactly as the hybrid-stems paper did.
3. **MLAAD (175 TTS models, 54 languages) is the single best generator-diversity asset** and it
   is CC-BY-NC — directly usable per DACON's answer.
4. **ASVspoof 2021 LA** is our best public source of genuine telephony/codec transmission
   degradation, which the test set explicitly contains.
5. ⚠️ **All augmentation intermediates need not be shipped** (DACON's answer) — originals +
   code + config + seed suffice. So prefer **on-the-fly augmentation with fixed seeds** over
   materializing an augmented corpus. This dramatically reduces what we must deliver.
6. Maintain a **per-file provenance ledger** from the first download: source, version, license,
   license URL, redistribution verdict, date. It is required for the 2nd-stage report (20 pts).
