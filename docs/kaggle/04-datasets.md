# 04 — Kaggle-Hosted Audio Deepfake Datasets

⚠️🔴 **Kaggle mirrors are a discovery tool only.** Uploaders routinely mislabel licenses (CC0 is
the careless default). Because DACON requires us to **ship the actual training files**, and bars
data whose license restricts third-party provision, **every dataset must be traced to its original
source and verified there** before entering the corpus. See
[data/01-rules-check](../data/01-rules-check.md) and [survey/06](../survey/06-datasets.md).

| Dataset | Content | Original source to verify | Note |
|---|---|---|---|
| **The Fake-or-Real (FoR)** ☆ [kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) | >198,000 utterances; **33 synthetic voices** from DeepVoice 3, Google Cloud TTS, WaveNet, AWS Polly, Azure TTS, Baidu Cloud TTS | York University (FoR) | ⭐ Ships four variants: `for-original`, **`for-norm` (resampled to 16 kHz, volume-normalized, mono, silence-trimmed)**, `for-2sec`, **`for-rerecorded` (replayed and re-recorded — real channel distortion)** |
| **WaveFake** ☆ [kaggle](https://www.kaggle.com/datasets/walimuhammadahmad/fakeaudio) | 104,885 synthetic clips from **7 generative networks**, over 18,100 bona fide from LJSpeech + JSUT | RUB / Zenodo | Vocoder-family diversity |
| **ASVspoof 2019 LA** ☆ | 12,483 bona fide + 108,978 spoofed | ASVspoof.org / Edinburgh DataShare | ODC-BY (verify) |
| **ASVspoof 2021** ☆ [kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/avsspoof-2021) | LA + DF, with telephony/VoIP transmission | ASVspoof.org / Zenodo | ⭐ Our telephone-channel asset |
| **In-the-Wild (IWA)** ☆ | 20.8 h bona fide + 17.2 h fake, 31,779 files, 54 English speakers | [Fraunhofer AISEC](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) | ⚠️ Scraped from social media → redistribution doubtful. Best used as an *eval-only* proxy, which our rules make awkward |
| "Real vs Fake Human Voice" ☆ | Mixed community collection | unknown | ⚠️ Provenance unclear — avoid |

## 🔴 The FoR `for-norm` observation

FoR's `for-norm` variant applies **exactly the normalization the DACON organizers describe**:
resample to 16 kHz, volume normalize, mono, trim silence. That makes it (a) a directly usable
training source with minimal chain mismatch, and (b) **a reference implementation of the
normalization pipeline** we are trying to reconstruct from 3 dummy files
([data/07](../data/07-eda-plan.md#tier-s)).

⚠️ But note `for-2sec` exists specifically to "eliminate length-related bias" — a reminder that
**clip duration correlated with label** is a known trap in this exact dataset family. Our shortcut
audit already checks for it ([data/07](../data/07-eda-plan.md)).

## What to do with these

1. Treat FoR, WaveFake and ASVspoof as **Pool B (fake voice)** candidates
   ([data/04](../data/04-sources.md)) — they add generator families cheaply.
2. Verify each at source before download; record the verdict in the provenance ledger.
3. ⚠️ None of these covers **music**. The AI-music side has no Kaggle-hosted, redistributable
   dataset — self-generation remains mandatory ([data/05](../data/05-synthesis-plan.md)).
