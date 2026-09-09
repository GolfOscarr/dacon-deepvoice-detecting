# 10 — Open Questions & Verification Backlog

## Must verify before relying on it

| # | Item | Why it matters |
|---|---|---|
| V1 | **Every dataset license** in [06](06-datasets.md) | A wrong call means retraining from scratch, or disqualification |
| ~~V2~~ | ✅ **CLOSED 2026-09-08.** ND does **not** bar us. [#417333 A5](../competition/05-talkboard-qa.md): ND data may be used and augmented, provided the result is reproducible from **원본 파일 + 코드** | CtrSVDD is unblocked — 260 h of sung fake at 16 kHz, plus Codecfake, ST-Codecfake, SceneFake and ~63k ND music tracks |
| V3 | **W2V-BERT 2.0 license** (Seamless components vary; some CC-BY-NC) | It's the AT-ADD Track 1 winner's frontend |
| V4 | **MERT license** (likely CC-BY-NC) and its behaviour at **16 kHz** (trained at 24 kHz) | Our only dedicated music encoder option |
| V5 | **Mamba prebuilt wheels** for torch 2.7.1+cu128 / py3.11 / CUDA 12.8 | 10-min pip budget; source compilation will fail |
| V6 | **Commercial API ToS** (ElevenLabs / Suno / Udio) re: using outputs to build detectors | Likely prohibited; also unshippable |
| V7 | **SVDD 2024 WildSVDD numbers** — my secondary summary drifted into speaker-verification language | Don't quote until re-read |
| V8 | **DACON 2024 top solutions** ([09](09-challenge-playbooks.md)) | Not yet read |

## Research gaps — no published answer found

| # | Question | Note |
|---|---|---|
| G1 | 🔴 **What survives at 16 kHz for AI-music detection?** No paper evaluates AI-music detection under an 8 kHz Nyquist. | This is a genuine gap and possibly our edge. Requires our own experiment. |
| G2 | **Music-SSL (MERT) vs general-audio SSL (BEATs/EAT/SSLAM) at 16 kHz** for the music-fake head | AT-ADD's winner chose EAT-large for non-speech, *not* MERT. Untested at our rate. |
| G3 | **Best `FILE_FAKE_PROB` construction** — noisy-OR vs learned head vs max | No prior art; no other benchmark has this label structure |
| G4 | **Is HT-Demucs worth its runtime at 16 kHz?** Trained at 44.1 kHz; telephone audio is far OOD | The hybrid-stems paper used separation only for **SNR estimation**, not detection input — a cheap middle path |
| G5 | **Pooling function per head** — max/attention (any-part-fake) vs median (AT-ADD robustness) | Likely differs between presence and fake heads |
| G6 | **Denoised-real vs generated** boundary | The rules make enhancement REAL; neural enhancers are generative. No literature on this specific confusion. |

## Questions to consider asking DACON

Not answered in the rules or the existing talkboard, and each would materially change design.
Post as `[DACON 답변 요청] …`.

1. Ground-truth labeling of **vocoder/codec resynthesis of real audio** and **voice conversion**
   — generated (FAKE) or post-processing (REAL)? Same for AI-based **denoising, separation, and
   bandwidth extension** applied to real audio.
2. Is `FILE` FAKE when e.g. voice is real and music is fake? (The rule implies yes —
   "하나라도 FAKE이면" — but worth confirming explicitly.)
3. Is the 1,200-file test set **balanced** across the three audio types and across real/fake?
   Voice EER and Music EER are computed on different subsets, so composition affects strategy.
4. Are **commercial TTS/music-generation API outputs** acceptable given the requirement to
   submit training data files, when the API ToS restricts redistribution?

## Sanity thresholds

If our local CV shows any of these, suspect a leak or shortcut rather than success:

| Head | Suspicious if |
|---|---|
| Music fake, **unseen generator** | < 3% EER (published cross-generator is 46.4%) |
| Voice fake, **unseen generator** | < 1% EER (ASVspoof 5 best is ~4%) |
| Any head | perfect separation on a random split — re-split by generator |
