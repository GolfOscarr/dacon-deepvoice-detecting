# 09 — Challenge Playbooks (what actually won)

## ⭐ AT-ADD 2026 — the closest prior challenge to ours

ACM Multimedia 2026 Grand Challenge, **All-Type Audio Deepfake Detection**.
[eval plan](https://arxiv.org/html/2604.08184v1) · [summary](https://arxiv.org/html/2608.14249)

| | Track 1 | Track 2 |
|---|---|---|
| Task | Robust **speech** ADD under channel/acoustic variation | **Type-agnostic** real/fake over speech, sound, singing, music (type labels hidden at test) |
| Train | 49,575 clips (50/50) | 146,781 clips |
| Eval | 146,346 clips | 229,373 clips |
| Metric | Macro-F1 | Macro-F1, averaged equally across 4 types |
| Best | **90.71%** | **96.10%** |

### Track 2 winner — *starfire* (96.10%) 🔴 most relevant to us

```
BEATs (frozen) ──→ hard 4-way audio-type router
                     ├─ speech     → wav2vec2-XLSR + AASIST
                     └─ non-speech → EAT-large + AASIST  (type-biased training composition)
inference: class-specific thresholds + conservative multi-crop pooling
```

**Key lesson: hard type routing beat unified detection.** Type-aware > type-agnostic.
→ argues for **separate specialist frontends** for our voice-fake and music-fake heads rather
than one shared fake head.

### Track 1 winner — *WaveShield* (90.71%)

| Stage | Choice |
|---|---|
| Frontend | **W2V-BERT 2.0**, 3-member ensemble |
| Backend | AASIST, AASIST3, Adapter-MFA variants |
| Training | staged **frozen → LoRA → joint fine-tune**; **AM-Softmax + focal loss** ⚠️ **not evidence for the loss** — no ablation is published, it is confounded with a 3-member W2V-BERT 2.0 ensemble, LoRA staging and six augmentation families, and the metric is **Macro-F1 at a fixed threshold**, which is exactly what margin and focal losses buy ([training/03 §2](../training/03-ruled-out.md#2-focal-loss---provably-zero)) |
| Augmentation | MUSAN, RIR, codec artifacts, signal perturbation, replay simulation, bonafide segment construction |
| Inference | logit averaging, fixed threshold |

### Runners-up — fusion styles worth stealing

| Rank | Team | Approach |
|---|---|---|
| 2 | Fosafer | Multi-scale **XLSR ensemble (0.3B / 1B / 2B)** with score-level fusion |
| 3 | sonomsl | Per-window **median** aggregation + logistic-regression fusion |
| 4 | ThreeTO | Weighted logit fusion incl. **CQCC cross-attention** branches |

### AT-ADD's data recipe (a ready-made blueprint)

| Type | Real | Fake |
|---|---|---|
| Speech | AISHELL-3, LibriTTS-R, LJSpeech, Common Voice, internal multi-device | 26 unseen generators at eval; train pool incl. ProDiff, PortaSpeech, DiffSpeech, FastSpeech2, Kokoro, WaveNet, FastDiff, MeloTTS, CosyVoice, Parler-TTS, GradTTS, FastPitch, Tacotron2, Glow-TTS, WaveGlow, Tortoise-TTS, Llasa 1B, Index-TTS; StarGANv2-VC; MultiBandMelGAN |
| Sound | AudioCaps (+OOD: AVQA, CompA-R, VocalSound, TUT2016) | AudioLDM, AudioLDM 2, AudioGen; **4 unseen at eval** |
| Singing | OpenCpop, M4Singer, KiSing | Soft-VITS-SVC, NeuCoSVC, SeedVC; **5 unseen at eval** |
| Music | MusicCaps (+OOD: **FMA**, FortisAVQA) | MusicGen, MusicLDM, AudioLDM2, Stable Audio Open; **4 unseen at eval** |

Note the design principle throughout: **train generators ≠ eval generators.** Copy this.

### Cross-cutting patterns (both tracks, top-5)

| Pattern | Detail |
|---|---|
| SSL frontends, universally | W2V-BERT 2.0, XLSR, BEATs, EAT-large — frozen or LoRA |
| Broad augmentation | signal + segment level, see [08](08-augmentation.md) |
| **Multi-crop inference** | 4–5 crops, median/mean pooled |
| Structured fusion, not naive averaging | score-level, per-window median + LR, weighted logit |
| Type-aware routing | Track 2 winner |

⚠️ AT-ADD rules capped ensembles at **5 subsystems**; our binding constraint is instead the
**3.0 s/file runtime budget** ([05](05-models.md)).

---

## ASVspoof 5 (2024) — speech, unseen generators at scale

| | minDCF | EER |
|---|---|---|
| Baselines | >0.7 | >29% |
| Top-5 | <0.5 | <15% |
| Best individual | 0.115 / 0.1348 | **4.04% / 5.02%** |

Organizers' conclusions: SSL frontends dominate; ensembles of subsystems win.
[paper](https://arxiv.org/pdf/2408.08739)

---

## SVDD 2024 — singing voice

**CtrSVDD**: 47 teams, 37 beat baselines, **top 1.65% EER**. Baselines (graph-attention backend):
raw waveform 13.75% / LFCC 16.15% / Mel 25.19% / Spec 25.50% / MFCC 26.67% EER.
Raw waveform and LFCC clearly best. Details in [03](03-sota-singing-mixed.md).

---

## Korean precedent — DACON 2024 SW중심대학 AI부문 (가짜 음성 검출)

5-second English clips, up to **2 simultaneous real/fake voices** per sample; train = clean
studio recordings, eval = varied environments. Structurally similar meta (self-built training
data, domain gap between train and eval).
[competition](https://dacon.io/competitions/official/236253/overview/description) ·
[code shares](https://dacon.io/competitions/official/236253/codeshare)

| Rank | Author | Post |
|---|---|---|
| 1st (Public & Private) | rkdrn7979 | "SKKU AI Code" |
| 4th | co1dtype | "Solution" |
| 5th | – | "NightMouseHear 발표자료 & Code" (slides included) |
| 11th | 최범규 | "VoiceWizards 코드 공유" |

⚠️ Not yet read. Worth mining for Korean-competition meta and `submit.zip` engineering patterns.
