# Paper Index

Flat, grep-able list. `depth`: ⭐ = deep-read from primary text · `card` = abstract-level ·
`xref` = link only (dataset-only papers). Policy and card schema in [README](README.md).

Compiled 2026-09-06 from a 5-agent broad sweep + 12 primary-source deep reads.

---

## ⭐ Deep-read (12)

| # | Title | Year | Link | Headline result | Axis |
|---|---|---|---|---|---|
| D1 | **Assessing AI-generated music detection in real-world broadcast monitoring** | 2026 | [arXiv](https://arxiv.org/html/2608.07359) | F1 0.992→**0.186** / AUC 0.998→**0.707** from clean to 8 kHz+40 kbps AAC-LC; broadcast-trained 0.472/0.775 | [02](02-music-detection.md) |
| D2 | **ArtifactNet: AI-Generated Music via Forensic Residual Physics** | 2026 | [arXiv](https://arxiv.org/html/2604.16254v2) | F1 **0.9829** unseen, 4.0M params, codec drift −83%. ⚠️ **requires 44.1 kHz — explicitly degrades at 16 kHz; weights not released; patents pending (KR+PCT)** | [02](02-music-detection.md) |
| D3 | **Finding the noise: Zero-shot AI Music Detection** ("fakeprint") | 2026 | [arXiv](https://arxiv.org/html/2607.25530) | EER 0.2–2.0% on Suno/Udio/ElevenLabs. ⚠️ **excludes 16 kHz data as unsuitable** | [02](02-music-detection.md) |
| D4 | **PC-Mix: Partial-Component Audio Spoofing Detection** | 2026 | [arXiv](https://arxiv.org/html/2607.10345) | 3 parallel branches, **no separation**. Speech on mixed 29.35% vs separated **51.38%** EER | [04](04-component-partial.md) |
| D5 | **CompSpoof: Component-Level Audio Anti-spoofing** | 2025 | [arXiv](https://arxiv.org/html/2509.15804v2) | Direct 0.827 F1 · separation w/o joint **0.668** · separation **+ joint 0.908** | [04](04-component-partial.md) |
| D6 | **A Data-Centric Approach to Generalizable Speech Deepfake Detection** (DOSS) | 2025 | [arXiv](https://arxiv.org/pdf/2512.18210) | 🔴 **0.2k h (3% of data) → 2.77% EER** vs 6.4k h → 3.29%. DOSS-Weight 2.34% | [05](05-generalization.md) |
| D7 | **A General Model for Deepfake Speech Detection: Diverse Bonafide or Diverse Generators** | 2026 | [arXiv](https://arxiv.org/html/2603.27557v1) | ITW AUC: generators-only 0.74 · speakers-only 0.69 · **balanced 0.82** | [05](05-generalization.md) |
| D8 | **From SSL Speech Models to Mixture-of-Experts for Robust Anti-Spoofing** | 2026 | [arXiv](https://arxiv.org/abs/2606.14639) | Macro EER 5.46→**4.81%** over 14 datasets. ⚠️ **178M→329M params (+85%)** | [07](07-foundation-distillation.md) |
| D9 | **Low Pass Filtering and Bandwidth Extension Against Codec Variabilities** | 2022 `[foundational]` | [arXiv](https://arxiv.org/abs/2211.06546) | Low-frequency subbands as input; **EER reduced up to 25% relative** under codecs | [06](06-robustness-channel.md) |
| D10 | **TFPARN / Training-Efficient Transformer Anti-Spoofing** | 2026 | [arXiv](https://arxiv.org/abs/2606.02980) | EER 12.52%, minDCF 0.2430; **0.79 ms/utterance**, 1.4 GB. Focal + pairwise-ranking loss | [09](09-training-losses.md) |
| D11 | **Probing Token Spaces under Generator Shift** | 2026 | [arXiv](https://arxiv.org/pdf/2606.08663) | Codec token spaces (EnCodec/DAC/XCodec) transfer unevenly across generators; ☆ no extractable table numbers | [02](02-music-detection.md) |
| D12 | **Measuring the Robustness of Audio Deepfake Detectors** | 2025 | [arXiv](https://arxiv.org/abs/2503.17577) | 10 detectors × **18 corruptions**; most vulnerable to compression, **especially neural codecs**; ☆ numbers not in abstract | [06](06-robustness-channel.md) |

---

## Speech detection & SSL backbones → [01](01-speech-detection.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| AASIST `[foundational]` | 2022 | [code](https://github.com/clovaai/aasist) | Graph-attention spectro-temporal backbone; EER 0.83% (19LA); still the default lightweight CM | card |
| XLSR-MamBo: Hybrid Mamba-Attention Backbone | 2026 | [arXiv](https://arxiv.org/pdf/2601.02944) | ACL 2026 Findings; scaling study on XLS-R features | card |
| Fake-Mamba: Real-Time Speech Deepfake Detection | 2025 | [arXiv](https://arxiv.org/abs/2508.09294) | 0.97% / 1.74% / **5.85%** EER (21LA / 21DF / ITW) | card |
| A SUPERB-Style Benchmark of SSL Models for ADD | 2026 | [arXiv](https://arxiv.org/pdf/2603.01482) | Leaderboard comparing SSL encoders for CM — **use to pick our frontend** | card |
| Comprehensive Layer-wise Analysis of SSL Models for ADD | 2025 | [arXiv](https://arxiv.org/pdf/2502.03559) | Which wav2vec2/HuBERT/WavLM layers are spoof-discriminative | card |
| Probing-Guided Layer Selection from SSL Speech Models | 2025 | [arXiv](https://arxiv.org/abs/2606.30791) | 🔴 **4 probed layers of XLS-R-300M match the full model, 1.34M trainable params** | card |
| WavLM model ensemble for audio deepfake detection | 2024 | [arXiv](https://arxiv.org/abs/2408.07414) | EER 6.56% / 17.08% on two eval sets | card |
| SLIM: Style-Linguistics Mismatch Model | 2024 | [arXiv](https://arxiv.org/pdf/2407.18517) | Detects via style/linguistics mismatch — generalizes to unseen TTS | card |
| Mixture of Experts Fusion using Frozen wav2vec 2.0 | 2024 | [arXiv](https://arxiv.org/pdf/2409.11909) | MoE head on frozen features | card |
| Supervised Post-training of Speech Foundation Models | 2026 | [arXiv](https://arxiv.org/html/2606.25328) | Post-training recipe for CM adaptation | card |
| Towards Scalable AASIST: Refining Graph Attention | 2025 | [arXiv](https://arxiv.org/abs/2507.11777) | AASIST refinement + RawBoost | card |
| Two Views, One Truth: Spectral + SSL Feature Fusion | 2025 | [arXiv](https://arxiv.org/pdf/2507.20417) | Handcrafted spectral ⊕ SSL embeddings | card |
| BUT Systems and Analyses for ASVspoof 5 | 2024 | [ISCA](https://www.isca-archive.org/asvspoof_2024/rohdin24_asvspoof.pdf) | Top ASVspoof5 CM submission + analysis | card |
| FADEL: Uncertainty-aware Fake Audio Detection (Evidential DL) | 2025 | [arXiv](https://arxiv.org/pdf/2504.15663) | Evidential head for uncertainty | card |
| Naturalness-Aware Curriculum Learning | 2025 | [arXiv](https://arxiv.org/pdf/2505.13976) | Curriculum by sample naturalness + dynamic temperature | card |
| Dynamic knowledge condensation w/ audio-selective transformer | 2025 | [Springer](https://link.springer.com/article/10.1007/s10791-025-09746-4) | KD + selective transformer | card |
| [DATASET] ASVspoof 5 (overview / design / evaluation) | 2024–26 | [overview](https://arxiv.org/abs/2408.08739) · [design](https://arxiv.org/abs/2502.08857) · [eval](https://arxiv.org/abs/2601.03944) | 32 legacy+advanced attacks; the standard benchmark | xref |

## AI music detection → [02](02-music-detection.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| Detection of AI-generated stems within hybrid human-AI music | 2026 | [arXiv](https://arxiv.org/html/2607.26874v1) | Naive separate-then-detect: 38% / 94.7% FPR. Uses per-band SNR as features | card |
| MusicDET: Zero-Shot AI-Generated Music Detection | 2026 | [arXiv](https://arxiv.org/html/2605.18072v1) | 4.51% EER FakeMusicCaps; ⚠️ pitch shift +40 pts, MP3-64k +37 pts | card |
| Improved Robustness in AI-Generated Music Detection | 2026 | [arXiv](https://arxiv.org/abs/2607.27454) | Robustness to compression/manipulation | card |
| AI-Generated Music Detection in Broadcast Monitoring (companion) | 2026 | [arXiv](https://arxiv.org/pdf/2602.06823) | Same line of work as D1 | card |
| Evaluating Fake Music Detection Under Audio Augmentations | 2025 | [arXiv](https://arxiv.org/pdf/2507.10447) | Near-perfect on raw → collapse under speed/pitch/codec | card |
| The AI Music Arms Race | 2025 | [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.254) | Suno 192 kbps / Udio 320 kbps / 48 kHz fingerprints; Boomy transfer 3–12% | card |
| Data-Driven Analysis of Text-Conditioned AI Music (Suno/Udio) | 2025 | [arXiv](https://arxiv.org/html/2509.11824v1) | Characterizes the two dominant commercial generators | card |
| From Audio Deepfake Detection to AI-Generated Music Detection | 2024 | [arXiv](https://arxiv.org/pdf/2412.00571) | Overview bridging the two fields; good citation hub | card |
| SpecTTTra / SONICS | 2025 | [arXiv](https://arxiv.org/abs/2408.14080) · [code](https://github.com/awsaf49/sonics) | ICLR 2025; long-range temporal modeling, +8% F1 over ViT on long songs | card |
| [DATASET] FakeMusicCaps · M6 · ArtifactBench | 2024–26 | [FMC](https://zenodo.org/records/13732524) · [M6](https://arxiv.org/abs/2412.06001) | Cross-ref [data/11](../data/11-source-inventory.md) | xref |

## Singing & mixed audio → [03](03-singing-mixed.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| SingFake `[foundational]` | 2023 | [arXiv](https://arxiv.org/pdf/2309.07525) · [site](https://singfake.org/) | Speech CMs: **~50% EER on mixtures vs ~38% on separated vocals** | card |
| SVDD 2024 Challenge | 2024 | [arXiv](https://arxiv.org/abs/2408.16132) | CtrSVDD top team 1.65% EER; baselines raw-waveform 13.75% vs MFCC 26.67% | card |
| Deepfake Detection of Singing Voices With Whisper Encodings | 2025 | [arXiv](https://arxiv.org/abs/2501.18919) | Noise-variant Whisper features for SVDD | card |
| Not All Deepfakes Are Created Equal (singer ID triage) | 2025 | [arXiv](https://arxiv.org/html/2510.17474) | Severity-based triage | card |
| From Talking to Singing: A New Challenge | 2026 | [arXiv](https://arxiv.org/html/2605.27944) | Singing as a distinct, harder detection problem | card |
| Alethia: A Foundational Encoder for Voice Deepfakes | 2026 | [arXiv](https://arxiv.org/pdf/2605.00251) | Claims cross-task generalization incl. singing — **backbone candidate** | card |
| Source Attribution of Singing Voice Deepfake | 2025 | [arXiv](https://arxiv.org/html/2506.03364) | Which SVS/SVC system produced the fake | card |
| [DATASET] CtrSVDD · SingNet | 2024–25 | [CtrSVDD](https://arxiv.org/html/2406.02438) · [SingNet](https://singnet-dataset.github.io/) | Cross-ref [data/11](../data/11-source-inventory.md) | xref |

## Component / partial / localization → [04](04-component-partial.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| PartialSpoof `[foundational]` | 2021–23 | [arXiv](https://arxiv.org/pdf/2204.05177) · [code](https://github.com/nii-yamagishilab/partialspoof) | Multi-resolution 20–640 ms labels; 0.77% utterance EER | card |
| Localizing Speech Deepfakes Beyond Transitions | 2026 | [arXiv](https://arxiv.org/html/2601.21925) | Segment-aware learning for scattered fake regions | card |
| Frame-level Temporal Difference Learning | 2025 | [arXiv](https://arxiv.org/pdf/2507.15101) | Temporal-difference features for short forged segments | card |
| Boundary-aware Attention for Partially Spoofed Localization | 2025 | [ICASSP](https://arxiv.org/abs/2407.21611) | Boundary info exploited explicitly | card |
| Robust Spoofed Speech Detection via Temporal Pyramid Modeling | 2026 | [arXiv](https://arxiv.org/pdf/2606.16837) | Multi-scale frame→utterance | card |
| Fine-Grained Frame Modeling in Multi-head Self-Attention | 2026 | [arXiv](https://arxiv.org/pdf/2602.04702) | Frame-level attention for localization | card |
| Anomaly Detection & Localization via Feature Pyramid Matching | 2025 | [arXiv](https://arxiv.org/abs/2503.18032) | **One-class, trained on real speech only** + localization | card |
| UDA for Locating Manipulated Regions | 2024 | [arXiv](https://arxiv.org/pdf/2407.08239) | Cross-domain localization | card |
| Manipulated Regions Localization: A Survey | 2025 | [arXiv](https://arxiv.org/html/2506.14396v1) | Survey / citation hub | card |

## Generalization & domain shift → [05](05-generalization.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| Generalizable speech deepfake detection via meta-learned LoRA | 2025 | [arXiv](https://arxiv.org/html/2502.10838v1) | Avg EER **8.84→5.30%** vs full fine-tune, **~1.1% trainable params** | card |
| Mixture of Low-Rank Adapter Experts | 2025 | [arXiv](https://arxiv.org/abs/2509.13878) | OOD EER 8.55→6.08% | card |
| Gradient Reversal + Multitask, Datasets-Aware | 2026 | [arXiv](https://arxiv.org/abs/2607.23961) | Avg EER 9.484→8.238% (13.14% rel.) | card |
| Hierarchical Structure Learning in Poincaré Sphere | 2025 | [arXiv](https://arxiv.org/html/2508.01897) | EER 0.11% (19LA) / 1.40% (21DF) / **4.91% (ITW)** | card |
| Curved Worlds, Clear Boundaries (hyperbolic/spherical) | 2025 | [arXiv](https://arxiv.org/pdf/2511.10793) | Non-Euclidean embedding spaces | card |
| Generalizable Detection via Information Bottleneck | 2025 | [arXiv](https://arxiv.org/pdf/2509.23618) | IB discards generator-specific nuisance info | card |
| Dual-Granularity Orthogonal Disentanglement | 2026 | [arXiv](https://arxiv.org/html/2606.16532) | Disentanglement + curriculum | card |
| One-Class Learning with Adaptive Centroid Shift | 2024 | [arXiv](https://arxiv.org/pdf/2406.16716) | One-class boundary vs unseen attacks | card |
| QAMO: Quality-aware Multi-centroid One-class Learning | 2025 | [arXiv](https://arxiv.org/pdf/2509.20679) | Generator-agnostic, quality-weighted | card |
| Latent Space Refinement and Augmentation | 2025 | [arXiv](https://arxiv.org/abs/2501.14240) | Latent prototype refinement | card |
| How Well Do Current Methods Generalize to the Real World? | 2026 | [arXiv](https://arxiv.org/pdf/2603.05852) | Empirical audit of real-world failure | card |
| The Generalization Gap: Detectors vs Modern Vishing | 2026 | [MDPI](https://doi.org/10.3390/electronics15132846) | Measures the gap against voice-clone vishing | card |
| Continual learning family (4 papers) | 2024–25 | [region-based](https://arxiv.org/pdf/2412.11551) · [self-adaptive](https://arxiv.org/pdf/2312.09651) · [UAP](https://arxiv.org/pdf/2511.19974) · [benchmark](https://arxiv.org/pdf/2405.08596) | Incremental adaptation to new generators | card |

## Robustness: codec, channel, laundering → [06](06-robustness-channel.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| Benchmarking Robustness in Real-world Communication (ADD-C) | 2025 | [arXiv](https://arxiv.org/abs/2504.12423) | 6 codecs: AMR-WB, EVS, IVAS, OPUS, Speex-WB, SILK | card |
| Toward Noise-Aware ADD: Survey + SNR-Benchmarks + Recipes | 2025 | [arXiv](https://arxiv.org/abs/2512.13744) | **EER improves 10–15 pts at 10→0 dB SNR after fine-tuning** | card |
| Audio Deepfake Detection in the Age of Advanced TTS | 2026 | [arXiv](https://arxiv.org/abs/2601.20510) | **41.4% relative EER reduction**, codec-trained vs vocoder-trained | card |
| CodecFake+ | 2025 | [arXiv](https://arxiv.org/abs/2501.08238) | 31 open codecs as proxy → detects 17 unseen CoSG systems | card |
| Mitigating Proxy-to-Wild Domain Gap | 2026 | [arXiv](https://arxiv.org/abs/2606.07494) | Gap between codec-proxy training and real wild deepfakes | card |
| Towards Generalized Source Tracing for Codec-Based Deepfake Speech | 2025 | [arXiv](https://arxiv.org/abs/2506.07294) | Codec source tracing | card |
| Proteus: Automated Adversarial Robustness Testing | 2026 | [arXiv](https://arxiv.org/abs/2606.29544) | **35 augmentation types across 11 categories** — usable as our aug library | card |
| RTCFake: Detection in Real-Time Communication | 2026 | [arXiv](https://arxiv.org/abs/2604.23742) | VoIP-channel conditions | card |
| Multilingual Dataset Integration (SAFE Challenge system) | 2025 | [arXiv](https://arxiv.org/abs/2508.20983) | SSL frontends × data composition × compression | card |
| Unmasking Deepfakes: Augmentations + Feature Variability | 2025 | [arXiv](https://arxiv.org/abs/2501.05545) | Augmentation strategy study | card |
| RAT: Reference-Augmented Training | 2026 | [arXiv](https://arxiv.org/abs/2606.10908) | Reference-augmented anti-spoofing | card |
| [DATASET] RADAR 2026 · ESDD 2026 · ShiftySpeech | 2025–26 | [RADAR](https://arxiv.org/abs/2605.09568) · [ESDD](https://arxiv.org/abs/2603.04865) · [Shifty](https://arxiv.org/abs/2502.05674) | Cross-ref [data/11](../data/11-source-inventory.md) | xref |

## Foundation models & distillation → [07](07-foundation-distillation.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| DK-CAST (compression-agnostic KD) | 2025 | *no primary link found* | EER 0.38% (19LA) / 2.18% (21DF); multi-level KD + MMD/center-loss. ⚠️ **unverified — locate before citing** | card |
| FTDKD: Frequency-time domain KD for compressed audio | 2024 | *no direct link found* | Targets compressed/low-quality robustness. ⚠️ unverified | card |
| Noise robust distillation of SSL speech models `[foundational]` | 2023 | [arXiv](https://arxiv.org/abs/2312.12153) | Correlation-metric distillation into small robust students | card |
| Adversarial Speaker Distillation `[foundational]` | 2022 | [arXiv](https://arxiv.org/pdf/2203.17031) | Teacher→small student compression | card |
| A Parameter-Efficient Multi-Scale Convolutional Adapter | 2025 | [arXiv](https://arxiv.org/pdf/2510.24852) | Cheaper than a full LoRA sweep | card |
| Wav2DF-TSL: Two-stage Learning + Hierarchical Experts | 2025 | [arXiv](https://arxiv.org/pdf/2509.04161) | LoRA in XLSR Q/K/V + expert fusion | card |
| OpenBEATs | 2025 | [arXiv](https://arxiv.org/pdf/2507.14129) | Fully open BEATs reimplementation — **license-safe tagging backbone** | card |

## Tagging & separation → [08](08-tagging-separation.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| SSLAM | 2025 | [arXiv](https://arxiv.org/abs/2506.12222) · [code](https://github.com/ta012/SSLAM) | AudioSet **0.502 mAP** — highest found; trained on audio *mixtures* | card |
| Transformer-to-CNN KD `[foundational]` | 2022 | [arXiv](https://arxiv.org/abs/2211.04772) | **0.483 mAP**; distils AST → MobileNetV3 — the efficient-tagging recipe | card |
| BEATs `[foundational]` | 2022 | [arXiv](https://arxiv.org/pdf/2212.09058) | Iterative tokenizer + SSL; AT-ADD winner's audio-type router | card |
| OpenBEATs | 2025 | [arXiv](https://arxiv.org/pdf/2507.14129) | Fully open BEATs reimplementation — license-safe | card |
| PANNs CNN14 `[foundational]` | 2019 | [arXiv](https://arxiv.org/pdf/1912.10211) | **0.439 mAP**, preinstalled on the eval server | card |
| EAT | 2024 | [code](https://github.com/cwx-worst-one/EAT) | AT-ADD winner's non-speech branch | card |
| Hierarchical Label Propagation | 2025 | [arXiv](https://arxiv.org/abs/2503.21826) | Label-hierarchy-aware training; biggest gains on smaller models | card |
| Mel-Band RoFormer / BS-RoFormer | 2023–24 | [Mel-Band](https://arxiv.org/abs/2310.01809) · [BS](https://ar5iv.labs.arxiv.org/html/2309.02612) | MSS SOTA at 44.1 kHz | card |
| Hybrid Spectrogram-TasNet (real-time MSS) | 2024 | [arXiv](https://arxiv.org/pdf/2402.17701) | **SDR 4.65 MUSDB, 23 ms latency** | card |
| Towards Practical Real-Time Low-Latency MSS | 2025 | [arXiv](https://arxiv.org/abs/2511.13146) | Joint latency/compute optimization | card |
| Is MixIT Really Unsuitable for Correlated Sources? | 2025 | [arXiv](https://arxiv.org/pdf/2505.07631) | Unsupervised MixIT pretraining, no stem labels needed | card |

🔴 **Two negative findings on this axis** — no dedicated 2024–26 speech/music discrimination paper
exists, and **no MSS method is confirmed native at 16 kHz**. Both detailed in [08](08-tagging-separation.md).

## Training, losses, metric alignment → [09](09-training-losses.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| Online AUC Optimization via Second-order Surrogate Loss | 2025 | [arXiv](https://arxiv.org/abs/2510.21202) | O(ln T) regret vs O(√T) for pairwise-hinge AUC | card |
| Ensemble Learning for AUC Maximization via Surrogate Loss | 2025 | [OpenReview](https://openreview.net/forum?id=kbxjkoF42x) | Stacking with an AUC-maximizing combiner | card |
| Beyond Silence: Loss & Asymmetric Approaches | 2024 | [arXiv](https://arxiv.org/html/2406.17246) | FocalLoss vs SuperLoss vs CurricularFace vs GCE | card |
| DIN-CTS: Depthwise-Inception + Contrastive Training | 2025 | [arXiv](https://arxiv.org/pdf/2502.20225) | Low-complexity + contrastive | card |
| SZU-AFS ASVspoof 5 System | 2024 | [arXiv](https://arxiv.org/abs/2408.09933) | RawBoost + SpecAugment + codec + speed perturbation | card |
| RawBoost `[foundational]` | 2021 | [arXiv](https://arxiv.org/abs/2111.04433) | **27% relative improvement** over raw E2E baseline (ASVspoof21 LA) | card |

## Synthesis & augmentation → [10](10-synthesis-augmentation.md)

Largely covered by [06](06-robustness-channel.md) and [09](09-training-losses.md); see also
[kaggle/05](../kaggle/05-transferable-playbook.md) for implementation-level recipes.

## Interpretability → [11](11-interpretability.md)

| Title | Year | Link | Note | Depth |
|---|---|---|---|---|
| Explainable-by-Design via Wiener-Hopf Linear Prediction | 2026 | [arXiv](https://arxiv.org/html/2607.12584) | Interpretable by construction; recovers under noise/MP3/telephone filtering | card |
| XAI-Grounded Explanation Generation with Training-Free MLLMs | 2026 | [arXiv](https://arxiv.org/pdf/2606.16137) | Natural-language rationales from saliency | card |
| ⚠️ The Perceived Fragility of Explanations in Audio Models | 2026 | [arXiv](https://arxiv.org/html/2606.14466) | **Post-hoc attribution can be manipulated with unchanged predictions** — caveat for our report | card |
| Diffusion-based Audio Deepfake Explanations | 2025 | [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250603425G/abstract) | Counterfactual explanations | card |
| Towards Explainable Spoofed Speech Attribution | 2025 | [arXiv](https://arxiv.org/pdf/2502.04049) | Attributes cues to synthesizer components | card |
| Interpreting Multi-Branch Anti-Spoofing Architectures | 2026 | [arXiv](https://arxiv.org/pdf/2602.17711) | **What multi-branch models key on** — directly relevant to our 3-branch design | card |
| Interpretable All-Type ADD with Audio LLMs (Freq-Time RL) | 2026 | [arXiv](https://arxiv.org/pdf/2601.02983) | Rationales across fake types | card |
| Towards Reliable Attribution and Model Recognition | 2025 | [arXiv](https://arxiv.org/pdf/2508.02521) | Multi-level autoencoder attribution | card |
