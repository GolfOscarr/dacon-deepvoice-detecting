# 08 — Audio Tagging (Presence) & Separation-as-Feature

Serves `VOICE_PRESENT_PROB` / `MUSIC_PRESENT_PROB` (combined weight 0.10, ROC-AUC) and the
separation-as-feature question from [04](04-component-partial.md).

---

## Audio tagging — the presence heads

| Model | AudioSet mAP | Link | Note |
|---|---|---|---|
| **SSLAM** | **0.502** | [arXiv](https://arxiv.org/abs/2506.12222) · [code](https://github.com/ta012/SSLAM) | ICLR 2025. Trained on **audio mixtures** with a source-retention loss — designed for polyphonic overlapping sources, i.e. our mixed case |
| Transformer-to-CNN KD `[foundational]` | 0.483 | [arXiv](https://arxiv.org/abs/2211.04772) | Distils AST-family transformers into **MobileNetV3** — the efficient-tagging recipe. Directly relevant to our 3.0 s/file budget |
| BEATs `[foundational]` | SOTA at publication (no mAP in abstract) | [arXiv](https://arxiv.org/pdf/2212.09058) | Iterative tokenizer + SSL; the AT-ADD Track 2 winner's **audio-type router** |
| OpenBEATs | — | [arXiv](https://arxiv.org/pdf/2507.14129) | Fully open reimplementation — **license-safe alternative** given we must ship weights |
| PANNs CNN14 `[foundational]` | 0.439 | [arXiv](https://arxiv.org/pdf/1912.10211) | **Preinstalled on the eval server** (`panns-inference==0.1.1`) — zero packaging risk |
| Hierarchical Label Propagation | — (bigger gains on smaller models) | [arXiv](https://arxiv.org/abs/2503.21826) | Label-hierarchy-aware training across PANNs-CNN6 / ConvNeXT / PaSST |

**For us** — the ordering is settled: ship **PANNs** in Phase 0 (preinstalled, 0.439 mAP is ample
for a 0.10-weight ranking task), upgrade to **BEATs or SSLAM** only if LB probing shows the presence
heads are costing points, or if we want the same encoder as the shared trunk. The
**Transformer-to-CNN KD** recipe is the template if we need tagging quality inside a tight budget.

---

## 🔴 Negative finding 1 — no recent speech/music discrimination literature

The sweep found **no dedicated 2024–2026 speech-music discrimination paper**. Closest hits were
pre-2024 (SwishNet 2018, CRNN 2021) or off-topic.

**Interpretation** — this is reassuring rather than alarming. It means there is no specialised
recent method we are missing, and **general AudioSet tagging is the right proxy** for the presence
heads. It also confirms these heads are a solved-enough problem to do cheaply and stop investing
in, as planned in [data/03](../data/03-presence-data.md).

⚠️ Coverage caveat: the sweep was arXiv-centric. **DCASE 2024–2026 task proceedings were not
searched** and are the likeliest home for such work. Worth one targeted check if the presence heads
underperform.

---

## 🔴 Negative finding 2 — no music source separation method confirmed native at 16 kHz

Every MSS benchmark found assumes **44.1 kHz**. The sweep flagged this explicitly as a gap.

| Method | Link | Note |
|---|---|---|
| BS-RoFormer `[foundational]` | [arXiv](https://ar5iv.labs.arxiv.org/html/2309.02612) · [code](https://github.com/lucidrains/BS-RoFormer) | SOTA quality on MUSDB, heavy compute |
| Mel-Band RoFormer | [arXiv](https://arxiv.org/abs/2310.01809) | Overlapping mel subbands; beats BS-RoFormer on vocals/other |
| Hybrid Spectrogram-TasNet (real-time) | [arXiv](https://arxiv.org/pdf/2402.17701) | **SDR 4.65 on MUSDB, 23 ms latency** — the efficiency point of the curve |
| Towards Practical Real-Time Low-Latency MSS | [arXiv](https://arxiv.org/abs/2511.13146) | Joint latency/compute optimization |
| Is MixIT Really Unsuitable for Correlated Sources? | [arXiv](https://arxiv.org/pdf/2505.07631) | **Unsupervised MixIT pretraining** — usable as feature-extractor pretraining without stem labels |

**For us** 🔴 — This corroborates the concern already recorded in
[survey/05](../survey/05-models.md): **HT-Demucs (preinstalled) is trained at 44.1 kHz and is
out-of-domain on our 16 kHz, telephone-inclusive audio**, and nothing in the current literature
fixes that. Combined with [04](04-component-partial.md)'s finding that naive separate-then-detect
*hurts*, the case for separation-as-preprocessing gets weaker still.

Two paths remain open, in this order:
1. **No separation** — three parallel branches on the mixture (PC-Mix). Cheapest, no 16 kHz risk.
2. **Jointly-trained separation** (CompSpoof) — the only configuration shown to beat the direct
   baseline, and jointly training it *at 16 kHz* sidesteps the domain-mismatch problem entirely,
   since we would train the separator ourselves rather than importing a 44.1 kHz one.

Using an off-the-shelf 44.1 kHz separator as a frozen preprocessor is the one option the evidence
now argues against on three independent grounds.
