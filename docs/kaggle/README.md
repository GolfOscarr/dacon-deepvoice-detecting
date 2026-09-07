# Kaggle & Competition Intelligence

Practical tips mined from high-effort competition solutions. Compiled 2026-09-05.

**Every claim carries an inline source tag** like `[BirdCLEF 2024, 1st]`. Confidence marks:
★ from the winner's own repo/paper · ☆ from a secondary summary or aggregator · ⚠️ risk ·
🔴 decision-changing · ❌ **forbidden by our competition rules**.

⚠️ **Access note**: kaggle.com serves an empty client-side shell (5.7 KB) to non-browser clients —
discussion and write-up pages cannot be fetched directly. **Notebooks, however, are available
through the authenticated API** (`kaggle kernels pull`) — see [06](06-notebook-code.md), which is
sourced from real code rather than prose. Everything here comes from winners'
GitHub repos, CEUR-WS/arXiv working notes, personal blogs, and aggregators. Where only a
secondary source was available it is marked ☆.

⚠️ **Source-reliability note**: the BirdCLEF 2024 top-3 details come from a third-party
*AI-generated* knowledge base ([Galaxy-Dawn/claude-scholar](https://github.com/Galaxy-Dawn/claude-scholar/blob/main/skills/kaggle-learner/references/knowledge/time-series/birdclef-2024.md)).
The specifics are plausible and the 3rd-place items cross-check against
[TheoViel's repo](https://github.com/TheoViel/kaggle_birdclef2024), but the 1st/2nd-place claims
(statistics-T filter, `min()` ensemble, checkpoint soup) are **unverified against the winners'
own write-ups** and should be treated as hypotheses to test, not established facts.

| File | Contents |
|---|---|
| [01-ai-content-detection.md](01-ai-content-detection.md) | Detecting AI-generated content: LLM-Detect-AI, DFDC, plus the RADAR/SAFE audio-deepfake challenges |
| [02-audio-classification.md](02-audio-classification.md) | BirdCLEF 2023–2026, Cornell Birdcall, Freesound 2019, Bengali.AI — the audio craft |
| [03-weak-signal-anomaly.md](03-weak-signal-anomaly.md) | G2Net, SETI, HMS — faint signal under noise, label noise |
| [04-datasets.md](04-datasets.md) | Kaggle-hosted audio deepfake datasets + license warning |
| [05-transferable-playbook.md](05-transferable-playbook.md) | ⭐ Synthesized by theme: EDA, augmentation, synthesis, features, validation, inference |
| [06-notebook-code.md](06-notebook-code.md) | ⭐ **Actual code** pulled via the Kaggle API: SED attention head, distillation, augmentation params, VAD |

---

## 🔴 Read this first: the most reliable Kaggle trick is illegal for us

**Pseudo-labeling the unlabeled target domain is the single most repeated winning technique** in
audio competitions — it appears in *every* BirdCLEF top solution from 2023 to 2026
`[BirdCLEF 2023–2026, 1st–5th]`. It is also **explicitly forbidden** by DACON rule 2.3:

> 비공개 평가 데이터셋을 활용한 추가 학습, 모델 튜닝, **Pseudo-Labeling** 등 … 허용되지 않습니다.

Likewise, cross-file score normalization — another standard Kaggle move — is barred by rule 2.4
(파일 단위 독립 예측 원칙). See [competition/04-rules](../competition/04-rules.md).

**What we do instead**: build a *proxy* target domain we own — audio pushed through the test-time
signal chain, from generators held out of training — and pseudo-label / self-train on **that**.
The technique is legal as long as the unlabeled pool is ours, not DACON's evaluation set.

## Top 12 transferable lessons

| # | Lesson | Source |
|---|---|---|
| 1 | **Validation first, before any serious modeling.** Decide fold logic before tuning or the LB will mislead you. Decision hierarchy: OOF mean → fold stability → subgroup robustness → runtime → public LB *only* as a weak cross-check | ★ `[BirdCLEF playbook 2026]` |
| 2 | 🔴 **Curated, distribution-matched training data beats better models.** Adding off-distribution data "caused severe data drift which further increased the CV/LB gap"; the winning move was finding data of *similar distribution* | ☆ `[LLM-Detect-AI 2024, efficiency prize]` |
| 3 | **Generate your own training data across many generators.** Winners fine-tuned Mistral, LLaMA-13B, Falcon, MPT, OPT, GPT-2, Bloom, Pythia, TinyLlama to produce in-domain text, plus instruction-tuned LLMs for *adversarial* samples | ★ `[LLM-Detect-AI 2024, 1st]` |
| 4 | 🔴 **Low-SNR curriculum generalizes upward, not downward.** Injecting synthetic signals at low SNR (`max(N(3.6,1),1)`) gave 2–8 bps; models trained on strong signals fail on weak ones, not vice versa | ★ `[G2Net 2021, 3rd]` |
| 5 | **Aggregation beats architecture.** A confidence-based frame-averaging heuristic alone moved a single model 0.25 → 0.22 — more than architectural complexity did | ★ `[DFDC 2020, 1st]` |
| 6 | **Ensemble aggregation function is a real hyperparameter.** `min()` across 6 models cut false positives `[BirdCLEF 2024, 1st ☆]`; geometric mean of 7+3 models `[Freesound 2019, 1st ★]`; CMA-ES-optimized logit blend `[G2Net 2021, 3rd ★]` |
| 7 | **Channel-simulating augmentation is the domain-shift fix.** Pink noise was called "very important" for the focal→soundscape gap; also RoomSimulator, AddBackgroundNoise, Gain, compression | ★ `[Cornell 2020, 1st]` ☆ `[Bengali.AI 2023, top]` |
| 8 | **Attention/SED pooling for weak labels.** PANNs-style SED with attention over frames, clipwise **and** framewise outputs thresholded separately | ★ `[Cornell 2020, 1st]`, ☆ `[BirdCLEF 2023, 3rd]` |
| 9 | **Mixup with `max()` label combination** for multi-label audio — additive waveform mixing, labels = elementwise max | ★ `[BirdCLEF 2024, 3rd]` |
| 10 | **Two-stage clean→noisy training** for label noise: train on high-confidence subset, then fine-tune on everything with down-weighting | ☆ `[HMS 2024, top]`; different loss per label-quality tier ★ `[Freesound 2019, 1st]` |
| 11 | **Runtime constraints shape the whole design.** Every BirdCLEF top solution converted to ONNX/OpenVINO and chose B0-class backbones. "A model that cannot survive notebook deployment is a teacher candidate, not a final submission candidate" | ★ `[BirdCLEF playbook 2026]` |
| 12 | **Ranking metrics reward ordering, not calibration.** "The best loss may be the one that yields the strongest OOF ordering rather than the nicest raw probabilities" | ★ `[BirdCLEF playbook 2026]` |
| 13 | 🔴 **SED attention pooling instead of GAP**, so a short event isn't diluted across a long window; train and infer on `0.5·clip + 0.5·frame_max` | ★ `[BC2026 Distilled-SED]` |
| 14 | 🔴 **Distill a frozen foundation model into a small backbone with a stop-gradient** — reported 0.898 vs 0.876. Buys foundation-model quality at student-model runtime | ★ `[BC2026 Distilled-SED]` |

## Structural parallels to our competition

| Their situation | Our situation |
|---|---|
| Clean focal recordings → messy field soundscapes `[BirdCLEF]` | Self-generated clean audio → 16 kHz telephone/codec-degraded test set |
| No training data given; generate your own `[LLM-Detect-AI]` | 별도의 학습 데이터셋을 제공하지 않으며 |
| Generalize to unseen manipulation methods `[DFDC]` | Generalize to unseen TTS/TTM generators |
| Faint signal under detector noise `[G2Net]` | Generator artifact under music at −15 dB |
| CPU-only, 90–120 min hard runtime cap `[BirdCLEF]` | L4 GPU, 60 min for 1,200 files, offline |
| Threshold-free ranking metric (macro ROC-AUC) `[BirdCLEF]` | EER + ROC-AUC, also threshold-free |
