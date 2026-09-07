# 05 — Generalization & Domain Shift

The binding constraint. Two deep reads here directly validate — and sharpen — our data plan.

---

### ⭐ D6 · A Data-Centric Approach to Generalizable Speech Deepfake Detection (DOSS) (2025)
[arXiv](https://arxiv.org/pdf/2512.18210) · ★

**Field** — Training-data *selection* for cross-generator generalization.

**Contribution** — **Diversity-Optimized Sampling Strategy**. Premise: *"for robust generalization
against unpredicted future attacks, there is no prior evidence that any single known attack is
more important than another."* So define granular domains — real domains = source datasets, fake
domains = **source × generator combinations** — then **cap each fake domain at N_c samples** and
scale real domains proportionally by a ratio ρ.

**Result** 🔴:

| Strategy | Data | Avg EER |
|---|---|---|
| Naive aggregation (k=12) | **6.4k hours** | 3.29% |
| **DOSS-Select (N_c=500)** | **0.2k hours (≈3%)** | **2.77%** |
| DOSS-Weight (optimal) | — | **2.34%** (29% better than best baseline) |
| XLS-R-1B + DOSS vs XLS-R-2B baseline | — | **1.65% vs 3.94%** |

**For us** 🔴🔴 — This is the most directly validating paper we found. Our license gate caps the
pool at roughly **240 hours** ([data/04](../data/04-sources.md)), which I had treated as a
constraint to work around. DOSS says **200 hours of domain-balanced data beat 6,400 hours of
naively aggregated data**. Volume is not our problem; *domain coverage and per-domain capping* are
the levers. Concretely:
- Adopt **source × generator** as the domain key in the provenance ledger
- **Cap per fake domain** rather than taking everything a dataset offers
- This supersedes the "more hours is better" instinct in [data/08](../data/08-build-plan.md)

**Limitations (stated)** — no interaction study with model scaling or compute budget; corpus
"linguistically concentrated on English and Chinese" (relevant to our Korean slice).

---

### ⭐ D7 · A General Model for Deepfake Speech Detection: Diverse Bonafide or Diverse Generators (2026)
[arXiv](https://arxiv.org/html/2603.27557v1) · ★

**Field** — Which axis of training-data diversity matters more.

**Contribution** — Isolates the two factors. **AG set**: ASVspoof 2019 + 2021 LA/DF —
53,552/861,304 bona/fake, few speakers, **100+ generators**. **BR set**: ASVspoof 2024 train/dev
over LibriSpeech — 50,131/273,176, **many speakers, few generators**.

**Result** — In-The-Wild, probability boundary 0.5 (Table III):

| Training data | Accuracy | F1 | AUC |
|---|---|---|---|
| AG (generator-diverse) | 0.71 | 0.72 | 0.74 |
| BR (speaker-diverse) | 0.75 | 0.58 | 0.69 |
| **BR + AG balanced** | **0.80** | **0.72** | **0.82** |

Best model cross-dataset (Table IV): ITW 0.87/0.79/0.83 · ASVspoof5 0.91/0.91/0.92 ·
FakeAVCeleb 0.99/0.99/0.99.

Conclusion, verbatim: *"the balance of Bonafide Resources (BR) and AI-based Generators (AG) is the
key factor to train and achieve a general Deepfake Speech Detection model."*

**For us** 🔴 — Neither axis alone suffices. Our plan ([data/05](../data/05-synthesis-plan.md))
optimizes hard for **generator count** (≥20 families) and treats real audio as a supporting pool.
This says that is half a strategy: **real-speaker/source diversity must scale alongside**. Pairs
with D6 — cap per fake domain, and spend the freed budget on real-side breadth.

---

### ⭐ D8 · From SSL Speech Models to Mixture-of-Experts for Robust Anti-Spoofing (2026)
[arXiv](https://arxiv.org/abs/2606.14639) · ★ · Odyssey 2026

**Contribution** — Replace FFN blocks in the **last 6 of 13 WavLM-Large layers** with **4 experts**,
top-k=1 routing, statistical-pooling gate.

**Result** — Across **14 datasets** (ASVspoof19 LA / 21 LA / 21 DF / 5, Sonar, FakeOrReal, DFADD,
Codecfake, LibriSeVoc, ADD2022 ×2, ADD2023 ×2, InTheWild): **macro EER 5.46% → 4.81%** (11.9% rel.),
micro EER 14.95% → 12.34%. Per-dataset range 0.04% (19LA) to 21.46% (21DF).

**For us** ⚠️ — Real cross-dataset gain, but **parameters go 178M → 329M (+85%)**. Against a
3.0 s/file budget that is expensive for a 0.65-point macro-EER gain. Card it; prefer the
parameter-efficient alternatives (meta-learned LoRA: EER 8.84→5.30% at **1.1% trainable params**;
probing-guided layer selection: **4 layers of XLS-R-300M match the full model at 1.34M trainable
params**).

---

## What this axis changes in our plan

| Change | Where |
|---|---|
| 🔴 Domain-balanced **capping** beats aggregation — 3% of data can win | [data/08](../data/08-build-plan.md) |
| 🔴 Scale **real-source diversity** alongside generator count, not after it | [data/05](../data/05-synthesis-plan.md) |
| Use **source × generator** as the domain key in the ledger | [data/08](../data/08-build-plan.md) |
| Prefer LoRA / layer-selection over MoE given our runtime budget | [survey/05](../survey/05-models.md) |

Remaining generalization papers (one-class, hyperbolic embeddings, information bottleneck,
disentanglement, continual learning) are listed in [INDEX](INDEX.md).
