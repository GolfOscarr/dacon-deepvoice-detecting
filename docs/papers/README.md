# Papers

Per-paper reference index for DACON 236749. Built for lookup, not reading end-to-end.

**Entry point: [`INDEX.md`](INDEX.md)** — one line per paper, grep-able, with a `depth` column.
Topic files carry the full cards.

| File | Axis |
|---|---|
| [01-speech-detection.md](01-speech-detection.md) | Speech deepfake detection — SOTA systems & backbones |
| [02-music-detection.md](02-music-detection.md) | AI-generated music detection |
| [03-singing-mixed.md](03-singing-mixed.md) | Singing voice & vocals-over-accompaniment |
| [04-component-partial.md](04-component-partial.md) | Component-level, partial & localized detection |
| [05-generalization.md](05-generalization.md) | Generalization & domain shift |
| [06-robustness-channel.md](06-robustness-channel.md) | Codec, telephone, bandwidth, laundering |
| [07-foundation-distillation.md](07-foundation-distillation.md) | Audio SSL foundation models & distillation |
| [08-tagging-separation.md](08-tagging-separation.md) | Audio tagging (presence) & separation-as-feature |
| [09-training-losses.md](09-training-losses.md) | Losses & metric-aligned training |
| [10-synthesis-augmentation.md](10-synthesis-augmentation.md) | Data synthesis & augmentation methods |
| [11-interpretability.md](11-interpretability.md) | Interpretability & explainability |

Related: [`../survey/`](../survey/README.md) is the *synthesized narrative*; this is the
*per-paper index*. [`../data/11-source-inventory.md`](../data/11-source-inventory.md) holds
datasets — dataset-only papers are cross-referenced there rather than carded here.

---

## Recency policy

| Window | Treatment |
|---|---|
| **2025-01 → 2026-09** | Primary target |
| **2024** | Included — still current in this field |
| **Pre-2024** | Only if still the live backbone of 2026 systems (AASIST, wav2vec2/XLS-R, WavLM, PANNs/BEATs, RawBoost, PartialSpoof). Tagged `[foundational]` |
| **Excluded** | LCNN · CQCC/GMM-era anti-spoofing · pre-SSL frontends · GTZAN-era music classification · anything since beaten by a wide margin |

## Read depth

| Depth | Meaning |
|---|---|
| ⭐ **deep** | Read from the primary source. Full card with method detail, exact numbers vs baseline, ablations, limitations, and "what to steal" |
| **card** | Abstract-level card from the broad sweep |
| **xref** | One line + link only. Dataset-only papers land here |

## Selection criteria (how `deep` was chosen)

Scored, weighted for this competition specifically:

| Criterion | Weight | Rationale |
|---|---|---|
| Reports **cross-dataset / unseen-generator** results | ×3 | Our binding constraint — in-distribution numbers are near-meaningless here |
| States a **delta vs a named baseline** | ×2 | An absolute number without a baseline is not evidence |
| Peer-reviewed venue | ×2 | ICASSP · Interspeech · ASRU · ACM MM · ISMIR · ICLR · NeurIPS · TASLP |
| **Code or weights** released, usable license | ×2 | We must ship weights inside a 10 GB offline zip |
| Relevance to our weak heads | ×2 | Music (weight 0.27, thin literature) and component-level |
| Recency | ×1 | |

## Card schema

```
### <Title> (venue, year)                                   [depth] [tier]
Field         which axis, and which of our five heads it touches
Contribution  what is new
Result        the number — metric, benchmark, and the baseline it beat
Code          link + license
For us        concrete applicability, or an explicit "not applicable, because…"
```

## Source marks

★ read from the paper's own text · ☆ from an abstract or listing only · ⚠️ risk · 🔴 decision-changing

**Rule**: numbers must come from the paper's own abstract or tables. Never from a secondary
summary — a summarizer previously rendered a deepfake paper in speaker-verification terms, and
that class of error is invisible once it lands in a reference doc.

---

## Top papers to actually read (ranked)

1. **PC-Mix** — our exact architecture, at our sample rate → [04](04-component-partial.md)
2. **CompSpoof** — the separation counterargument, with the decisive ablation → [04](04-component-partial.md)
3. **DOSS** — 3% of data beats 100% → [05](05-generalization.md)
4. **Broadcast monitoring** — the 16 kHz answer → [02](02-music-detection.md)
5. **ArtifactNet** — 4M-param codec-aware music detector → [02](02-music-detection.md)
6. **Diverse Bonafide or Diverse Generators** → [05](05-generalization.md)
7. **LPF + bandwidth extension** — low-subband input for codec robustness → [06](06-robustness-channel.md)
8. **Probing-Guided Layer Selection** — 4 XLS-R layers ≈ full model → [INDEX](INDEX.md)
9. **Meta-learned LoRA** — EER 8.84→5.30% at 1.1% params → [05](05-generalization.md)
10. **Proteus** — 35 augmentations, 11 categories, as a library → [06](06-robustness-channel.md)

*Status (2026-09-06): broad sweep complete (5 agents, ~75 candidates); 12 deep reads done.
Deep-read budget was concentrated on axes 02/04/05/06 where our open questions were.*
