# Data Strategy

| File | Contents |
|---|---|
| [00-idea-log.md](00-idea-log.md) | Your ideas, each with a verdict and where it landed |
| [01-rules-check.md](01-rules-check.md) | 🔴 **Crawling verdict** + allowed source classes |
| [02-label-taxonomy.md](02-label-taxonomy.md) | 4 pools → 8 cells; edge cases; the composition trap |
| [03-presence-data.md](03-presence-data.md) | Stage-1 presence dataset + acceptance criteria |
| [04-sources.md](04-sources.md) | Concrete source inventory per pool, with hours and licenses |
| [05-synthesis-plan.md](05-synthesis-plan.md) | Self-generation; the paired T1/T2/T3 design |
| [06-augmentation-spec.md](06-augmentation-spec.md) | On-the-fly composition + augmentation, label-safety rules |
| [07-eda-plan.md](07-eda-plan.md) | Dummy-file forensics, feature analysis, shortcut audit |
| [08-build-plan.md](08-build-plan.md) | Phases, volumes, splits, provenance ledger, shipping |
| [09-risks-and-checks.md](09-risks-and-checks.md) | Adversarial self-review with pre-committed checks |
| [10-preprocessing-and-filtering.md](10-preprocessing-and-filtering.md) | ⭐ Preprocessing / filtering / salvage catalog + **the human review gates (G1–G8)** |
| [11-source-inventory.md](11-source-inventory.md) | ⭐ **Dataset inventory** — ~70 candidate sources with links, sizes and license verdicts |

## The plan in six lines

1. **No crawling** — blocked by the competition rules, not by copyright ([01](01-rules-check.md)).
2. Collect **four component pools** (real/fake × voice/instrumental), not eight dataset types.
3. **Compose** the 8 label cells on the fly; presence labels come free from composition.
4. Cells **real-voice+fake-music** and **fake-voice+real-music** are why two fake heads exist —
   they must be built from stems, and cannot be scraped.
5. Optimize for **generator-family count**, not hours; hold 25–30% of families out for validation.
6. Every transform is **label-independent** and every file is normalized through the **test signal
   chain** — violating either invalidates all local measurement.

## Method tiers (the actual work queue)

[05](05-synthesis-plan.md), [06](06-augmentation-spec.md) and [07](07-eda-plan.md) are now tiered
catalogs. Each entry states **what it analyzes / why** and **what artifact it produces**.

| Tier | Meaning | When |
|---|---|---|
| **S** | Blocking — later work is invalid without it | Phase 0–1 |
| **A** | High expected value, evidence-backed | Phase 1–2 |
| **B** | Worth doing; moderate value or cost | Phase 2–3 |
| **C** | Speculative / low-yield | Spare time only |
| **X** | Rejected, with the reason recorded | — |

### Tier S at a glance — 12 items, all blocking

| Area | Items |
|---|---|
| **EDA** | `E-S1` dummy-file forensics → `signal_chain.yaml` · `E-S2` shortcut audit (**gate: AUC < 0.60**) · `E-S3` data memo · `E-S4` metadata role assignment |
| **Synthesis** | `S-S1` T3 resynthesis twins · `S-S2` generator-family breadth with pre-assigned splits · `S-S3` the 4-way voice×music grid · `S-S4` REAL-processed slice |
| **Augmentation** | `A-S1` test-chain normalization · `A-S2` label-independent RNG discipline · `A-S3` codec round-trip · `A-S4` telephone chain |

### The three Tier-A items I'd start immediately after

`E-A1` 16 kHz survivability probe (tests the premise the music head rests on) ·
`A-A1` cross-domain MixUp (legal domain bridging where pseudo-labeling is not) ·
`A-A3` low-SNR-skewed component mixing.

## Review gates — where I stop and ask you

Defined in [10 §5](10-preprocessing-and-filtering.md). A gate exists wherever a decision is
irreversible, threshold-dependent without ground truth, affects a large share of the corpus, or
trades our training distribution against the test distribution.

| Gate | Trigger |
|---|---|
| **G1** | Signal-chain spec inferred from the 3 dummy files |
| **G2** | Contested license verdict |
| **G3** | **Any threshold, before it is applied at scale** |
| **G4** | A filter would drop >5% of a pool, or salvage rates differ >10 pts across cells |
| **G5** | Any operation that would modify or delete stored audio |
| **G6** | Symmetric preprocessing policy (channels, loudness, silence trimming) |
| **G7** | Automatic pool/cell reassignment |
| **G8** | Corpus freeze |

## Non-negotiables

- License audit **before** download; provenance ledger row **before** any file enters the corpus
- Generator-disjoint and source-disjoint splits
- Shortcut audit (metadata-only AUC < 0.6) after every corpus change
- Corpus and ledger shippable at all times (2nd-stage materials are due 6 days after the LB closes)
- **Originals are immutable**; all curation lives in sidecar metadata ([10 §1](10-preprocessing-and-filtering.md))
- **No threshold is hardcoded from intuition** — quantile + review pack + your approval ([10 §6](10-preprocessing-and-filtering.md))
