# 08 — Build Plan

**Today 2026-09-05. Leaderboard closes 2026-09-29 10:00 KST → 24 days.**
2nd-stage materials due 2026-10-05, i.e. 6 days after that — the corpus and ledger must be
shippable at all times, not assembled at the end.

The plan is time-boxed. Data work must finish early enough to leave room for modeling.

## Phases

> **Tier mapping**: Phase 0–1 = all Tier S + the three lead Tier-A items · Phase 2 = remaining
> Tier A · Phase 3 = Tier B and evidence-driven repair · Tier C only if Phase 3 finishes early.
> Catalogs: [05 synthesis](05-synthesis-plan.md) · [06 augmentation](06-augmentation-spec.md) ·
> [07 EDA](07-eda-plan.md).

### Phase 0 — Foundations (days 1–2) 🔴 blocking

| Task | Output |
|---|---|
| **License audit** of every row in [04](04-sources.md) | Ledger with redistribution verdict per source; ❌ rows dropped before download |
| **Dummy-file forensics** ([07 E-S1](07-eda-plan.md)) | Written signal-chain spec → **gate G1** |
| Build the **test-chain normalizer** to that spec | `normalize()` used by every downstream step; policy decisions → **gate G6** |
| Stand up the **curation sidecar layer** ([10 §1](10-preprocessing-and-filtering.md)) | `verdict` / `validity_mask` / `quality` schemas; originals immutable |
| Skeleton **`submit.zip`** with a trivial model | Validates I/O contract, runtime, offline packaging |
| **First LB submission** (constant 0.5 everywhere) | Should score **exactly 0.5000** — confirms our understanding of the metric |

⚠️ Do not download anything before the license audit. A wrong verdict means re-doing the corpus.

### Phase 1 — Minimum viable corpus (days 3–6)

Priority order from [04](04-sources.md#priority-order-if-time-runs-short):

1. MLAAD + LibriTTS-R + Zeroth-Korean → Pools A/B
2. **MUSDB18-HQ** (one download unlocks cells 5–8)
3. ACE-Step generation → Pool D
4. MTG-Jamendo instrumental → Pool C

Plus: T3 vocoder/codec resynthesis of Pool A (cheap, perfectly-paired fakes), and the composition
pipeline from [06](06-augmentation-spec.md).

**Exit criteria**: all 9 cells populated; shortcut audit AUC < 0.6; presence head hits its
[03 acceptance criteria](03-presence-data.md#acceptance-criteria); one real model submitted.

### Phase 2 — Generator breadth (days 7–12)

Add 8–12 TTS families, VC/SVC, SVS, remaining TTM models ([05](05-synthesis-plan.md));
Codecfake; ASVspoof 2021 LA for genuine telephony. Fill Korean and sung-voice slices.
Add the REAL-side processed slice (denoise/enhance/normalize labeled REAL).

**Exit criteria**: ≥20 distinct fake-voice generator families, ≥5 fake-music families, with
25–30% held out unseen for validation.

### Phase 3 — Targeted repair (days 13–18)

Driven by LB probing and per-cell validation. Add data where the evidence says we're weak.
Freeze the corpus at the end of this phase.

### Phase 4 — Freeze & package (days 19–24)

No new data. Modeling, ensembling, runtime tuning. Corpus + ledger + reports packaged for the
2nd-stage submission.

## Volume targets

| Pool | Hours | Notes |
|---|---|---|
| A — real voice | ~70 | incl. ≥10 h sung, ≥5 h Korean |
| B — fake voice | ~70 | **≥20 generator families** |
| C — real instrumental | ~45 | vocal-free verified |
| D — fake instrumental | ~45 | **≥5 generator families** |
| E — non-musical sound | ~10 | |
| **Total** | **~240 h** | ~27.6 GB WAV / ~14–17 GB FLAC |

Composed/augmented samples are generated on the fly and never stored.

## Splits — generator-disjoint, source-disjoint {#splits}

🔴 A random split will report ~0.99 and teach us nothing. The whole task is generalization to
unseen generators ([survey README](../survey/README.md) #6).

| Axis | Rule |
|---|---|
| **Generator** | Assign each generator family to train **or** val, never both. Decide *before* generating. |
| **Source corpus / artist / speaker** | Disjoint across splits — no MUSDB18 track, Jamendo artist, or LibriTTS speaker in both |
| **T3 pairs** | A real file and its resynthesized twin must land in the **same** split |
| **Held-out probe** | A third slice using generator families never seen in train *or* val, opened rarely |

Report validation **per cell** and **per generator**, not just pooled — a good pooled number can
hide a collapsed cell.

## Provenance ledger {#provenance-ledger}

One row per source file, written at acquisition/generation time. Required by DACON's 출처 명시
의무 and the 2nd-stage 학습데이터 구성 보고서 (20 pts).

```
file_id · path · sha256 · pool(A–E) · duration_s · orig_sr · orig_channels
origin_type(dataset|generated|resynthesized) · source_name · source_url · source_version
license · license_url · redistribution_verdict(✅/⚠️/❌) · verified_by · verified_date
generator_model · generator_version · checkpoint_hash · seed · prompt_or_text
speaker_ref_id · sampling_params · postproc_params
label_voice_present · label_music_present · label_voice_fake · label_music_fake
pair_id (T1/T2/T3 twin) · split(train|val|probe)
```

Retrofitting this is impossible for generated audio — emit it at generation time.

## Storage & delivery

- Working: FLAC 16 kHz mono/stereo, ~15 GB
- Delivery: Google Drive (explicitly sanctioned by DACON, #417198)
- Ship: **pools + code + config + seeds**, not composed output ([06](06-augmentation-spec.md))
