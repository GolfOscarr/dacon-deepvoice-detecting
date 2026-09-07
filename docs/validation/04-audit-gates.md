# 04 — Validation Gates (VG1–VG6)

Automated checks that run with **every** experiment. A red gate voids the result — the number is
not quotable, not comparable, and not promotable ([03 P6](03-decision-protocol.md#3-the-promotion-rule)).

These are distinct from the **human review gates G1–G8** in
[data/10 §5](../data/10-preprocessing-and-filtering.md), which are decision points where we stop
and ask. VG gates are machine checks with pre-committed thresholds.

| Gate | Checks | Threshold | Cost |
|---|---|---|---|
| [**VG1**](#vg1--split-integrity) | Split integrity | zero violations | seconds |
| [**VG2**](#vg2--shortcut-audit) | Metadata-only shortcut | AUC < 0.60 | seconds |
| [**VG3**](#vg3--adversarial-validation) | Train/VAL distribution match | AUC < 0.60 | minutes |
| [**VG4**](#vg4--corpus-identity-leakage) | Corpus-identity shortcut | T3 EER ≤ pooled EER + 10 pts | free (reuses preds) |
| [**VG5**](#vg5--output-sanity) | Output ranking resolution | n_unique > 0.5 × n | free |
| [**VG6**](#vg6--probe-budget) | Sealed-slice budget | ≤3 openings total | free |

---

## VG1 — Split integrity

Pure assertions over `folds.parquet` joined to the [provenance ledger](../data/08-build-plan.md#provenance-ledger).
Cheap, and catches the class of bug that silently inflates every number downstream.

```
A1  artifact_family appears in exactly one of {train, val, probe}
A2  no source_name spans train and val
A3  no speaker_ref_id / artist_id spans train and val
A4  every pair_id's members share a slice
A5  every dup_group is wholly within one slice
A6  PROBE families appear in neither train nor val
A7  every SHADOW row with shadow_kind='a' has shadow_of pointing at a real VAL file_id
A8  every VAL fold meets the per-class size floor (>=1,200) for BOTH masked pools
A9  every cell 1-9 is present in every fold with >=100 samples
A10 scheme_version recorded on the run matches folds.parquet
```

🔴 **A8 and A9 are evaluated against `val_specs.parquet`, not `folds.parquet`.** Both are
statements about *compositions*, and a component row has no cell — the fold table is keyed on
source files ([01 §3](01-split-scheme.md#-the-table-is-keyed-on-components-not-on-composed-files)).
They therefore run at eval-set materialization
([pipelines/02 §4](../pipelines/02-sampler.md#5-the-eval-sampler-is-the-same-code-run-once)), while
A1–A7 and A10 run against `folds.parquet`. The gate is unchanged; only where each assertion can be
computed is.

⚠️ **A8 is the one that will fail first.** The masked pools mean the Voice pool draws only from
cells 1,2,5,6,7,8 and the Music pool only from 3,4,5,6,7,8 — sizing VAL by total file count
under-fills both. See [01 §4](01-split-scheme.md#4-size-floors).

## VG2 — Shortcut audit

The existing [`E-S2`](../data/07-eda-plan.md), promoted to a per-experiment gate.

**Logistic regression on metadata only** — duration, loudness, silence ratio, effective bandwidth,
channel count, container, source bitrate. **No learned audio representation.** Fitted on TRAIN,
evaluated on VAL, reported per head and per cell.

**Gate: AUC < 0.60.** Above that, a confound separates real from fake with no acoustic content,
and every model number is measuring the confound.

🔴 The specific trap here is [09 R1](../data/09-risks-and-checks.md): cells 6/7 can *only* be our
own mixes, so if cell 5 is all natural songs then "artificially mixed" perfectly predicts FAKE.
The [taxonomy rule](../data/02-label-taxonomy.md#-the-composition-trap) — cell 5 must contain
artificial A+C mixes built by the identical code path — exists to prevent it, and VG2 is how we
confirm the prevention worked.

Standing rule from [data/07](../data/07-eda-plan.md): re-run after **every** corpus change, not
only when convenient.

## VG3 — Adversarial validation

The existing [`E-A2`](../data/07-eda-plan.md). A classifier trained to separate TRAIN from VAL.

**Target ≈ 0.5. Gate: AUC < 0.60.** ★ `[G2Net 2021, 3rd]` used exactly this to confirm train/test
similarity. Detects corpus-identity leakage ([09 R2](../data/09-risks-and-checks.md)) and
composition drift between slices.

Reports the ranked discriminating features, which is usually more useful than the AUC — a high
AUC driven by `source_name`-correlated bandwidth points straight at the fix.

⚠️ Interpretation asymmetry: a **high** AUC is decisive evidence of a problem; a **low** AUC is
only weak evidence of its absence, since the adversarial model sees the same metadata features
VG2 does and can miss an acoustic-domain difference. It is a smoke detector, not a proof.

## VG4 — Corpus-identity leakage

[09 R2](../data/09-risks-and-checks.md), operationalized. Real voice comes from LibriTTS, fake
voice from MLAAD — different recording chains entirely. A model can score beautifully by learning
"LibriTTS-ness" and generalize to nothing.

**T3 matched pairs share a chain by construction** — same utterance, one real, one
vocoder/codec-resynthesized ([`S-S1`](../data/05-synthesis-plan.md)). So:

```
EER restricted to T3 pairs   vs   pooled EER on the same VAL fold
```

**Gate: `eer_t3 − eer_pooled ≤ 0.10`.** A large positive gap means the pooled number is carried by
corpus identity, not by artifact detection.

This is the single most informative diagnostic in the set, because it is the only one with a
*matched control*. It costs nothing extra — both numbers come from the same prediction file.

## VG5 — Output sanity

Guards the saturation failure in
[02 §4](02-metric-harness.md#score-saturation-is-the-one-output-formatting-risk-that-matters-),
where collapsing ranking near the operating point took EER from 0.095 to 0.302 with no warning.

```
B1  all 5 columns present, finite, within [0, 1]
B2  n_unique(column) > 0.5 * n_files, per column
B3  no column is constant
B4  row count and ID set match the reference exactly
B5  no NaN produced by the per-file fallback path
```

B2 is satisfied by float64 logits and a float64 sigmoid, with no rounding or clipping anywhere
in the export path. ❌ It must **not** be satisfied by rank-normalizing the column — that is a
cross-file statistic, forbidden by [rule 2.4](../competition/04-rules.md). B5 matters because the submission needs a per-file `try/except` fallback; the fallback
must emit a *finite* value, and a constant fallback across many files will trip B2 honestly.

## VG6 — PROBE budget

The sealed slice ([01 §2](01-split-scheme.md#probe-is-sealed)) is worthless once it has been
optimized against. Enforced mechanically, not by discipline:

- `probe_openings.log` is append-only; each entry records `exp_id`, date, reason, and the result.
- The harness **refuses** to score against PROBE if the log already has 3 entries.
- Opening PROBE requires a stated decision it will inform, written **before** the numbers appear.

Planned spend: first end-to-end model · corpus freeze ([G8](../data/10-preprocessing-and-filtering.md))
· final candidate selection.

---

## Relationship to the human gates

| VG gate | Escalates to | When |
|---|---|---|
| VG1 A8/A9 | — | Fix the sampler; not a judgement call |
| VG2 fail | **G6** (preprocessing policy) or **G4** (salvage parity) | A confound traced to preprocessing |
| VG3 fail | **G7** (pool/cell reassignment) | Slices drifted apart structurally |
| VG4 fail | Corpus composition — [Phase B](../data/08-build-plan.md) | Needs more T-pairs or chain-matched reals |
| VG5 fail | — | Engineering bug |
| VG6 exhausted | **You** | No mechanical remedy; a deliberate decision to spend a 4th opening |
