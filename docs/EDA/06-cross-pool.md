# 06 — Cross-Pool Analyses

The five pool plans measure *what each pool is*. These measure *what the pools are relative to each
other* — which is where the failure mode that actually kills this kind of project lives. None of
them can be run inside a single pool, and all of them must be **re-run after every corpus change**.

---

## X1 — 🔴 The shortcut audit (E-S2) — Tier S, the blocking gate

**Compute.** A logistic regression on **metadata-only** features — no learned audio representation
— predicting each of the five labels, and each cell, in turn.

Feature set (all from [00 §4](00-harness.md), at both planes):
`duration_s`, `orig_sr`, `orig_channels`, `container`, `codec`, `bitrate`, `lufs_integrated`,
`peak_dbfs`, `crest_factor`, `dc_offset`, `clipping_ratio`, `silence_ratio`, `lead_silence_s`,
`tail_silence_s`, `effective_bandwidth_hz`, `near_nyquist_ratio`, `band_energy[8]`,
`spectral_flatness`, `statistics_T`.

Report **AUC per head and per cell**, plus per-feature coefficients and a permutation importance
ranking.

**Why it is meaningful.** This is the single most important number in the entire EDA. It detects
*"the failure mode that kills this kind of project — a confound separating real/fake with no
acoustic content"* ([data/07](../data/07-eda-plan.md)). Our corpus is built from archives that
were never meant to sit beside each other; A1, B3, C6, D2 each predict a specific confound, and
this is where they sum. ☆ `[LLM-Detect-AI 2024, efficiency prize]` is the cautionary case: adding
off-distribution data *"caused severe data drift which further increased the CV/LB gap."*

**What we discover.** Not just whether a shortcut exists, but **which feature carries it** — the
permutation ranking tells you what to neutralize, and neutralizing the wrong thing is worse than
neutralizing nothing.

**🔴 The gate is AUC < 0.60**, per head, per cell. Above it: neutralize before training anything.
The neutralization is always a **transform applied symmetrically to every pool** (R2) — a codec
round-trip, a resampler draw, a crop policy, a loudness normalization — never a feature deletion,
because the feature will still be there in the audio.

⚠️ **Run it on the population, not on a stratified sample.** Stratifying by `source_name` destroys
exactly the effect being measured ([00 §3](00-harness.md)).

⚠️ **And run it per cell, not only per head.** A confound can be absent marginally and present
inside a presence stratum — the same structure as the marginal-composedness gap that
[data/02](../data/02-label-taxonomy.md) found and fixed: *"`P(composed|FAKE) = P(composed|REAL)` can
be satisfied while composedness still predicts the label inside a presence stratum."* A marginal
AUC of 0.52 is not evidence of safety.

---

## X2 — 🔴 The metadata-leak question (A5) — Tier A, and the one genuinely open feature question

**Compute.** X1's model, but treated as a **candidate submission feature** rather than as a
diagnostic: measure its AUC on our own generator-disjoint, source-disjoint split, then measure it
again after pushing every file through the test-chain normalizer (16 kHz, mono, the containers the
test set uses). Report the drop. Extract the same features from the 3 dummy test files and check
whether they are even *in range*.

**Why it is meaningful.** [architecture/09](../architecture/09-open-questions.md) A5 and
[PROGRESS](../../PROGRESS.md) both flag this as unresolved and decision-changing: per-file metadata
is **legal under rule 2.4** — it is computed from one file, with no cross-file statistics — and may
separate REAL from FAKE almost for free. *"If the leak is real it dominates every architecture
decision."* It may equally be a pure CV mirage: our metadata reflects **our** archives, and DACON
built theirs.

**What we discover.** The size of the mirage in our own corpus, and — the only real evidence
available — whether the dummy files' metadata is consistent with a chain that preserves any of it.
🔴 If DACON normalized every test file to 16 kHz through one pipeline, then container and rate
carry *their* pipeline, not the generator's, and the entire feature is worth zero on the
leaderboard while scoring beautifully in CV.

**⚠️ X1 and X2 are the same measurement read in opposite directions**, and confusing them is easy.
X1 wants the AUC **low** (no shortcut). X2 asks whether a high AUC is exploitable. They cannot both
be satisfied by the same number: **a metadata AUC high enough to be worth shipping is, by
construction, a shortcut that will not generalize.** Resolve it in X1's direction — neutralize —
unless E-S1 produces positive evidence from the dummy files that the test chain preserves the
signal. Record the decision either way; it is the kind of thing that gets silently re-litigated.

---

## X3 — Adversarial validation (E-A2) ⚠️ Tier A

**Compute.** Train a classifier to separate TRAIN from VAL, then TRAIN from a **proxy-eval** slice
built by pushing held-out generators through the test-chain normalizer. Report AUC and the ranked
discriminating features.

**Why it is meaningful.** ★ `[G2Net 2021, 3rd]` used it to confirm train/test similarity at
**AUC ≈ 0.5**. A high AUC here means our own splits differ systematically — corpus-identity
leakage ([data/09](../data/09-risks-and-checks.md) R2) — and that the VAL score is measuring
domain transfer rather than generator transfer.

**What we discover.** Whether artifact-family disjointness has accidentally produced
*source* disjointness too, which would make every VAL number pessimistic in a way that does not
transfer to the leaderboard.

⚠️ We cannot run this against DACON's test set — 3 dummy files. Against our own splits only. That
is a real limitation, and it is why the **shadow split** matters: `shadow_of` / `shadow_b` are the
repo's instrument for domain-shift stress testing and have **never been built**
([PROGRESS](../../PROGRESS.md)). X3 is what they are for.

---

## X4 — 🔴 Dummy-file forensics (E-S1) — Tier S, blocking, and it unblocks the render chain

**Compute.** On `TEST_0000–0002.wav`: `ffprobe` headers and every tag; LTAS with the rolloff shape
at the top of the band; near-Nyquist resampler shelf; LUFS / peak / DC; dither noise floor;
leading/trailing silence; inter-channel correlation. Write `_shared/signal_chain.yaml`.

**Why it is meaningful.** *"The **only** direct evidence of the organizers' signal chain"*
([data/07](../data/07-eda-plan.md)). Everything symmetric depends on reading it right: a rolloff
near ~3.4 kHz would reveal how they represent 전화채널; the presence or absence of loudness
normalization decides P-A1; the silence policy decides P-A2.

**What it changes — immediately.** 🔴 `normalize` in the render path is **unparameterized until
this lands**, so A-S1, the highest-leverage augmentation step, is *structurally present and doing
nothing today*. Three files, an afternoon, and it turns a no-op back into the step it was designed
to be.

⚠️ **Three files is three files.** Anything read from them is an `n = 3` inference, marked ☆ at
best. Treat the signal chain as a hypothesis with a confidence, not as ground truth, and prefer
policies that are *robust* to getting it wrong over policies that are optimal if it is right.

---

## X5 — The composition confound, over the spec stream ⚠️ Tier S

**Compute.** Run `training.audit.audit_specs` against the real corpus once it is built, and read
off the per-cell composed fraction `f_c`, the marginal `P(composed | FILE_FAKE)`, and the
within-presence-stratum version. No audio decode is required — `sample_spec()` is pure and
label-free, so auditing a stream costs nothing.

**Why it is meaningful.** [data/02](../data/02-label-taxonomy.md)'s composition trap is the one
confound this project has already solved *in design* — `f8`, `balance_marginal_composedness`, and
the C1/C3 checks in `SamplerConfig.__post_init__`. But the reference cell mix *was itself trapped
when first published* (`P(mixed|FAKE)=0.667` vs `P(mixed|REAL)=0.300`, so "is a mixed file"
predicted FAKE at 0.769), and the shipped mix was solved against a **synthetic** corpus whose pool
sizes differ from the real one.

**What we discover.** Whether the solved mix survives contact with the real pool sizes. It may not:
`f_c` equality requires enough whole-file material in cells 5 and 8 to match the composed fraction,
and [C1](03-pool-c-real-instrumental.md)'s licence intersection may leave too little real music to
do it with.

**⚠️ Do not read this as optional because the code enforces it.** The enforcement is on the *mix*,
not on whether the pool can supply it — a mix that satisfies C1 and C3 and cannot be drawn from the
corpus fails later, in `build_folds`, with a message about fold feasibility rather than about
composedness.

---

## X6 — Metadata role assignment (E-S4) ⚠️ Tier S

**Compute.** Every column in `eda/*/files.parquet` assigned to exactly one of **feature** /
**split key** / **leakage risk**, with missingness and cardinality quantified. Freeze it as
`_shared/roles.md`.

**Why it is meaningful.** ★ `[BirdCLEF playbook 2026]`. A column used as a feature and as a split
key at the same time is a leak by construction; a column with 90% missingness used as a split key
silently collapses its groups. This is what determines the grouping keys *before* any fold is
built, and [data/12](../data/12-acquisition-status.md) records what happens when it is decided
late: a `source_name` at the wrong granularity validates clean and makes `build_folds` infeasible
at every fold count.

**What we discover.** The frozen list, and the cardinality of every proposed grouping atom —
which is [A3](01-pool-a-real-voice.md), [B2](02-pool-b-fake-voice.md), [C3](03-pool-c-real-instrumental.md),
[D1](04-pool-d-fake-instrumental.md) and [E4](05-pool-e-noise.md) collected into one table that
`build_folds` can be run against before a single file is copied.

---

## X7 — The data memo (E-S3) ⚠️ Tier S, the exit condition

**Compute.** One page: file inventory, schema, label counts per cell, suspected leakage variables,
risk list — with the numbers from X1–X6 inline.

**Why it is meaningful.** ★ `[BirdCLEF playbook 2026]`: *"Do not advance to hyperparameter tuning
until the data memo explains class imbalance, domain shift, and the first leakage hypothesis."*
It is the artifact that says the EDA is finished, and the three things it must explain are exactly
the three this corpus is most at risk from.

**Output.** `docs/EDA/data_memo.md`, versioned, regenerated on every corpus change.

---

## The standing rule

🔴 **Re-run X1 (shortcut audit) and X3 (adversarial validation) after *every* corpus change.**
They are cheap, they take minutes against a metadata table that already exists, and they catch most
of the failure modes in [data/09](../data/09-risks-and-checks.md) before those cost weeks.
