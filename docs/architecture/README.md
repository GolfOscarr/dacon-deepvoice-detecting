# Architecture — Index

How we build the model. Phase **D** of [PROGRESS.md](../../PROGRESS.md).

Written 2026-09-07, after the survey ([`docs/survey/`](../survey/README.md)), the paper reads
([`docs/papers/`](../papers/INDEX.md)) and the Kaggle sweep ([`docs/kaggle/`](../kaggle/README.md))
were complete. **Nothing here has been trained yet.** These are candidates, constraints, and the
reasoning that ranks them — not results.

| File | Use it when you need… |
|---|---|
| [01-design-envelope.md](01-design-envelope.md) | What the rules, the L4, and the metric permit — the filter every candidate passes through |
| [02-pretrained-catalog.md](02-pretrained-catalog.md) | **Which pretrained models we can adopt**, per role, with licence / size / runtime / verdict |
| [03-candidates.md](03-candidates.md) | The whole-system candidates A–G, ranked, and which one we build |
| [04-heads-and-pooling.md](04-heads-and-pooling.md) | Branch heads, SED attention, pooling, the `FILE_FAKE` construction, the output contract |
| [05-multi-model.md](05-multi-model.md) | **Combining several models** — fusion taxonomy, what rule 2.4 permits, ranking-aware combiners |
| [06-compression.md](06-compression.md) | Train-big-ship-small: layer truncation, distillation, adapters, precision |
| [07-runtime-budget.md](07-runtime-budget.md) | The L4 inference budget, line by line, and how to measure it |
| [08-training-recipe.md](08-training-recipe.md) | Joint training schedule, losses, what runs on the H200s |
| [09-open-questions.md](09-open-questions.md) | Probes that gate design decisions, and the verification backlog |

Confidence marks, as elsewhere in this repo: ★ verified from a primary source · ☆ secondary,
re-verify · ⚠️ risk · 🔴 decision-changing · ❌ forbidden by competition rules.

One additional convention used only in this directory, because much of it is reasoning rather
than citation: **🔷 = our own inference, not a published result.** Anything marked 🔷 is a claim
we derived and have not tested. Treat it as a hypothesis with an owner, not as evidence.

---

## The governing fact

**Training compute is free; inference compute is scarce.**

We train on a single node of **8×H200**. We ship to **one L4** with a 60-minute wall clock over
1,200 files — roughly **10× real-time** for ~10 hours of audio, including decode
([07](07-runtime-budget.md)).

⚠️ **But "free" applies to H200-hours, not to engineer-days.** The corpus build has not started and
takes days 3–12 of 22, leaving ~10 days for all modelling — against six distinct training
programmes. That is the largest un-modelled risk here, and
[08 §4b](08-training-recipe.md#4b--the-budget-nobody-costed-engineer-days) states the drop order
rather than leaving it to be discovered late. **Candidate A is the rung that must always work.**

That asymmetry is the most important input to the design, and it inverts the usual advice. Every
large model becomes a **teacher that never enters the inference path** rather than a component we
cannot afford. Distillation and layer truncation are not optimizations bolted on at the end; they
are the mechanism by which foundation-model quality reaches an L4 at all ([06](06-compression.md)).

## The decisions, and where they are argued

| Decision | Choice | Argued in |
|---|---|---|
| Overall layout | **Three branches on the mixture, jointly trained** (PC-Mix layout) | [03](03-candidates.md) |
| Build order | 🔴 **A is stage one of B, not a fallback** — B strictly contains A; the increment is one encoder, adapters and a time-base alignment | [03](03-candidates.md#-a-is-stage-one-of-b-not-a-fallback) |
| Promoting B over A | Pre-committed **P1–P6**; on an inconclusive result the tiebreaker favours **A** | [validation/03 §3](../validation/03-decision-protocol.md) |
| Separation in the inference path | **No** — four independent grounds against | [03](03-candidates.md#g--frozen-separator--per-stem-detectors--rejected) |
| Frontends | **Two specialists**, truncated: speech SSL for voice, general-audio SSL for music | [02](02-pretrained-catalog.md) |
| Why two and not one | CompSpoof: one encoder serves voice and music unequally | [02](02-pretrained-catalog.md#the-case-for-two-frontends) |
| Why truncated | 4 probed layers of XLS-R-300M match the full model at 1.34M trainable params | [06](06-compression.md) |
| Head per branch | SED attention head: GeM freq pool → attention over time → clip + framewise | [04](04-heads-and-pooling.md) |
| `FILE_FAKE_PROB` | **Learned third branch** over both frontends; analytic noisy-OR as the baseline | [04](04-heads-and-pooling.md#5-the-file-head) |
| Presence heads | Two more SED heads on the audio branch; **PANNs is the day-1 shippable baseline** | [02](02-pretrained-catalog.md#c-presence-heads) |
| Combining models | **Distil an ensemble into one student**; run a small inference ensemble only if measured headroom allows | [05](05-multi-model.md) |
| Hard type routing | ❌ **Rejected** — AT-ADD's four types are mutually exclusive; our 혼합 class is not, so a switch is undefined on mixed files | [03](03-candidates.md#d--hard-type-routing--rejected) |
| Separation at training only | ⭐ **C-lite** — auxiliary separation head, discarded before packaging. Try right after B works | [03](03-candidates.md#c-lite--separation-as-an-auxiliary-training-head--live) |

## Corrections already made to this directory

Kept visible rather than silently edited, because the reasoning is the deliverable:

- 🔴 **The rejection of hard type routing was argued wrongly at first.** The original argument —
  that nothing absorbs the score-scale mismatch between subsystems — is refuted by
  [05 §1](05-multi-model.md), which shows offline per-subsystem calibration is both legal and
  sufficient. The conclusion survives on a different and stronger basis: **overlapping classes**
  ([01 §3.4](01-design-envelope.md#34--overlapping-classes-and-one-pooled-ranking)).
- 🔴 **The prescribed inference blend defeated our own saturation finding** — two sigmoids averaged,
  when the measured cost of saturation is EER 0.0950 → 0.3017. Now blended in logit space
  ([04 §3](04-heads-and-pooling.md#3-the-loss-clip--frame-max-and-why-it-matches-our-labels)).
- 🔴 **`P0` was one submission and had to become two.** A PANNs-presence baseline with constant fake
  columns scores `0.45 + 0.1·CPS ≈ 0.535`, **not** 0.5000 — so shipping it as the contract probe
  would have tripped [validation/05](../validation/05-lb-probe-plan.md)'s "stop if not 0.5000" rule
  on a correct submission ([03 P0](03-candidates.md#p0--the-baselines-that-are-not-models)).
- 🔴 **A claim about noisy-OR was mathematically false.** `1 − (1−a)(1−b)` *is* strictly increasing
  in each argument; the true, weaker property is that the order it induces over pairs is not
  invariant to independent monotone reparametrisation
  ([04 §5](04-heads-and-pooling.md#5-the-file-head)).
- 🔴 **Freezing the frontends removes the mechanism the borrowed distillation recipe relies on**
  ([06 §3](06-compression.md)) — now stated as an open choice rather than an inherited result.
- **Tiling may be unnecessary.** The 5 s window was inherited from a notebook whose clips were 5 s;
  running the whole file in one pass costs ~1.6× and deletes the duration-bias problem outright
  ([04 §6](04-heads-and-pooling.md#6-temporal-coverage-tiling-not-sampling)).
- 🔴 **The runtime table double-counted the tiling factor**, understating the margin by 3×. Full
  tiling covers the file exactly once, so it does not reduce throughput measured against file
  duration. Corrected margin: **~3.7–10×**, not ~1.25–3.3×
  ([07 §2](07-runtime-budget.md#2-estimated-line-items-)) — which loosens the ensemble decision.
- 🔴 **"Probed-layer selection" is not "truncation".** The cited result says *which* layers feed the
  head, not that they are early. If the useful layers are deep, the 3–6× saving — and the
  affordability of two frontends — evaporates ([06 §1](06-compression.md)).
- 🔴 **We predicted the decode path would be the bottleneck. It almost certainly is not** — the test
  set is already 16 kHz so no resampling is needed, and 20 h decodes in single-digit minutes across
  6 cores. The claim had spread to three files ([07 §3](07-runtime-budget.md#3-the-cpu-side)).
- **Per-file metadata was missing entirely** as a design consideration — legal under rule 2.4,
  potentially decisive, and equally potentially a CV-only mirage
  ([01 §3.5](01-design-envelope.md#35--per-file-metadata-fully-legal-possibly-decisive-possibly-a-mirage)).
- 🔴 **The day-one presence baseline probably fails on songs.** Vocals count as *voice*, so a song
  must fire both presence heads — and AudioSet tagging conflates singing with `Music` without
  reliably firing `Speech` ([02 §C](02-pretrained-catalog.md#-sung-vocals-are-the-failure-case-and-panns-is-likely-bad-at-it)).
- 🔴 **Engineer-days were never budgeted**, only H200-hours. Six training programmes against ~10
  days of modelling time; now a stated drop order
  ([08 §4b](08-training-recipe.md#4b--the-budget-nobody-costed-engineer-days)).
- **The experiment backlog outran the measurement floor.** LB noise is ±1.7/±2.5 pts and several
  queued sweeps have smaller expected effects; [09 B](09-open-questions.md) now carries a
  "Resolvable?" column, and three entries are marked *decide by argument, do not sweep*.
- **"Frame-level ground truth is free" is true only of composed files** — i.e. exactly the subset
  where the composition shortcut lives, which makes the frame-loss weight a shortcut-exposure knob
  ([04 §4](04-heads-and-pooling.md#4-multi-resolution-supervision--an-advantage-we-can-produce-but-should-not-lean-on)).
- Several claims carried ★ (primary-source) marks when their sources are card-level entries in
  [papers/INDEX](../papers/INDEX.md) — including the **layer-truncation result the whole inference
  budget rests on**. All downgraded to ☆. Two model-size figures and one blockquote were our own
  arithmetic or our own summary presented as sourced; now marked 🔷.

## The two things that could still reorder this

1. 🔴 **`E-A1`, the 16 kHz survivability probe.** Decides whether the music branch gets a
   forensic-residual front-end or a purely learned one. Worth 0.27 of the score.
   [09](09-open-questions.md#a1---the-16-khz-survivability-probe-e-a1)
2. 🔴 **Whether Mamba is installable at all.** One `pip download` settles whether the best
   published speech backbones exist for us. [09](09-open-questions.md#a2---is-mamba-installable-at-all)

Neither blocks starting. Both should be answered in the first two days.

## Reading order

If you read three files, read [01](01-design-envelope.md), [03](03-candidates.md) and
[07](07-runtime-budget.md) — the envelope, the choice, and the budget that decides whether the
choice is real. [02](02-pretrained-catalog.md) and [05](05-multi-model.md) are lookup tables;
go to them when you need to pick a component or a fusion rule.
