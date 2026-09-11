# 09 — Open Questions, Probes, and the Verification Backlog

What is unresolved in this directory, who resolves it, and what changes if it goes either way.

Grouped by whether it **blocks a decision** (A), **improves a decision** (B), or is a
**verification chore that could invalidate work** (C).

---

## A. Probes that gate a design decision

### A1 — 🔴 The 16 kHz survivability probe (`E-A1`)

**Question.** Does the residual-concentration mechanism that separates AI music from human music
survive an 8 kHz Nyquist?

**Why it is genuinely open.** ArtifactNet's Table XI reports effective residual bandwidth of
**~291 Hz for AI generators** (Suno v3.5 170, Riffusion 219, Stable Audio 237, Udio 245,
MusicGen 255) versus **~1,996 Hz for human music** — a 6.9× separation at numbers that sit
comfortably inside our 0–8 kHz range. Yet the same paper insists on 44.1 kHz and states that
16 kHz *"attenuates the forensic signal"*, and the fakeprint paper **excludes 16 kHz data as
unsuitable** ([papers/02](../papers/02-music-detection.md)).

🔷 The likely resolution is that *bandwidth* there measures the residual's spectral
**concentration**, not its **location** — a narrow band sitting high in the spectrum. We could not
settle it from the text.

⚠️ A contradiction to resolve alongside it: `lofcz/ai-music-detector` claims to implement fakeprint
**at 16 kHz**, while the fakeprint paper says that is unsuitable. One of the two is wrong.

**What changes.** If the mechanism survives: promote candidate **E**, the music branch gains a
forensic-residual channel. If not: the music branch rests entirely on a learned general-audio
representation. **Worth 0.27 of the score**, and these are different architectures.

**Owner.** Phase B EDA ([data/07](../data/07-eda-plan.md)). Blocks nothing — B is buildable either
way — but it should be answered in the first days.

### A2 — 🔴 Is Mamba installable at all?

`pip download mamba-ssm causal-conv1d` against `torch 2.7.1+cu128 / py3.11 / CUDA 12.8`. One
command.

If a prebuilt wheel exists, the best published speech backbones re-enter the menu
(Fake-Mamba: **5.85% ITW EER**). If not, they are permanently out and the AASIST family is the
answer ([01 §1.1](01-design-envelope.md#11-the-10-minute-pip-budget-)). This is **V5** in
[survey/10](../survey/10-open-questions.md).

⚠️ Cheap to run, expensive to discover late. An install error does not consume a daily submission,
but a late architecture change does consume days.

### A3 — 🔴 Runtime measurement

[07 §4](07-runtime-budget.md#4-the-measurement-protocol-) in full. Until it runs, every number in
[07 §2](07-runtime-budget.md) is an extrapolation, and the decision rule in
[05 §7](05-multi-model.md#7-the-decision-rule) has no input.

🔷 Our prediction, recorded so it can be scored: **the 6 vCPU decode path, not the GPU, is the
binding constraint.** If that is wrong, the ensemble budget is larger than we assumed.

### A5 — 🔴 Per-file metadata: real signal, or CV-only mirage?

Container, bitrate, channel count, duration, encoder fingerprint, spectral cutoff. All legal under
rule 2.4 ([01 §3.5](01-design-envelope.md#35--per-file-metadata-fully-legal-possibly-decisive-possibly-a-mirage)).

**Why it is urgent rather than interesting.** If the organizers assembled REAL and FAKE from
different pipelines, metadata may separate the classes almost for free — and if so it **dominates
every other decision in this directory**. If it does not, a metadata feature that shines on our own
corpus is a pure mirage, because our corpus carries *our* pipeline's fingerprints.

**The only direct evidence we will ever have is the 3 dummy files** — which is `G1` /`E-S1` dummy
forensics ([data/07](../data/07-eda-plan.md#tier-s)), already listed as blocking in
[`PROGRESS.md`](../../PROGRESS.md). This is a second, independent reason to run it on day one.

⚠️ Note what we cannot do: adversarial validation against the test set, because we have 3 files
([kaggle/05](../kaggle/05-transferable-playbook.md)).

### A6 — 🔴 Which music frontend, and how deep? (was B2/B3)

**Promoted out of section B.** The music branch carries **0.27 directly plus most of the 0.45 file
head**, and it is simultaneously the least-settled thing in this directory: the frontend is unchosen
(SSLAM / EAT / BEATs / MERT), its licences are unverified (C2–C3 below), its truncation depth is
unmeasured, its published cross-generator baseline is **46.4% EER**, and fold construction is
blocked at 5 families (C6).

🔷 Nothing else in the plan re-prioritises to reflect that **the highest-weight, highest-uncertainty
branch is the least evidenced.** This entry is that correction. Two sub-questions:

- **Which encoder at 16 kHz** (G2) — ⚠️ MERT is trained at 24 kHz and untested at 16, and its only
  appearance in our music literature is as the backbone of the 46.4% failure.
- **Truncation depth, and whether the probed layers are early** — the speech result may not
  transfer, and layer *selection* is not layer *truncation* ([06 §1](06-compression.md)).

### A7 — 🔴 Does PANNs fire `VOICE_PRESENT` on sung vocals?

**One song answers it**, and it is worth an A-slot because it decides whether our day-one baseline
is valid on the most common mixed case. AudioSet tagging conflates singing with `Music` and does not
reliably fire `Speech` on sung vocals, while the rules class vocals as **voice**
([02 §C](02-pretrained-catalog.md#-sung-vocals-are-the-failure-case-and-panns-is-likely-bad-at-it)).

### A4 — `FILE_FAKE_PROB` construction (G3)

Learned branch vs analytic noisy-OR vs `max`. No prior art — no other benchmark has this label
structure. Argued in [04 §5](04-heads-and-pooling.md#5-the-file-head); 🔷 we expect the learned branch
to win *because* ranking training does not produce the calibrated components the analytic forms
need, and 🔷 we expect the gap to be largest on mixed audio. **Worth 0.45 — build all three.**

---

## B. Sweeps that improve a decision

🔴 **Read the "Resolvable?" column before queuing any of these.**
[`validation/`](../validation/README.md) puts the LB noise floor at **±1.7 pts EER** on the File
head and **±2.5** on each component head, and says a VAL pool needs **≥1,200 per class per masked
pool** to resolve a 1-point gap at 95%. Several questions below have expected effects *below that
floor*. On a corpus that does not exist yet, in 22 days, **a question we cannot resolve is not an
experiment — it is a decision to make by argument and leave alone.**

| # | Question | Cost | Resolvable? | Note |
|---|---|---|---|---|
| **B1** | Which speech SSL frontend? | one read + one probe | ✅ | ☆ *A SUPERB-Style Benchmark of SSL Models for ADD* (2026) is a leaderboard for exactly this and is only card-level in our index. **Read it before finalizing** |
| **B4** | Does the joint-training gain survive **frozen frontends + adapters**? | ablation | ✅ effect is large in the source papers | ⚠️ PC-Mix and CompSpoof co-adapted full branches. This is an assumption our design rests on ([06 §2](06-compression.md)) |
| **B5** | 🔴 **Tile, or run the whole file in one pass?** | one benchmark | ✅ it is a runtime + structure question, not an accuracy delta | Whole-file deletes the cross-window aggregation bias entirely at ~1.6× encoder cost ([04 §6](04-heads-and-pooling.md#6-temporal-coverage-tiling-not-sampling)). The 5 s window is inherited from a notebook whose clips were 5 s — an assumption, not a requirement |
| **B7** | Low-band-only input (0–4 kHz) as a model or a member | ablation | ✅ | ★ D9: EER down **up to 25% relative** under codecs; our telephone slice lives there |
| **B8** | Stereo: downmix, or use both channels? | ablation | ✅ as a *leak* test | ⚠️ Mid/side may carry cues — **or may be a shortcut** if fake sources skew mono. Test as a leak first, a feature second |
| **B9** | 🔴 Does distillation still work with a **frozen** frontend? | ablation | ✅ | The borrowed recipe's mechanism is handing the *backbone* to the distillation loss; freezing removes it ([06 §3](06-compression.md)) |
| **B10** | Cross-window / whole-file aggregator, and its **duration bias** | ablation + new slice | ✅ as a *bias* measurement, not an EER delta | Needs a binned `duration` stratum in [`metrics/breakdown.py`](../../metrics/breakdown.py), which does not exist yet |
| **B6** | Pooling function per head (G5) | free — GeM learns it | n/a — no experiment | 🔷 The learned `p` values are a reportable finding, not a decision we make |
| **B5b** | Window length and mel/STFT params, *if* we tile | small grid | ⚠️ **probably not** — sub-floor effects | ★ *"Keep this grid small."* Pick defaults by argument; do not sweep |
| **B11** | ✅ **CLOSED** — component loss weights | cheap | ⚠️ **no**, as predicted | **Resolved by argument, metric-proportional.** A targeted search found *no* study testing loss-weights against evaluation-metric weights, and every adaptive alternative (GradNorm/PCGrad/DWA/uncertainty) has strong negative results against tuned constants ([training/02 §4](../training/02-the-loss.md#4-per-head-weights--metric-proportional-by-argument)) |

⚠️ Two entries moved **out** of this section because they gate a decision rather than improve one —
they are now **A6** and **A7** below.

## C. Verification that could invalidate work

| # | Item | Risk if wrong |
|---|---|---|
| **C1** | ⚠️ **V3** — W2V-BERT 2.0 licence (Seamless components vary; some CC-BY-NC) | It is the AT-ADD Track 1 winner's frontend. An ND-style term makes it unusable *at all*, not merely unshippable. **Still unanswered, and no longer blocking: candidate A now runs on BEATs** |
| **C2** | ⚠️ **V4** — MERT licence, and its behaviour at 16 kHz | Our only dedicated music encoder |
| **C3** | ⚠️ SSLAM / EAT licences | Both are our preferred music frontends. **BEATs (MIT) is the floor if either fails** — and the floor is now wired and trained against (MIT read at origin), so C1–C3 gate the *upside*, not the ability to train |
| **C4** | ⚠️ **ArtifactNet patents (KR + PCT)** on bounded-mask residual extraction and codec-invariant training | 🔴 Reimplementing that specific formulation in a Korean government competition whose rules assign winning-work copyright to the host may warrant a legal look — or we avoid that exact formulation |
| **C5** | ⚠️ Version skew: local 3.12 / numpy 2.5.3 / pandas 3.0.5 vs server 3.11.15 / 1.26.4 / 2.0.3 | Discovered at packaging time, it costs days. **Narrowed**: the training venv is now CPython **3.11.15**, the server's exact version, with torch 2.7.1+cu128 |
| **C6** | 🔴 **Music-family shortfall** — ≥8 families needed, 5 planned | No music-branch result is trustworthy until resolved. Blocks fold construction ([validation/01](../validation/01-split-scheme.md)). **Confirmed empirically**: the smoke corpus has 4, `build_folds` refused 5/4/3 folds outright, and at 2 folds `I21` still FAILs on the VAL side |
| **C7** | 🔴 **Pool D has no acquisition path at all** — fake *music* exists in no acquired or queued source | The 0.27 music head has nothing real to learn from. CompSpoof V2 is **not** the answer: its second component is environmental sound, not music. The smoke corpus substitutes spoofed environmental audio as an explicit placeholder, and the leak tripwire fires on it unaided (music EER 0.0201 vs a 0.03 floor) |

---

## D. Papers worth promoting to a deep read

Card-level in [`docs/papers/INDEX.md`](../papers/INDEX.md), and each one bears on a decision made
in this directory:

| Paper | Why now |
|---|---|
| ☆ **A SUPERB-Style Benchmark of SSL Models for ADD** (2026) | Directly picks our frontend (B1) |
| ☆ **Probing-Guided Layer Selection** (2025) | The 4-layer result is load-bearing for the entire inference budget ([06 §1](06-compression.md)) |
| ☆ **Interpreting Multi-Branch Anti-Spoofing Architectures** (2026) | Literally about what multi-branch models key on — our design, and the 15-point 결과 해석 section |
| ☆ **Ensemble Learning for AUC Maximization via Surrogate Loss** | The combiner should optimize our metric, not log-loss ([05 §6](05-multi-model.md)) |
| ☆ **Explainable-by-Design via Wiener-Hopf Linear Prediction** (2026) | Claims recovery under noise, MP3 **and telephone filtering** — our exact degradation set, and interpretable by construction |
| ⚠️ **DK-CAST / FTDKD** | Marked *unverified, locate before citing* in [papers/07](../papers/07-foundation-distillation.md). Either verify or stop referencing |

---

## E. What we would be wrong about

🔷 Recorded so that being wrong is cheap to notice. Each of these is a belief this directory rests
on that we have **not** measured.

| Belief | How we would find out it is false |
|---|---|
| A hard switch is undefined on 혼합 files, so routing is out | Someone specifies sane switch behaviour on mixed audio that beats B on our CV. (The *earlier* rejection argument — unabsorbable score-scale mismatch — has already been withdrawn as wrong) |
| C-lite's auxiliary separation head transfers CompSpoof's joint-learning gain | It does not: the gain needs the separator *in* the inference path, and the auxiliary head shapes nothing useful |
| A fixed-k top-k window aggregator avoids the duration bias | Duration still correlates with score on the REAL class in the `E-S2` audit |
| Two specialist frontends beat one shared trunk | Candidate A matches B at a third of the cost. That is a legitimate result and belongs in the report |
| The decode path binds before the GPU | [07 §4](07-runtime-budget.md#4-the-measurement-protocol-) says otherwise |
| Layer truncation transfers to the music frontend | B3 probe shows the music head needs deep layers, and the two-frontend budget collapses |
| The probed layers are *early*, so selection implies truncation | They are not — layer selection and truncation are different techniques, and only the second saves time ([06 §1](06-compression.md)) |
| Speech-measured techniques transfer to the music head | They do not, on the head worth 0.27 plus most of 0.45 ([02](02-pretrained-catalog.md#-the-evidence-asymmetry-nobody-should-forget)) |
| Tiling is the right coverage strategy | B5 shows whole-file is affordable, and §6.1's entire aggregation problem was self-inflicted |
| Distillation recovers what truncation costs | B9 — with a frozen frontend there may be nothing for it to recover *into* |
| The learned file head beats the analytic noisy-OR | A4. If noisy-OR wins, our calibration argument is wrong somewhere |
| Frozen frontends + adapters preserve the joint-training gain | B4 |
