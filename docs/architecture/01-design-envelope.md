# 01 — The Design Envelope

Every candidate in [03](03-candidates.md) is filtered through this file first. The sections are
ordered by how hard the constraint is: §1 eliminates designs outright, §2 sets the compute
ceiling, §3 shapes what the model must output and how it must be trained.

---

## 1. Hard gates (a design that fails one of these is out)

### 1.1 The 10-minute pip budget 🔴

`requirements.txt` must install in **≤10 min**, with no internet beyond pip, against
`torch 2.7.1+cu128 / Python 3.11.15 / CUDA 12.8` ([competition/02](../competition/02-submission.md)).

⚠️ **This is the gate that most likely costs us the published SOTA backbone.** `mamba-ssm` and
`causal-conv1d` compile CUDA kernels at install time. Unless a prebuilt wheel exists for that
exact triple, the entire Mamba family is unavailable:

| System | Best published EER | Status |
|---|---|---|
| ★ Fake-Mamba (ASRU 2025) | 0.97% / 1.74% / **5.85%** (21LA / 21DF / ITW) | ⚠️ gated on V5 |
| ★ XLSR-Mamba | 6.71% ITW | ⚠️ gated on V5 |
| ★ XLSR-MamBo (ACL 2026) | — | ⚠️ gated on V5 |
| AASIST / attention backends | 0.83% (19LA) | ✅ pure PyTorch, no gate |

An install error does **not** consume one of our 3 daily submissions, so the failure is cheap to
discover on the server — but expensive to discover late. Settle it with `pip download` on day one
([09 A2](09-open-questions.md#a2---is-mamba-installable-at-all)). This is item **V5** in
[survey/10](../survey/10-open-questions.md).

### 1.2 Everything loads offline from `model/`

The eval runtime has **no internet**. Anything that lazily fetches at runtime fails:
`torch.hub.load` (which is how the Silero VAD recipe in
[kaggle/06 §9](../kaggle/06-notebook-code.md) is written), `from_pretrained` hitting the hub,
`panns_inference` fetching its checkpoint, `demucs` fetching its models. Each needs a vendored
cache inside the zip plus `HF_HOME` / `TORCH_HOME` / `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.

**Architectural consequence:** every additional pretrained component carries a fixed packaging
cost and a fixed failure mode, independent of its accuracy. Prefer few, large-value components
over many small ones.

Zip ≤ **10 GB** (≤32 GB uncompressed). 🔷 For scale — params × 4 bytes, our arithmetic, not a
figure from any source doc: XLS-R-300M ≈ 1.2 GB fp32 / 0.6 GB fp16; XLS-R-1B ≈ 4 GB fp32;
XLS-R-2B ≈ 8 GB fp32. ⚠️ Verify against the actual checkpoints before planning around them. A multi-model ensemble of large frontends runs into
this ceiling before it runs into the runtime ceiling ([05](05-multi-model.md)).

### 1.3 Shipped weights are part of the deliverable 🔴

Rule 3 requires finalists to hand DACON the training code, two HWP reports, **the actual training
data files**, and member info ([competition/04 §3](../competition/04-rules.md)) — it does *not*
mention inspecting weights. The "weights we ship are part of the deliverable" framing comes from
[survey/05](../survey/05-models.md), and rule 2.1 makes licence compliance for every pretrained
model our responsibility either way. Official answers confirm **CC-BY-NC and CC-BY-NC-SA are
usable**, and that third-party licences survive submission unchanged
([competition/04 §2.1, §6](../competition/04-rules.md)).

⚠️ But a licence that **forbids third-party provision** makes an asset unusable *at all* here —
not merely unshippable. **ND terms are the specific danger.** This is a licence filter on the
frontend menu, applied in [02](02-pretrained-catalog.md), and it is why every role there carries a
named licence-safe option that we can commit to without waiting for a verification answer —
BEATs (MIT) for the general-audio role, and XLS-R-300M ahead of the unverified W2V-BERT 2.0 for the
speech role, despite the latter being the AT-ADD Track 1 winner's frontend.

### 1.4 Rule 2.4 — per-file independence ❌

> 비공개 평가 데이터의 각 파일 샘플은 **서로 독립적으로 예측**해야 합니다.

| Permitted | Forbidden |
|---|---|
| ✅ Segment a file, infer per segment, pool within the file — **explicitly encouraged** | ❌ Any statistic computed across test files |
| ✅ Per-file instance normalization (uses only that file) | ❌ Score standardization / rank calibration over the cohort |
| ✅ Within-file temporal smoothing ([kaggle/06 §4](../kaggle/06-notebook-code.md)) | ❌ Adaptive thresholds, clustering, batch statistics |

⚠️ Accidental violations are the real risk: BatchNorm left in train mode, a normalizer fitted on
the test set, or any behaviour that depends on how files are batched. **Inference must be a pure
function of one file** — which also means the inference batch size must not change any output.

🔷 A useful test we can run ourselves: score the same file alone and inside a batch of 32, and
compare. ⚠️ **Do not assert bitwise equality** — kernel selection varies with batch shape and padded
attention changes reduction order, so a *correct* implementation fails it, the test gets disabled,
and the real rule-2.4 check goes with it. Assert instead: agreement to a tight tolerance, **and**
that the induced ranking over a canned set is unchanged. Ranking is what the metric reads.

### 1.5 Rule 2.3 — no test-time adaptation ❌

No pseudo-labeling, no test-time training, no entropy minimization, no BN-statistic adaptation on
the eval set. A large family of domain-adaptation techniques is unavailable. The legal substitute
is **cross-domain MixUp at training time** ([kaggle/06 §6](../kaggle/06-notebook-code.md)) — mixing
our own clean synthetic audio with real degraded audio so the model sees synthetic content in
deployment-like acoustic context.

### 1.6 A silent failure is worse than a crash ⚠️

Our own per-file try/except fallback will happily write 1,200 rows of 0.5 and score **exactly
0.5000** without raising ([competition/02](../competition/02-submission.md#-guard-against-a-silently-broken-submission)).
The architecture must therefore expose a **canned-input fingerprint** — a fixed tensor whose
output is asserted at startup — which means the model's forward pass has to be deterministic in
eval mode. No dropout-at-inference, no MC sampling, unless seeded and asserted.

---

## 2. The compute ceiling

| | |
|---|---|
| Eval hardware | 1× **NVIDIA L4**, 22.4 GiB VRAM · **6 vCPU** · 28 GB RAM |
| Wall clock | **60 min** for 1,200 files, 4–60 s each, 16 kHz |
| Working budget | ~10 h of audio in 60 min ≈ **10× real-time**, decode included |
| Training hardware | 1 node × **8×H200** |

The ratio between those last two rows is the design's central fact. It is developed in
[06](06-compression.md) and accounted line-by-line in [07](07-runtime-budget.md).

Two consequences that are easy to state and easy to forget:

- **VRAM is not the constraint.** 22.4 GiB is ample for anything we can afford to *run*. Do not
  design around memory; design around wall clock.
- ⚠️ **The 6 vCPU decode path needs measuring, but is probably not the constraint.** An earlier
  draft of this file predicted it would bind. The arithmetic says otherwise: **no resampling is
  needed** — the test set is already 16 kHz — and 20 h of audio decodes in single-digit minutes
  across 6 cores. It stays on the measurement list as a place implementation bugs hide, not as a
  structural limit ([07 §3](07-runtime-budget.md#3-the-cpu-side)).

---

## 3. What the metric does to the architecture

### 3.1 Effective weights

| Head | Metric | Weight |
|---|---|---|
| `FILE_FAKE_PROB` | 1 − EER | **0.45** |
| `MUSIC_FAKE_PROB` | 1 − EER | **0.27** |
| `VOICE_FAKE_PROB` | 1 − EER | 0.18 |
| `VOICE_PRESENT_PROB` | ROC-AUC | 0.05 |
| `MUSIC_PRESENT_PROB` | ROC-AUC | 0.05 |

Capacity should be allocated in roughly this order. The presence heads together are worth 0.10 on
an *easier* metric — do them correctly, cheaply, once, and stop ([02](02-pretrained-catalog.md#c-presence-heads)).

### 3.2 Ranking-only scoring, and the saturation trap 🔴

EER and ROC-AUC are invariant to monotone transforms, so **no calibration is needed** for any
single head. But the measured risk is the opposite of what people expect:

| Perturbation | Effect on EER |
|---|---|
| Rounding predictions to 2 decimal places | harmless |
| **Saturating the operating point** | **0.0950 → 0.3017** |

(from [`PROGRESS.md`](../../PROGRESS.md), verified by
[`scripts/verify_metric.py`](../../scripts/verify_metric.py))

**Architectural requirement:** the output stage must preserve score resolution near the decision
region. A sigmoid over large logits collapses to 0.0/1.0 in float32 and manufactures ties. Emit
float64, and prefer a non-saturating monotone map into [0,1]. Rank normalization would also fix
ties and is ❌ forbidden by rule 2.4 — so tie-freedom has to come from the numerics.

### 3.3 Masked EERs mean masked losses

Voice EER is computed **only over voice-present files**, Music EER only over music-present files,
using the organizers' ground truth — our presence predictions cannot corrupt the fake heads
([competition/03](../competition/03-evaluation.md)).

Two consequences:

- `VOICE_FAKE_PROB` on a music-only file is **ignored**. There is no penalty for an arbitrary
  value and no benefit either. What matters is ranking quality *within the present pool*.
- ★ The training loss should mirror this exactly: mask each component loss to files where that
  component is present. This is precisely PC-Mix's `λ_s = λ_e = λ_m` scheme
  ([papers/04](../papers/04-component-partial.md)). See [08](08-training-recipe.md).

### 3.4 🔴 Overlapping classes, and one pooled ranking

Two separate constraints live here, and an earlier draft of this file conflated them. They are
untangled because the difference decides whether hard type routing is usable
([03 D](03-candidates.md#d--hard-type-routing--rejected)).

**Constraint 1 — our audio types overlap; AT-ADD's did not.** ★ The AT-ADD 2026 Track 2 winner
(96.10% macro-F1) used a frozen BEATs router to hard-switch between a speech branch and a
non-speech branch ([survey/09](../survey/09-challenge-playbooks.md)). Their four types — speech,
sound, singing, music — are **mutually exclusive**, and their metric was macro-F1 **averaged
equally across the four**, computed independently per type.

🔴 Ours are not exclusive. 혼합 means voice **and** music, simultaneously or 순차적으로, and the
rules class any sung song as mixed ([competition/01](../competition/01-overview.md)). On a mixed
file both components are present, each can be independently fake, and
`FILE_FAKE = VOICE_FAKE OR MUSIC_FAKE` needs evidence from **both** branches at once. A hard 2-way
switch has no defined behaviour there — and cells 5–8 of our 8-cell taxonomy are exactly those
files ([data/02](../data/02-label-taxonomy.md)).

**Constraint 2 — one pooled ranking, which is a *solvable* problem.** Our File EER pools
voice-only, music-only and mixed files into a single ranking, so scores must be comparable across
conditions. Independently trained subsystems sit on different monotone scales, and AT-ADD's
per-type macro-F1 never posed this question at all.

⚠️ **But this does not by itself forbid multiple subsystems.** [05 §1](05-multi-model.md) gives the
remedy and establishes that it is legal: fit a monotone calibration per subsystem **offline on our
own validation pool**, freeze it into the weights, and combine after it. Per-type thresholds in
AT-ADD are a special case of exactly that. So cross-condition comparability is an **engineering
requirement**, not a prohibition — and any design that combines separately trained parts owes it.

🔷 What follows for the architecture: prefer designs where score comparability holds **by
construction** (one head over one representation) because it removes a whole class of calibration
error — but treat comparability as a property to *test*, not as an argument that settles designs
on its own.

⚠️ Related, and genuinely constraining: EER is **not** invariant to the eval set's cell
composition — measured at 0.034 → 0.297 across compositions ([`PROGRESS.md`](../../PROGRESS.md)).
The composition of our validation pool must be frozen by seed
([validation/02](../validation/02-metric-harness.md)).

### 3.5 🔴 Per-file metadata: fully legal, possibly decisive, possibly a mirage

Container (MP3/WAV/FLAC), source bitrate, channel count, duration, encoder fingerprint, spectral
cutoff — every one of these is a **pure per-file property**, so using them as model inputs is
entirely legal under rule 2.4. And in a bring-your-own-data competition where the organizers
assembled REAL and FAKE audio from *different* pipelines, they are a classic high-yield signal.

🔴 They are also a classic **CV-only mirage**, and the two cases are indistinguishable without
evidence. Our own corpus will carry our pipeline's fingerprints, not theirs, so a metadata feature
that scores brilliantly locally may be worthless — or actively harmful — on the real test set.
[data/09 R1](../data/09-risks-and-checks.md) already names the in-corpus version of this
("artificially mixed → fake"), but the *cross-corpus* version is a different question.

⚠️ Note the asymmetry that makes this urgent rather than interesting: **if the leak exists in the
test set, it dominates every other choice in this directory.** It deserves a day-one answer, not a
late ablation — and the only direct evidence we will ever have is the 3 dummy files
([09 A5](09-open-questions.md)).

🔷 Our default until then: keep metadata **out** of the shipped model and **in** the diagnostics,
so it can be measured as a confound rather than silently ridden.

### 3.6 The label structure is an OR over segments

`FILE_FAKE = VOICE_FAKE OR MUSIC_FAKE` over present components, and a component is FAKE if *any*
generated segment is present. Files run 4–60 s and the rules state voice and music may appear
**순차적으로** (sequentially), so a fake component can occupy 3 s of a 60 s file.

★ Whole-file mean pooling dilutes exactly this. The established answers — multi-resolution
segment supervision (PartialSpoof, 0.77% utterance EER), attention pooling, and a `frame_max`
term — all follow from the label semantics rather than from taste. Developed in
[04](04-heads-and-pooling.md).

### 3.7 Denoised real audio stays REAL 🔴

> 실제 원천 오디오에 품질 개선, 잡음 제거, 음량 조정 등 … 후처리만 적용된 경우 → **REAL**

⚠️ A model that fires on "processed-sounding" audio produces false positives by construction.
Meanwhile vocoder/neural-codec resynthesis of real audio **is** FAKE. The boundary between
"AI enhancement" and "AI regeneration" is the single most confusable region in the label space,
and no literature addresses it (G6 in [survey/10](../survey/10-open-questions.md)).

🔷 Architecturally this argues against any design whose discriminative signal is "distance from
pristine" — which includes naive one-class / anomaly framings over real audio. It does *not* rule
out one-class methods, but it means the real-class training pool must contain enhanced, denoised
and loudness-normalized audio, and the architecture must be able to learn a boundary that is not
simply an artifact detector.

---

## 4. Two constraints that are not about the leaderboard

### 4.1 The 2nd stage scores the architecture directly

Of 100 final points, only **30** are the leaderboard. **25** are 모델 개발 —
*"문제에 적합한 모델 설계·선정 / 학습 및 성능 개선 과정의 체계성"*. A further **15** are
결과 해석 및 일반화 — *"탐지 결과와 판단 근거의 해석·시각화"*
([competition/03 §5](../competition/03-evaluation.md)).

🔷 **Interpretability is therefore a scored property of the architecture, not a nice-to-have.**
A per-frame attention head yields time-localized evidence that can be plotted and defended; a
clip-level classifier yields a number. The SED head in [04](04-heads-and-pooling.md) earns points
twice — once on the leaderboard by matching the OR-over-segments semantics, once in the report by
being explainable. ⚠️ Caveat from the literature: post-hoc attribution can be manipulated while
predictions stay unchanged ([papers/INDEX](../papers/INDEX.md)), so prefer
explanations that are **structural** (the attention weights the model actually used) over
post-hoc saliency.

### 4.2 22 days, and the training corpus does not exist yet

Phase B is planned but not executed. Architecture complexity is bounded by *training* wall-clock
against a corpus we still have to build, and every additional independently-trained branch or
ensemble member multiplies it.

★ The Kaggle playbook's ordering is explicit and worth obeying:
`1) data split → 2) sampler/loss → 3) architecture → 4) optimizer`, with a named warning against
over-searching architecture before solving data shift
([kaggle/05 E8](../kaggle/05-transferable-playbook.md)).

🔷 Read against our situation: the architecture decisions that matter are the *structural* ones
made once — how many branches, which frontends, joint or independent training. The tuning-shaped
decisions should wait until there is data to tune against.
