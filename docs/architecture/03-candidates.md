# 03 — Whole-System Candidates

Each candidate is a complete answer to "what runs inside `script.py`". They are scored against
[01](01-design-envelope.md) and the evidence in [`docs/papers/`](../papers/INDEX.md).

**Chosen: candidate B.** 🔴 **A is stage one of B, not a fallback** — B strictly contains A, so
building A first costs nothing and leaves a scoring submission on the board throughout
([§ A-is-stage-one](#-a-is-stage-one-of-b-not-a-fallback)).

Only **A, B, C-lite and C** are whole-system alternatives to each other. **E, F and H** are
*additions to B*, not replacements for it; **D and G** are rejected; **P0-a/b** are harnesses.

| | Candidate | Kind | Separation | Frontends | Verdict |
|---|---|---|---|---|---|
| **P0-a** | All-constant contract probe | harness | none | none | ✅ **Ship first.** Must return **exactly 0.5000** |
| **P0-b** | PANNs presence + constant fakes | harness | none | PANNs | ✅ Then this. Expect **~0.535**. ⚠️ Check it on a *song* first ([09 A7](09-open-questions.md)) |
| **A** | Single shared trunk, 5 SED heads | **system** | none | 1 | ✅ **Stage one of B.** Cheapest, genuinely competitive, must always work |
| **B** | **3 branches on the mixture, jointly trained** | **system** | **none** | **2 specialists** | ⭐ **CHOSEN** |
| **C-lite** | B + separation as an **auxiliary training head**, discarded at inference | **system** | training only | 2 | ⭐ **Live and cheap.** Try as soon as B works |
| **C** | Jointly-trained separation (CompSpoof) | **system** | joint, ours, 16 kHz | 2–3 | ⚠️ Best published F1, most expensive. Hold |
| **E** | B + residual-teacher music branch | add-on | teacher only | 2 | ⚠️ **Live**, gated on `E-A1` |
| **F** | B + one-class real-only member | add-on | none | 1 extra | ⚠️ **Live** as an ensemble member, not a system |
| **H** | B + **presence-gated** frontend skipping | add-on | none | 2 + gate | ⚠️ Only against measured runtime pressure |
| **D** | Hard type routing (AT-ADD Track 2) | rejected | none | 2 + router | ❌ Undefined on 혼합 files |
| **G** | Frozen separator → per-stem detectors | rejected | frozen, 44.1 kHz | 2 | ❌ Four independent grounds |

---

## P0 — the baselines that are not models

🔴 **These are two separate submissions and must not be merged.** An earlier draft of this file
described one "P0" that was PANNs presence heads *plus* constant fake columns, and claimed it
should score exactly 0.5000. It cannot, and shipping it as the contract probe would trip
[validation/05](../validation/05-lb-probe-plan.md)'s stop rule on a perfectly correct submission.

With all three fake columns constant every fake EER is 0.5, so `ADS = 0.5` and
`Score = 0.45 + 0.1·CPS`. Real presence predictions push CPS well above 0.5:

| Submission | CPS | Score |
|---|---|---|
| **P0-a** — *all* columns constant 0.5 | 0.5 | **0.5000** ← the contract probe |
| **P0-b** — PANNs presence + constant fakes | ~0.85 | **0.5350** |
| P0-b, better presence heads | ~0.95 | 0.5450 |

**P0-a — the contract probe.** All five columns constant. Validates the I/O contract, offline
packaging, the glob-any-extension path, runtime, and the startup assertions, and must return
**exactly 0.5000**; anything else means we have misread the metric or the submission format, and
[validation/05](../validation/05-lb-probe-plan.md) says **stop**. ⚠️ Write it with `validate=False` —
our own VG5 gate rejects a constant column and would otherwise block the first submission
([`metrics/submission.py`](../../metrics/submission.py)).

**P0-b — the first real baseline.** PANNs presence heads, fake columns still constant. Its expected
score is ~0.535, **not** 0.5000, and it is the first measurement of how much the 0.10 presence
weight is actually worth on the real distribution. Run it only after P0-a has returned 0.5000.

🔷 Everything below is worthless until P0-a has scored. An architecture that cannot be packaged is
not an architecture.

---

## A — Single shared trunk, five SED heads

One encoder, five heads. Every head sees the same representation of the whole file.

```
audio (4–60 s, 16 kHz, mono|stereo)
  ↓ decode → downmix to mono → (tile, or one pass — [09 B5](09-open-questions.md))
  │
  ├─ general-audio SSL encoder, truncated + frozen, LoRA/adapters   →  H : (B, T, D)
  │        T ≈ 50 tokens per second of audio · D ≈ 768–1024
  │
  └─ 5 × SED head, each independently parameterised over the SAME H
           ├→ VOICE_FAKE_PROB
           ├→ MUSIC_FAKE_PROB
           ├→ FILE_FAKE_PROB
           ├→ VOICE_PRESENT_PROB
           └→ MUSIC_PRESENT_PROB
```

**One SED head, concretely** ([04](04-heads-and-pooling.md)):

```python
h    = dense(H)                                  # (B, T, 512)
att  = softmax(tanh(att_conv(h)), dim=time)      # (B, T, 1)  — where the model is looking
frame_logits = cla_conv(h)                       # (B, T, 1)  — per-frame evidence
clip_logits  = (att * frame_logits).sum(time)    # (B, 1)     — attention-pooled
score        = 0.5*clip_logits + 0.5*frame_logits.max(time)   # blended in LOGIT space
```

⚠️ **The GeM frequency-pooling step applies only to frontends that keep a frequency axis.**
wav2vec2 / XLS-R / WavLM emit `(B, T, D)` — already pooled over frequency, so the head starts at
`dense`. BEATs / EAT / SSLAM are ViT-style over mel patches, so their tokens can be reshaped to a
`(F', T')` grid and GeM-pooled over `F'` first. This distinction was implicit before and is easy to
get wrong when implementing.

**For it.** One forward pass, one set of weights, no cross-encoder time alignment, no
cross-subsystem calibration. 🔷 Best possible score comparability for the pooled File EER, because
every score comes from one representation on one scale
([01 §3.4](01-design-envelope.md#34--overlapping-classes-and-one-pooled-ranking)). Materially the
cheapest system on every axis — engineer-days included.

**Against it.** 🔴 Ignores CompSpoof's finding that one encoder serves voice and music unequally
([papers/04](../papers/04-component-partial.md)), and forgoes the joint-branch gain that is the
largest measured effect on this axis. ⚠️ A general-audio encoder is also the *less* proven choice
for the voice head, where speech SSL frontends dominate every published result.

**Verdict** ✅ **Build this first — it is stage one of B, not a fallback.** B strictly contains it,
so nothing here is throwaway work; it is rung 3 of the drop order
([08 §4b](08-training-recipe.md#4b--the-budget-nobody-costed-engineer-days)); and it gets all five
heads scoring end to end while the second frontend is still being built. See
[§ A-is-stage-one](#-a-is-stage-one-of-b-not-a-fallback) for why the A→B increment is small and
what B's marginal claim actually is.

---

## 🔴 A is stage one of B, not a fallback

Treating A as a contingency — *"build B, fall back to A if B fails"* — misprices it twice.

**1. B strictly contains A.** Same SED heads, same masked losses, same joint multi-task training
over a trunk. The increment from A to B is exactly three things:

| From A to B | Cost |
|---|---|
| A second (speech-specialist) frontend | one more truncated encoder in the zip and the forward pass |
| Per-branch adapters instead of five heads on one trunk | small |
| Time-base alignment between the two frontends' frame rates | one interpolation, easy to get wrong |

Nothing built for A is discarded. So **"A then B" is strictly cheaper than "B, and A if B fails"** —
the second ordering either writes A's code late, under deadline pressure, or ends with neither
finished. ⚠️ Given the corpus build consumes days 3–12 of 22
([data/08](../data/08-build-plan.md)), the realistic failure mode is not a clean verdict against B
on day 15; it is B at 80% on day 20 with nothing else packaged.

**2. 🔴 A already has the joint training — so B's marginal claim is narrower than it looks.**
Five heads over one trunk trained together with masked losses *is* multi-task joint learning.
PC-Mix's headline ablation was **independent → joint**, and A is joint by construction, so A
already captures the largest measured effect on this axis.

⚠️ What B adds over A is therefore **only frontend specialization** — one encoder versus two — and
the evidence for that single axis is a **single citation**: CompSpoof's stated limitation that
their speech detector performed worse on the non-speech component
([papers/04](../papers/04-component-partial.md)). Everything else attractive about B, A has too.

🔷 That is not an argument against B. Specialization is the right bet and it is cheap to add on
top. It is an argument that A is a **legitimate submission rather than a consolation prize**, and
that the A→B comparison is the cleanest ablation we will run — 2nd-stage report material either
way, since 25 of 100 points are 모델 개발 judged on *"학습 및 성능 개선 과정의 체계성"*
([competition/03 §5](../competition/03-evaluation.md)).

### 🔴 Decide "did B beat A?" before you measure it

⚠️ *"If B fails, use A"* is not yet a decision — **"B failed" is undefined**. The LB noise floor is
**±1.7 pts** EER on the File head and **±2.5** on each component head, and VAL needs **≥1,200 per
class per masked pool** to resolve a 1-point gap ([validation/](../validation/README.md)). If B
beats A by less than that, no measurement distinguishes them.

The rule already exists and must be applied here rather than reinvented under deadline: **B is
promoted over A only if P1–P6 all pass** ([validation/03 §3](../validation/03-decision-protocol.md)) —
paired bootstrap CI on the Score difference excluding 0, no cell regressing >2 pts, no artifact
family collapsing, fold variance not growing, runtime margin holding, gates green.

★ And when P1 is inconclusive, the pre-committed tiebreaker prefers **lower fold variance → better
worst-cell EER → better SHADOW → lower runtime**, never the higher public LB. 🔷 Note which way that
leans: on an inconclusive result, **the simpler model wins**, because A has one encoder, no
cross-frontend alignment, and more runtime headroom.

---

## B — Three branches on the mixture, jointly trained ⭐ CHOSEN

The PC-Mix layout ([papers/04](../papers/04-component-partial.md)), mapped onto our five outputs.

### 🔴 Read this first: the branches are *outputs*, not input types

The single most likely misreading of this design, and it is the exact line separating B from the
rejected [candidate D](#d--hard-type-routing--rejected).

> **Every file passes through every branch. Nothing is routed anywhere.**
> The "voice branch" is not the branch for voice-only files — it is the branch that **produces
> `VOICE_FAKE_PROB`**. A music-only file still goes through it. A 60-second song goes through all
> three.

| | Branch = output head (**B, correct**) | Branch = input type (**D, rejected**) |
|---|---|---|
| A voice-only file | runs all 3 branches | runs the speech branch only |
| A 혼합 (mixed) file | runs all 3 branches | ❌ **undefined** — belongs to both |
| Branches trained | together, on the same files | separately, on disjoint data |
| `FILE_FAKE = VOICE OR MUSIC` | ✅ both branches available on every file | ❌ only one branch ran |

The "mixture" in *"three branches on the mixture"* refers to the **input** — all three branches read
the full, unseparated audio mixture. It does **not** mean one of the branches is for mixed files.

### The graph

```
audio (any ext, 4–60 s, 16 kHz, mono|stereo)
  ↓ decode → downmix to mono → (tile, or one pass)
  │
  ├─ Fs = speech SSL, truncated + frozen + adapters        → (B, Ts, Ds)   ─┐
  └─ Fa = general-audio SSL, truncated + frozen + adapters → (B, Ta, Da)   ─┤
                                                                            │
   ┌────────────────────────────────────────────────────────────────────────┘
   ├→ voice branch   : adapter(Fs)              → SED head → VOICE_FAKE_PROB
   ├→ music branch   : adapter(Fa)              → SED head → MUSIC_FAKE_PROB
   ├→ file  branch   : adapter([Fs ⊕ Fa])       → SED head → FILE_FAKE_PROB
   └→ presence ×2    : adapter(Fa)              → SED head → VOICE/MUSIC_PRESENT_PROB
```

⚠️ **`Ts` and `Ta` will not match.** The two frontends have different frame rates, so the file
branch cannot simply concatenate them — one sequence must be resampled onto the other's time base
(linear interpolation over the time axis is sufficient) before `⊕`. Easy to miss, and it silently
misaligns evidence if skipped.

### How the branches are kept honest: masked losses

```python
m_v = voice_present            # GROUND TRUTH, 0/1 — not our prediction
m_m = music_present

L =  w_v * (m_v * L_voice).sum() / m_v.sum().clamp(min=1)      # voice loss only where voice exists
  +  w_m * (m_m * L_music).sum() / m_m.sum().clamp(min=1)
  +  w_f * L_file                                              # every file
  +  w_vp * L_vpres + w_mp * L_mpres                           # every file
  +  alpha * mse(student_emb, teacher_emb.detach())            # distillation
```

🔴 The masks mirror the metric exactly. Voice EER is computed **only over voice-present files**
using the organizers' own presence labels, so training the voice branch on music-only files teaches
it to fit something that will never be scored
([01 §3.3](01-design-envelope.md#33-masked-eers-mean-masked-losses)).

⚠️ **Masking is training-time only.** At inference we always emit all five columns for every file;
the organizers do the masking on their side. `VOICE_FAKE_PROB` on a music-only file is simply
ignored — so do not spend capacity on it, and do not emit `NaN`.

### Where to put unequal capacity

🔷 The branches need **not** be the same size, and the natural instinct — make the file branch
biggest because it carries 0.45 — is probably wrong. `FILE_FAKE` is *derived* from the components
(`OR` over what is present), so strong component branches largely carry it. The branch that
deserves more is the **music** branch:

| Branch | Metric weight | Difficulty |
|---|---|---|
| Music fake | **0.27** | 🔴 hardest — published cross-generator is **46.4% EER** |
| Voice fake | 0.18 | ~4–5% EER at SOTA on unseen generators |
| File fake | **0.45** | derived from the two above |

⚠️ But spend the asymmetry in this order, because **branch head size is the weakest of the four
levers** — the heads are a dense layer plus two Conv1d over already-pooled features, a rounding
error against two truncated SSL frontends:

1. **Loss weights** `w_v / w_m / w_f` — currently equal, inherited from PC-Mix whose metric weighted
   components equally. Ours is 0.45 / 0.27 / 0.18 ([08 §2](08-training-recipe.md), [09 B11](09-open-questions.md))
2. **Data** — more music generator families; we are blocked at 5 and need ≥8 ([09 C6](09-open-questions.md))
3. **Frontend choice and truncation depth for the music branch** — ★ the survey rates frontend
   ⭐⭐⭐⭐⭐ against backbone ⭐⭐ ([09 A6](09-open-questions.md))
4. **Branch width/depth** — last, and probably marginal

### Why this design

| Reason | Evidence |
|---|---|
| **Joint training across branches is the largest single reported gain on this axis** | ★ PC-Mix: 5-class utterance ACC **69.40 → 85.12**; env F1 **76.41 → 92.67**. ★ CompSpoof: F1 **0.668 → 0.908** |
| Same problem, same sample rate, same label structure | PC-Mix is 16 kHz, three branches, component + mixture labels — a 1:1 map onto our heads |
| Type-specialized frontends, without a router | ★ CompSpoof's stated limitation + ★ AT-ADD Track 2's frontend split ([02](02-pretrained-catalog.md#the-case-for-two-frontends)) |
| No separation ⇒ no 16 kHz separation risk, no separation runtime | ★ Three independent results ([G](#g--frozen-separator--per-stem-detectors--rejected)) |
| Cheaper to train than **C** | 🔷 ⚠️ *Not* cheap in absolute terms — two frontends, three branches, five heads, adapters, four training stages and a teacher ensemble. **Candidate A is materially cheaper on every axis**, and the A-vs-B call rests on a single citation against A's own advantage on score comparability for the 0.45 head. Treat A as a live comparison, not a formality |

### The two design choices inside B that are ours, not PC-Mix's

**1. The file branch reads features, not scores.** 🔷 PC-Mix's third branch distinguishes
constructed mixtures from originals; ours predicts `FILE_FAKE`. We feed it the concatenated
frontend features rather than the two component *scores*. ⚠️ Not because combining scores is
unworkable — [05 §1](05-multi-model.md) shows offline calibration handles that — but because one
head over one representation is comparable **by construction**, which removes a class of
calibration error rather than managing it
([01 §3.4](01-design-envelope.md#34--overlapping-classes-and-one-pooled-ranking)). The analytic
noisy-OR remains the baseline it must beat ([04](04-heads-and-pooling.md#5-the-file-head)).

**2. Frozen frontends + adapters, not two fine-tuned encoders.** Keeps the zip inside 10 GB, keeps
inference inside budget, and ☆ meta-learned LoRA beats full fine-tuning anyway (EER 8.84 → 5.30%
at ~1.1% trainable params). See [06](06-compression.md). ⚠️ Open: whether the joint-training gain
survives frozen frontends at all ([09 B4](09-open-questions.md)).

### What B does not solve

⚠️ PC-Mix's own stated limitation is that the env-sound branch degrades most under
**background-domain shift** (0.67% → 9.15% EER). Our analogue is the music branch under unseen
generators, where the published cross-generator number is **46.4% EER**. B inherits this; the
answer is data composition ([data/](../data/README.md)) and augmentation, not architecture.

---

## C — Jointly-trained separation (CompSpoof)

```
mixture → UNet complex-mask separation (STFT, ours, trained at 16 kHz)
            ├→ voice stem → detector
            └→ music stem → detector
       └────────────────────→ mixture detector
   independent for 4 epochs, then joint
```

**For it.** ★ The **best published number** on our exact problem shape: overall F1 **0.908**, vs
0.827 for the direct baseline and 0.668 for separation without joint learning. And because we
would train the separator ourselves at 16 kHz, it **sidesteps the 44.1 kHz domain mismatch
entirely** — the objection that kills candidate G does not apply here
([papers/08](../papers/08-tagging-separation.md)).

**Against it.** 🔷 Separation quality becomes our problem too: a second model to design, train and
validate, on a corpus that does not exist yet, in 22 days. It also puts the separator in the
inference path, spending runtime we would rather spend on full temporal coverage
([07](07-runtime-budget.md)).

**Verdict** ⚠️ **Hold.** It is the strongest published result and we should say so in the report.
🔷 Our reasoning for not starting here: B captures the *mechanism* that both papers agree on —
joint training across component branches — at a fraction of the cost, and PC-Mix explicitly
demonstrates that mechanism works without any separator. If B plateaus and time remains, C is the
upgrade path.

---

## D — Hard type routing ❌ REJECTED

```
BEATs (frozen) → hard 4-way audio-type router
                   ├─ speech     → XLSR + AASIST
                   └─ non-speech → EAT-large + AASIST
```

★ This won AT-ADD 2026 Track 2 with **96.10% macro-F1** — the closest prior challenge to ours.
Our survey's own summary of it is that **hard type routing beat unified detection**
([survey/09](../survey/09-challenge-playbooks.md)); that is the survey author's reading of the
result, not a quotation from the winning team.

**We still reject it** — but the reason in an earlier draft of this file was wrong, and the
correction matters because it changes what we keep.

❌ **The argument we withdraw.** We previously said: their macro-F1 used per-type thresholds, ours
is a threshold-free pooled EER, so nothing absorbs the score-scale mismatch between two
independently trained subsystems. The second half is false by our own reasoning —
[05 §1](05-multi-model.md) establishes that a monotone calibration per subsystem, fitted offline on
our validation pool and frozen into the weights, is both legal under rule 2.4 and exactly the thing
that absorbs it. Per-type thresholds are a special case of that. Score scale is an engineering
requirement, not a reason to reject a design.

🔴 **The argument that actually holds: their classes were mutually exclusive and ours are not.**
AT-ADD Track 2's four types — speech, sound, singing, music — partition the input, and macro-F1 was
computed **independently per type and averaged**. Our third audio type is 혼합: voice **and** music,
simultaneously or 순차적으로, with any sung song counted as mixed. On such a file both components
are present, each can be independently fake, and `FILE_FAKE = VOICE_FAKE OR MUSIC_FAKE` requires
evidence from **both** branches at once. **A hard 2-way switch has no defined behaviour there** —
and cells 5–8 of our 8-cell taxonomy are precisely those files
([data/02](../data/02-label-taxonomy.md)).

That is a structural mismatch, not a calibration one, and no amount of offline alignment fixes it.

⚠️ **What rejecting D costs us, stated honestly.** Routing's real benefit is compute: a voice-only
file never runs the music frontend, roughly a 2× saving on single-component files. 🔷 Against the
corrected margin in [07 §2](07-runtime-budget.md) — **~3.7–10×**, not the ~1.25–3.3× an earlier
draft computed from a double-counted tiling factor — that saving is worth having but is **not**
needed to fit. We give it up for correct behaviour on mixed audio; candidate **H** recovers part of
it if measurement ever makes it necessary.

**What we keep.** The *insight* — type specialization matters — is adopted in full via two
specialist frontends in B. What we drop is the **switch**.

## E — B + residual-teacher music branch ⚠️ LIVE, gated on `E-A1`

⭐ ArtifactNet's Phase 1 distils **Demucs v4 residuals** into a small forensic model, so separation
is a *teacher* and never enters inference ([papers/02](../papers/02-music-detection.md)). This is a
third way that neither PC-Mix nor CompSpoof covers, and 🔷 with 8×H200 the teacher is free.

Its measured results are strong — F1 **0.9829** on unseen generators at 4.0M params, cross-codec
probability drift **−83%** after codec-aware training, hard-negative FPR **98.7% → 8.0%**.

⚠️ Three cautions, all from the paper itself:

1. 🔴 **It is a 44.1 kHz method.** *"Our forensic features operate on 44.1 kHz residuals targeting
   high-frequency RVQ artifacts. Benchmarks distributed at reduced sample rates (e.g. 16 kHz)
   attenuate the forensic signal."* Whether the **mechanism** survives our band-limit is exactly
   what `E-A1` measures ([09](09-open-questions.md#a1---the-16-khz-survivability-probe-e-a1)).
2. ⚠️ **Patents pending (KR + PCT)** on bounded-mask residual extraction and codec-invariant
   training — in a Korean government competition whose rules assign winning-work copyright to the
   host. Worth a legal look before reimplementing that specific formulation.
3. 🔴 If we build *any* residual/mask channel, **bound the mask**. Their ablation: with an
   unbounded `[0,1]` sigmoid the UNet converges to mean mask ≈ 1.0 and residual energy exceeds 95%
   of the input — it degenerates into passing the input through, i.e. does nothing.

**Verdict** ⚠️ Adopt the **codec-aware training phase** unconditionally (it is a training schedule,
[08](08-training-recipe.md)). Adopt the residual channel only if `E-A1` says the mechanism survives
below 8 kHz.

---

## F — One-class / real-only member ⚠️ LIVE as an ensemble member

☆ MusicDET trains band-wise **normalizing flows on real music only**, and is generator-agnostic by
construction: **4.51% EER** on FakeMusicCaps, **2.89%** on SONICS. ★ The fakeprint work reaches
the same real-only framing from a different direction.

🔷 The appeal is structural, not numeric: our binding constraint is cross-generator generalization
(46.4% EER published), and a model that never sees a generator cannot overfit to one. It is also
the most *different* model we could add to an ensemble, which is what
[05](05-multi-model.md) says actually buys anything.

⚠️ Two hard warnings:

- MusicDET's own robustness section is our failure regime: **pitch shift +40 points EER**,
  **MP3 @ 64 kbps +37 points**. Our test set is MP3/WAV/FLAC at 16 kHz.
- 🔴 It collides with the rules' REAL definition: denoised, enhanced and loudness-normalized real
  audio **stays REAL** ([01 §3.6](01-design-envelope.md#37-denoised-real-audio-stays-real-)), so a
  pure "distance from pristine" score is systematically wrong on a whole slice of the real class.

**Verdict** ⚠️ Not a system. Worth one seat as a **diversity member** in the fusion of
[05](05-multi-model.md), and only after being tested under our own degradation chain.

---

## G — Frozen separator → per-stem detectors ❌ REJECTED

The design the preinstalled package list appears to invite (`demucs`, `panns-inference`,
`torchlibrosa` are all preinstalled). **The evidence rejects it on four independent grounds:**

| Ground | Number |
|---|---|
| ★ Separation without joint learning is **worse than not separating** | CompSpoof F1 **0.668 vs 0.827** |
| ★ Speech detection on separated streams is **much worse** than on the mixture | PC-Mix **51.38% vs 29.35%** EER |
| ★ Separation artifacts spread across *all* stems | Hybrid-stems **38% FPR** (vocals) / **94.7% FPR** (accompaniment) |
| ★ No MSS method is confirmed native at 16 kHz; HT-Demucs is a 44.1 kHz model and our audio includes telephone-band | [papers/08](../papers/08-tagging-separation.md) negative finding |

Their stated reasons agree: PC-Mix — *"source separation may introduce artifacts or remove
environmental-sound spoofing cues"*; hybrid-stems — *"artifacts associated with an AI-generated
stem are not reliably recovered by generic source separation systems."*

⚠️ Note this is the one place where we deliberately act against a signal from the organizers'
environment. That is a defensible call with four citations, and it belongs in the 2nd-stage report
as a documented negative result rather than an omission.

---

## C-lite — separation as an auxiliary training head ⭐ LIVE

```
training:   mixture → frontends → ├→ voice / music / file branches   (shipped)
                                  └→ separation head (voice+music stems)  ← auxiliary loss only
inference:  the separation head is deleted
```

🔷 [Candidate C](#c--jointly-trained-separation-compspoof) is deferred partly because it "puts the
separator in the inference path". That is a property of **CompSpoof's deployment, not of the
mechanism**. The gain they measure — F1 **0.668 → 0.908** — is credited to *joint learning*, and
joint learning is a training-time phenomenon.

So: train a separation head jointly with the branches as an auxiliary task, and **discard it before
packaging**. Zero inference cost, zero packaging cost, and the separator is trained by us at 16 kHz,
so the 44.1 kHz domain mismatch that kills [G](#g--frozen-separator--per-stem-detectors--rejected)
never arises.

⚠️ Honest caveats: CompSpoof's 0.908 is measured with the separator *in* the inference path, so we
should not expect to inherit all of it — the auxiliary head shapes the shared representation, which
is a weaker claim than reconstructing stems at test time. It also needs stem targets, which we have
for composed audio and not for natural songs.

**Verdict** ⭐ This is the cheapest way to test the single largest reported gain on this axis, and
it belongs immediately after B works. It is strictly cheaper than **E** and should be tried first.

---

## H — presence-gated frontend skipping ⚠️ LIVE under runtime pressure

Run the cheap presence heads first; where music is confidently absent, skip the music frontend
entirely (and symmetrically for voice). On mixed files, run both — so unlike [D](#d--hard-type-routing--rejected)
it **degrades to candidate B rather than to an undefined state**, which is the objection that
rejected D.

🔷 This recovers most of the compute saving that rejecting D forfeited — roughly 2× on
single-component files — without the structural defect.

⚠️ Two real risks:
1. **A presence false-negative silently zeroes a component's evidence**, and `MUSIC_FAKE` is worth
   0.27. The gate must be deliberately conservative — skip only on high-confidence absence — and
   the cost of a miss must be measured, not assumed.
2. Skipping changes the file head's input distribution between gated and ungated files. Zero-filling
   the missing features is the obvious handling, and it must be trained that way, not bolted on.

**Verdict** ⚠️ Do not build this speculatively. It is the first thing to reach for **if and only if**
[07 §4](07-runtime-budget.md#4-the-measurement-protocol-) shows we are over budget.

---

## What would change the choice

| If… | Then |
|---|---|
| `E-A1` shows residual concentration separates AI from human music below 8 kHz | Promote **E**; the music branch gains a forensic channel |
| B works and time remains | Try **C-lite** first — the joint-learning gain at zero inference cost |
| B plateaus on the music head with time remaining | Escalate to **C** (joint separation, trained by us at 16 kHz) |
| Runtime measurement shows we are over budget | **H**, presence-gated skipping, before dropping coverage |
| Two frontends fail to beat one on our CV | Fall back to **A**, and report it |
| Our CV shows hard routing winning | The argument in **D** is wrong — update it here |
| A licence blocks both SSLAM and EAT | BEATs (MIT), no design change |
