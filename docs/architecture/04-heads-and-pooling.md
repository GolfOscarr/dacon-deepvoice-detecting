# 04 — Heads, Pooling, and the Output Contract

Everything between the frontend features and the five numbers in `submission.csv`.

This is the part of the design most tightly determined by the *label semantics* rather than by
prior art: our labels are an OR over segments, and almost every choice here follows from that.

---

## 1. Why pooling is the whole problem

Three facts from the rules, together:

- Files are **4–60 s**.
- Voice and music may appear **순차적으로** — sequentially, not only overlapped
  ([competition/01](../competition/01-overview.md)).
- A component is FAKE if *any* generated part is present; `FILE_FAKE = VOICE_FAKE OR MUSIC_FAKE`.

So a fake component can occupy **3 seconds of a 60-second file**, and the correct label is still
FAKE. ★ Whole-file mean pooling dilutes that by a factor of 20.

The Kaggle SED notebook states our problem in its own domain's words:

> *"Instead of Global Average Pooling, which dilutes a brief vocalization across the full 5-second
> window, the SED head makes per-frame predictions and aggregates them via learned attention
> weights. A species that calls for 0.3 seconds gets a sharp attention spike at those frames,
> rather than being averaged with 4.7 seconds of background."*

Substitute "a fake voice segment occupying 3 s of a 60 s file" and it is our `VOICE_FAKE_PROB`
pooling problem verbatim ([kaggle/06 §1](../kaggle/06-notebook-code.md)).

★ The speech literature converged on the same answer from the other side: PartialSpoof trains
segment-level **and** utterance-level labels simultaneously at 20–640 ms resolutions and reaches
**0.77% utterance EER** ([survey/01](../survey/01-sota-speech.md)).

---

## 2. The head, concretely

One shared structure for all five heads ([kaggle/06 §1](../kaggle/06-notebook-code.md)):

```python
class GeMFreqPool(nn.Module):
    """Generalized Mean pooling over frequency. Learnable p starts at 3.0
    (sharper than mean, softer than max)."""
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):                      # (B, C, F, T)
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p).mean(dim=2)
        return x.pow(1.0 / p)                  # (B, C, T)

h = self.gem_freq(h).permute(0, 2, 1)          # (B, T, C)
h = self.dense(h).permute(0, 2, 1)             # (B, 512, T)
norm_att         = torch.softmax(torch.tanh(self.att(h)), dim=-1)
framewise_logits = self.cla(h)
clip_logits      = torch.sum(norm_att * framewise_logits, dim=2)
```

**Why GeM specifically.** G5 in [survey/10](../survey/10-open-questions.md) records
"pooling function per head — max vs attention vs median" as an open question. GeM with a
**learnable `p`** is a continuous interpolation between mean (`p=1`) and max (`p→∞`), so the model
settles it per head from data instead of us guessing once for all five. 🔷 We expect the fake heads
to learn a larger `p` than the presence heads, and **the learned values are a reportable finding**
for the 2nd-stage 결과 해석 section.

---

## 3. The loss: clip + frame-max, and why it matches our labels

```python
frame_max_logits = framewise.max(dim=1).values
bce = 0.5 * BCE(clip_logits, y) + 0.5 * BCE(frame_max_logits, y)
```

🔴 **Blend the logits at inference, not the probabilities.** The source notebook blends two
sigmoids:

```python
p = 0.5 * sigmoid(clip_logits) + 0.5 * sigmoid(frame_max_logits)   # ← do NOT ship this
```

That is exactly the construction [§7](#7-the-output-contract-) forbids: two saturating squashes,
averaged, collapsing to 0.0/1.0 in float32 and manufacturing ties across files — and the measured
cost of saturation is **EER 0.0950 → 0.3017**. ⚠️ fp16 inference ([06 §4](06-compression.md)) makes
it worse. Blend in logit space and squash once, in float64:

```python
z = 0.5 * clip_logits + 0.5 * frame_max_logits          # float64
p = expit(z)                                            # monotone in z; order preserved
```

The blend weights are unchanged, the ranking is unchanged where the sigmoid has not saturated, and
the ties disappear where it had. ⚠️ The **training** loss still uses two separate BCE terms —
averaging the losses and averaging the scores are different operations, and only the second one
had this defect.
🔴 `frame_max` **is** the "any part of this file is fake ⇒ the file is fake" operator. `clip` is
the "overall character" operator. Our label semantics decompose the same way, which is why this
blend is adopted rather than merely tried. Three independent sources converge on `max`-style
aggregation for union labels ([kaggle/06 §6](../kaggle/06-notebook-code.md)).

**Add a pairwise ranking term.** ★ EER is pure ranking, and TFPARN combines focal + pairwise
ranking loss *explicitly aimed at EER*; ★ the LLM-Detect-AI winner reached the same conclusion
independently ([papers/09](../papers/09-training-losses.md)). ⚠️ Adopt the loss, not the system —
TFPARN's 12.52% EER is weak against ~4–5% SOTA.

⚠️ **Do not smooth the fake heads.** The Gaussian temporal smoothing in
[kaggle/06 §4](../kaggle/06-notebook-code.md) is legal under rule 2.4 (within-file only) but it
smooths *toward neighbours*, which directly fights `frame_max` semantics. 🔷 Plausible for the
presence heads, questionable for the fake heads — test per head, do not apply globally.

---

## 4. Multi-resolution supervision — and the advantage we get for free

★ PC-Mix supervises frames at **40 / 80 / 160 / 320 / 640 ms** alongside the utterance label;
PartialSpoof does the same at 20–640 ms. Both report that this is what makes short fake components
detectable.

🔷 **We are unusually well placed to do this.** Because we *compose* the training corpus from four
component pools rather than collecting it ([data/02](../data/02-label-taxonomy.md)), we know
exactly which frames contain which component and which are generated. **Frame-level ground truth
is free for us** and expensive for everyone else.

🔴 **But free only on composed files.** Scraped REAL audio and natural AI songs have no frame
labels — so multi-resolution supervision is available on *exactly the subset where the composition
shortcut lives* (§ below). A heavily weighted frame loss therefore trains hardest on the files
whose structure most reliably predicts the label, which is the opposite of what we want.

🔷 The consequence, which an earlier draft left undrawn: **the frame-loss weight is a
shortcut-exposure knob, not just an accuracy knob.** Start it low, raise it only if the `E-S2`
shortcut audit stays clean, and never let composed and natural files differ in whether they carry
frame supervision without checking what the model learns from that difference.

⚠️ The corollary is a trap the data plan already names: if every mixed-and-fake file is
artificially composed while every mixed-and-real file is a natural song, then "artificially mixed"
predicts FAKE perfectly and the model learns our mixing pipeline
([data/02 § the composition trap](../data/02-label-taxonomy.md)). Frame supervision makes that
shortcut *easier* to learn, not harder. The countermeasure is entirely on the data side.

---

## 5. The file head

`FILE_FAKE_PROB` carries **0.45** — exactly as much as the two component heads combined
(0.27 + 0.18). G3 in
[survey/10](../survey/10-open-questions.md) records its construction as an open question with no
prior art, because no other benchmark has this label structure.

Three constructions, and one asymmetry between them that we think is decisive:

| Construction | Form |
|---|---|
| **Analytic noisy-OR** | `1 − (1 − p_vf·p_vp)(1 − p_mf·p_mp)` |
| **Max** | `max(p_vf·p_vp, p_mf·p_mp)` |
| **Learned branch** | a third SED head over the shared features |

🔷 **The asymmetry: both analytic forms need calibrated component probabilities, and ranking
training does not produce them.**

⚠️ Stated precisely, because an earlier draft of this file got it wrong. `1 − (1−a)(1−b)` **is**
strictly increasing in `a` and in `b` — that much is fine. The problem is one level up: EER is
invariant to monotone transforms, so each component head is free to sit on its own arbitrary
monotone scale, and the **order the noisy-OR induces over (voice, music) pairs is not invariant to
independent monotone reparametrisation of the two inputs.** Replace `a` by `g(a)` and `b` by `h(b)`
for any monotone `g, h` — each head's own EER is unchanged, and the file ranking moves. `max` has
the same property.

So the two heads' scales must be made commensurate before either analytic form is meaningful.

⚠️ **This is a cost, not a disqualification.** [05 §1](05-multi-model.md) shows the remedy is
available and legal: fit a monotone calibration per component head offline on our validation pool
and freeze it. So the honest statement is that the analytic forms carry an **extra fitted stage
that can itself be mis-fitted**, while a learned branch is comparable by construction. That is a
reason to expect the learned branch to win, not a reason to skip the comparison.

That is the argument for the **learned branch**, and it is why B's file branch reads *features*
from both frontends rather than *scores* from the two component heads
([03](03-candidates.md#b--three-branches-on-the-mixture-jointly-trained--chosen)).

🔷 **And commensurability is not the strongest argument for the learned branch anyway.** The two
that actually hold: it has **capacity** the analytic forms lack — it can learn that a quiet fake
music bed under real speech matters differently from a fake voice over real music, which no fixed
formula expresses — and it is **trained jointly with the component branches**, which is the
mechanism both PC-Mix and CompSpoof credit with their largest gains. Calibration is a cost the
analytic forms carry, not a reason they cannot work.

⚠️ It is an argument, not a measurement. **Build all three and compare on our own CV** — the
comparison is nearly free once the branches exist, and G3 says nobody has published the answer.
🔷 We expect them to differ most on **mixed** audio, where both components are present and the two
scales actually have to be reconciled.

---

## 6. Temporal coverage: tiling, not sampling

★ The challenge consensus is multi-crop inference, 4–5 crops, median or mean pooled
([survey/09](../survey/09-challenge-playbooks.md)).

🔷 **We think that consensus is mis-specified for our label structure.** For a 60 s file, 4 crops
of 5 s cover 20 s — one third of the audio. A 3 s fake segment in the uncovered two thirds is
invisible, and `frame_max` cannot recover what was never seen. Sampled crops are a *variance
reduction* technique; our labels need *coverage*.

Full tiling of a 60 s file at 5 s windows is 12 windows — 3× the cost of a 4-crop sample, and
[07](07-runtime-budget.md) says we can afford it. **Coverage first; spend leftover budget on
overlap, then on ensemble members.**

| Strategy | Coverage of a 60 s file | Relative cost |
|---|---|---|
| 4 sampled crops × 5 s | 33% | 1× |
| **Full tiling, 5 s, no overlap** | **100%** | **3×** |
| Full tiling, 5 s, 50% overlap | 100%, 2× redundancy | 6× |
| **Whole file in one pass** | **100%**, no boundaries | ~4.9× — see below |

### 🔴 The option the 5 s window hides: don't tile at all

⚠️ The 5 s window is **inherited from the BirdCLEF notebook** whose clips *are* 5 s
([kaggle/06 §1](../kaggle/06-notebook-code.md)). Nothing about our problem requires it, and the
assumption was never surfaced.

Both frontends are transformers, and 60 s at 16 kHz is only ~3,000 encoder frames at 50 fps. A
truncated encoder handles that in **one pass** on an L4 — well inside 22.4 GiB. Running the SED
head over the whole file's frame sequence gives 100% coverage with **no window boundaries, no
recomputed overlap, and no cross-window aggregation problem at all** (§6.1 simply disappears — the
attention pooling already spans the file). It is also the *native* form of an SED head, which was
designed to attend over a whole clip.

🔷 The cost, from FLOP counts (our arithmetic, unmeasured): whole-file is **~1.6× a non-overlapping
tiling of the same audio**, because attention is quadratic in sequence length and reaches ~42% of
encoder cost at 3,000 frames — versus ~6% at 250 frames. Against the 4-crop baseline that is ~4.9×.

| | Tiled 5 s × 12 | Whole file, 3,000 frames |
|---|---|---|
| Coverage | 100% | 100% |
| Window-boundary artifacts | ⚠️ yes | ✅ none |
| Cross-window aggregation (§6.1) | ⚠️ required, and biased | ✅ not needed |
| Duration bias | ⚠️ built in | ✅ attention normalizes over the file |
| Relative encoder cost | 1× | ~1.6× |
| Long-range structure | ⚠️ capped at 5 s | ✅ full 60 s |

⚠️ Two caveats. Batching files of unequal length wastes compute on padding unless bucketed by
duration, and the quadratic term means the cost gap grows with the longest files — the ratio above
is for the 60 s worst case and is smaller for typical files. 🔷 **Our reading: this is probably the
better design**, because it deletes an entire biased mechanism rather than tuning it, and ★
SpecTTTra's +8% F1 on long songs points the same way. But it is a 🔷 not a ★ — decide it with
[09 B5](09-open-questions.md), measured, not here.

### 6.1 🔴 Cross-window aggregation, and the duration trap

Tiling raises a question §3 does not answer: `frame_max` and `clip` are defined *within* a window,
but a 60 s file yields ~12 windows and a 4 s file yields 1. **How the per-window scores become one
file score is a separate decision, and the obvious choice is biased.**

🔷 A plain `max` over windows is stochastically larger the more windows there are, so **long files
score higher than short ones regardless of content**. In a pooled ranking across 1,200 files whose
durations span 15×, that is a systematic bias directly on the metric — and if duration correlates
with label at all in our composed corpus, it becomes a shortcut the model can ride.

⚠️ This trap is already documented in this repo from the other direction: the FoR dataset ships a
`for-2sec` variant *purely* to eliminate duration-vs-label bias
([kaggle/05 A6](../kaggle/05-transferable-playbook.md)).

🔴 **Measured, and it corrects the guess this section originally carried.**
`tests/test_outputs.py` runs each aggregator over identical per-window score
distributions while varying only the window count (W = 1 … 12, 20k files each).
Any movement in the mean file score is pure duration bias:

| Rule | Duration bias (spread over W=1…12) | Keeps "any part fake" |
|---|---|---|
| `max` over windows | 🔴 **1.63** | ✅ |
| **quantile** (90th) | 🔴 **1.08** | ✅ mostly |
| **top-k mean**, k=3 | 🔴 **1.18** | ✅ mostly |
| ☆ Confidence-gated | ⚠️ 0.16 — **an artifact**, see below | ✅ |
| `mean` over windows | ✅ **0.004** | ❌ dilutes — the problem we started from |

⚠️ **Top-k mean does not fix this.** It was assumed "mild" here and is in fact only
~28% better than a plain max. The reason is structural: the expected k-th largest of
W samples grows with W, so *every* order statistic inherits the bias. A fixed
quantile does not escape it either at these small W.

⚠️ **The confidence-gated row is not the win it looks like.** Its gate requires >40%
of windows above logit 1.386 (p = 0.8), which on roughly symmetric scores essentially
never fires — so it falls through to the plain mean and inherits mean's neutrality
*and* mean's dilution. Measured: it equals `mean` on 99.7% of files. Recorded so the
number is not misread as evidence.

🔴 **The honest conclusion is that this is an argument for not tiling at all.**
`segmentation.mode: whole_file` has no windows, so there is no cross-window
aggregation and no duration bias to trade against dilution — the whole dilemma
disappears rather than being managed. That is now the default in both shipped
configs, and it strengthens the case already made in
[09 B5](09-open-questions.md).

☆ That last row is worth a look: a confidence-gated frame rule moved a single model
**0.25 → 0.22** in `[DFDC 2020, 1st]` — mean of predictions >0.8 if enough frames exceed it, mean
of those <0.2 if almost all are below, else plain mean
([kaggle/05 G1](../kaggle/05-transferable-playbook.md)). *"Aggregation beats architecture"* is that
entry's own summary.

🔷 **If we tile at all, no aggregator is good** — the choice is between a duration bias
(order statistics) and dilution (mean). `topk_mean` remains the configured default as the
least-bad order statistic, but the measurement above says the real answer is `whole_file`.
⚠️ And whatever we choose, **duration-vs-score correlation on the REAL class must be an explicit
check**, not an assumption — it is exactly the kind of confound `E-S2`'s metadata-only shortcut
audit is built to catch ([data/07](../data/07-eda-plan.md#tier-s)).

🔴 **This needs a slice that does not yet exist.** [`metrics/breakdown.py`](../../metrics/breakdown.py)
reports per `cell`, `artifact_family` and `fold`; **there is no duration stratum anywhere in
[`docs/validation/`](../validation/README.md)**. Add a binned `duration` key and report per-stratum
EER — otherwise this bias is invisible in exactly the diagnostics built to catch such things.

⚠️ Window length is not free either: ★ SpecTTTra's long-range modelling gave +8% F1 on long songs,
and our 60 s cap means some structural cues span more than one window. 🔷 An open trade — short
windows localize, long windows contextualize. Decide with the mel/window sweep in
[09](09-open-questions.md), not by assertion.

---

## 7. The output contract 🔴

Three requirements that are easy to get wrong and expensive to get wrong.

**7.1 Do not saturate.** Measured: rounding to 2 dp is harmless; **saturating the operating point
took EER 0.0950 → 0.3017** ([`PROGRESS.md`](../../PROGRESS.md)). A `sigmoid` over large logits
collapses to exactly 0.0 / 1.0 in float32 and manufactures ties across files. Emit **float64**, and
prefer a monotone map into [0,1] that does not saturate in the decision region. ❌ Rank
normalization would also solve it and is forbidden by rule 2.4 — tie-freedom must come from the
numerics.

**7.2 Masked columns are free.** `VOICE_FAKE_PROB` on a music-only file is never scored. There is
no penalty for an arbitrary value there and no benefit — so **do not spend capacity on it**, but
also do not emit `NaN`, which would break the CSV contract.

**7.3 Write via the shipped writer.** Column order comes from the real `sample_submission.csv`,
not from a hardcoded tuple — three submission-contract defects were already found and fixed by
audit, including exactly this one ([`metrics/submission.py`](../../metrics/submission.py),
[`metrics/AGENTS.md`](../../metrics/AGENTS.md)).

---

## 8. What the head gives the report

🔷 The SED head is chosen partly for a non-leaderboard reason. **25 of 100** 2nd-stage points are
모델 개발 and **15** are 결과 해석 및 일반화 — *"탐지 결과와 판단 근거의 해석·시각화"*
([competition/03 §5](../competition/03-evaluation.md)).

`norm_att` is a per-frame attention distribution the model *actually used*. Overlaying it on a
spectrogram gives a defensible answer to "why did you flag this file, and where" — for free, as a
byproduct of the architecture rather than as a separate explainability pipeline.

⚠️ Prefer this to post-hoc attribution: ☆ *The Perceived Fragility of Explanations in Audio Models*
(2026) shows post-hoc attributions can be manipulated while predictions stay unchanged
([papers/INDEX](../papers/INDEX.md)). Structural explanations do not have that failure
mode. ☆ *Interpreting Multi-Branch Anti-Spoofing Architectures* (2026) is directly about what
multi-branch models key on and should be read before writing that section
([09](09-open-questions.md)).
