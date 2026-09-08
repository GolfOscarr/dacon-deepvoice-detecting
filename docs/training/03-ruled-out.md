# 03 — Ruled Out, and Why

Every objective-side technique investigated, with the measured number that decided it.

🔴 **Read [01 §2.2–2.4](01-what-the-metric-demands.md) first.** Almost every entry below is
explained by one of two facts: the technique buys **calibration**, which our ranking metric cannot
read; or it buys **imbalance repair**, which we decline by choosing our own cell mix.

⚠️ **The pattern is not coincidence.** The anti-spoofing and audio-tagging literature reports
minDCF, Cllr, actDCF, Macro-F1 and accuracy. All are threshold- or calibration-sensitive. When the
*same ablation* also reports EER or AUC, the effect is routinely an order of magnitude smaller.
Quoting a headline number from this literature without checking which metric produced it is the
single easiest way to adopt something worth nothing.

---

## 1. Pairwise ranking loss — ❌ weight stays 0

⚠️ **This reverses a decision already in the codebase.**
[`models.losses.pairwise_ranking_loss`](../../models/AGENTS.md) is implemented, and its docstring
and [papers/09](../papers/09-training-losses.md) both justify it as *"EER is a pure ranking metric,
so a pairwise loss optimises it directly — independently arrived at by TFPARN and by the
LLM-Detect-AI winner."* **TFPARN's own ablation refutes this for EER.**

★ Yin & Zhao, *A Training-Efficient Transformer-Based Anti-Spoofing Network for Logical Access in
ASVspoof 5* ([arXiv:2606.02980](https://arxiv.org/abs/2606.02980)) — Table VI, 3 seeds, ASVspoof 5
Track 1 closed:

| ID | Loss | Pairwise | Pooling | **EER (%)** | minDCF | Cllr |
|---|---|:--:|---|---|---|---|
| 3 | CE | **No** | Mean | **12.91 ± 0.09** | 0.2662 | 1.8796 |
| 4 | CE | **Yes** | Mean | **12.92 ± 0.07** | 0.2561 | 1.6786 |
| 5 | Focal | Yes | Mean | 12.70 ± 0.36 | 0.2499 | 0.7232 |
| 6 | Focal | Yes | Attention | 12.52 ± 0.11 | 0.2430 | 0.9243 |

Isolated contributions: **pairwise +0.01 EER (worse)** · focal −0.22 against a ±0.36 seed std ·
attention pooling −0.18. All three together move EER by **0.39**, against a ≈1 pt local resolution threshold.

The paper states it outright: *"Adding the pairwise ranking branch leaves the EER almost unchanged,
from 12.91% to 12.92%, but lowers minDCF, Cllr and actDCF; the ranking term acts on the decision
cost and score ordering rather than on the equal-error point."* Every gain in that paper lives in
the calibration-sensitive metrics — Cllr falls 1.6786 → 0.7232, worth zero to us.

**The strongest counter-case, and why it does not reach us.** ★ *Understanding the Ranking Loss for
Recommendation with Sparse User Feedback* (KDD 2024, Criteo/DCNv2) measures the gain as a function
of positive sparsity: **+0.095% relative AUC at 3.3% positives, +0.020% at 25.6%**. Their optimal
mixing weight (0.3 on the rank term) is the same λ TFPARN used. At our roughly balanced positive
rate the trend lands near **+0.0002 absolute AUC** — four orders of magnitude below our noise floor,
and the extrapolation is generous because the mechanism has switched off by 25%.

🔴 **Verdict: `LossConfig.ranking_weight = 0.0`** (its current default), and **stage S4 is dropped**
from [architecture/08 §1](../architecture/08-training-recipe.md#1-the-staged-schedule)'s ladder
rather than kept as a cheap final step. The remaining argument for S4 — ☆ G2Net's *"~1 bps"*
([kaggle/05 F4](../kaggle/05-transferable-playbook.md)) — is itself below our floor. Dropping it
*frees* engineer-days. Replace it with sampler constraint **C1** ([01 §2.4](01-what-the-metric-demands.md#24--we-control-the-imbalance-so-we-remove-the-pathology-instead-of-patching-it)).

---

## 2. Focal loss — ❌ provably zero

Not merely unmeasured — **proved at the population optimum**
([01 §2.2](01-what-the-metric-demands.md)). Charoenphakdee, Vongkulbhisal, Chairatanakul & Sugiyama
([arXiv:2011.09172](https://arxiv.org/abs/2011.09172)): Thm 3 (classification-calibrated, its proof
establishing order preservation), Thm 5 (not strictly proper), Thm 11 (closed-form recovery
`Ψᵢ^γ`), Lemma 14 (`h^γ` strictly increasing on (0,1)). ⚠️ The final step — from their
*across-class* order preservation to the *across-instance* ordering AUC actually reads — is 🔷 our
own derivation from Thm 11 and Lemma 14, not a statement in the paper. Conclusion: focal and BCE
have **identical AUC/EER at the population optimum**.

Consistent with TFPARN Table VI above: focal moves Cllr 1.6786 → 0.7232 (a pure calibration effect)
and EER by −0.22 against a ±0.36 seed std.

⚠️ The AT-ADD 2026 Track 1 winner (**WaveShield**, 90.71% Macro-F1) used *"AM-Softmax plus focal
loss"* ([survey/09](../survey/09-challenge-playbooks.md)). That is **not evidence**: the organizers
report **no ablation**; the loss is confounded with a 3-member W2V-BERT 2.0 ensemble, LoRA staging
and six augmentation families; and the metric is **Macro-F1 at a fixed threshold** — exactly what
margin and focal losses buy. The organizers' own cross-team summary names *"self-supervised
representations, data augmentation, multi-crop inference, and structured fusion or routing"* — the
loss is not in the list.

## 3. One-class and margin losses (OC-Softmax, AM-Softmax, SAMO) — ❌ declined

★ The widely cited result — Zhang, Jiang & Duan, IEEE SPL 2021
([arXiv:2010.13995](https://arxiv.org/abs/2010.13995)) — is **ASVspoof2019 LA eval EER 4.69
(Softmax) → 3.26 (AM-Softmax) → 2.19 (OC-Softmax)**, LFCC + ResNet-18, no augmentation. Two
disqualifying caveats, both read from the paper:

- **No cross-dataset evaluation.** "Unseen attacks" means unseen algorithms *within* 2019 LA — same
  recording chain, same speakers, same corpus. It does not test our binding constraint.
- **The gain is essentially one attack.** Per-attack eval EER, A17: 23.48 / 13.45 / **9.22**. Strip
  A17 and the three losses are indistinguishable. AM-Softmax is *worse* than plain softmax on A18
  (0.20 → 4.27) — margin losses are not uniformly safe.

🔴 **On the one benchmark that genuinely tests generalization, one-class loses to plain softmax.**
★ XMUspeech, ASVspoof 5 ([arXiv:2509.18102](https://arxiv.org/abs/2509.18102)) — new generators,
adversarial attacks, codec variation, absolute EERs of 12–29% (our regime, not 2019 LA's sub-1%):

| System | Loss | **Progress EER** |
|---|---|---|
| B1 | **plain Softmax** | **16.32** |
| S2 / S3 | OC-Softmax | 24.88 – 25.83 |
| S4–S11 | SAMO | 17.07 – 27.44 |

Their hardest codec is speex at 8 kHz narrowband — our telephone slice.

⚠️ **Confound, stated honestly**: B1 uses the AASIST backbone while S2–S11 use HM-Conformer, so
loss and backbone are not fully separated. This weakens the claim from *"one-class loses"* to
*"one-class did not rescue a different backbone"* — still sufficient to decline, given the cost
(embedding-space redesign plus a 3-hyperparameter sweep over α, m₀, m₁).

⚠️ **A model-selection warning worth keeping**: S11 has the **best dev EER (12.32)** and a progress
EER of **21.56**, while B1 goes 15.20 → 16.32. One-class variants appear to overfit the tuning split
harder. We select on VAL ([validation/03](../validation/03-decision-protocol.md)); this is the
failure mode our sealed PROBE slice exists to catch.

⚠️ **Zero evidence in the music domain**, which carries 0.27 of our metric.

## 4. Multi-resolution frame supervision — ⚠️ downgraded from headline to optional

⚠️ [architecture/04 §4](../architecture/04-heads-and-pooling.md#4-multi-resolution-supervision--an-advantage-we-can-produce-but-should-not-lean-on)
calls frame-level ground truth *"the advantage we get for free"*, since we compose the corpus. The
advantage is real; **the measured benefit is not**.

★ Zhang, Wang, Cooper, Yamagishi & Evans, *The PartialSpoof Database and Countermeasures*, TASLP
31:813–825, 2023 ([arXiv:2204.05177](https://arxiv.org/abs/2204.05177)) — Table VIII, utterance EER %:

| Train | Strategy | LA eval (**out-of-domain**) | PS eval (in-domain) |
|---|---|---|---|
| PS | single-resolution, utterance-level | **0.77** | **0.64** |
| PS | multi-resolution, all levels | **0.90** | **0.49** |

**+0.15 points in-domain, −0.13 out-of-domain.** Both far below our floor, and the sign flips on the
axis we actually care about. Their loss is an **unweighted sum** over six resolutions
(20/40/80/160/320/640 ms) with **no coefficient ablation**, and they find multi-res is *worse* than
single-res at fine segment levels. Their own conclusion: *"CMs and training strategies should be
adapted to a specific goal."* Our goal is utterance-level ranking.

🔴 **Two findings from the same paper matter more than the headline.**

**(a) Fig. 5 / §VI-D is primary-source evidence for our composition trap.** They break EER down by
number of concatenation boundaries: it degrades sharply as boundaries fall, and is **worst — above
14% — at zero boundaries**. Their hypothesis, stated: *"The overlap-add-based concatenation may
bring in artifacts around the concatenated boundaries, which are expected to be useful for the CM."*
The model learns the mixing pipeline and collapses without it. ⚠️ **This is worse for us than for
them** — their eval set is also composed, whereas a genuine AI song or a real phone call with hold
music in the DACON test set carries no splice artifact at all. This upgrades
[data/02](../data/02-label-taxonomy.md#-the-mechanism-one-composed-fraction-shared-across-cells)'s
`f₅ = f₆ = f₇ = f₈` constraint from 🔷 our inference to ★ measured, and it retroactively justifies
[architecture/04 §4](../architecture/04-heads-and-pooling.md)'s 🔷 note that the frame-loss weight
is *"a shortcut-exposure knob, not just an accuracy knob."*

**(b) The inventors declined our exact problem.** Footnote 11: combining PartialSpoof with LA data
*"requires significant changes in the proposed CM structure so that model training can be
effectively conducted even in situations where parts of audio files in the training database do not
have segment-level labels, which is beyond the scope of this paper."* Only our **composed** files
carry frame labels; scraped real audio and natural AI songs do not. The group that invented
multi-resolution frame supervision would not run it under partial frame labels.

**Verdict**: not a headline capability. See [02 §4](02-the-loss.md) for how the frame term is
treated — the decision there is dominated by §5 below, not by this section.

## 5. Noise-robust losses (Lsoft, Lq, GCE, SCE, ELR, co-teaching) — ⚠️ one cheap run, no budget

Our corpus has two confidence tiers by construction: composed audio is **exactly** labelled,
scraped real audio is **reported** only ([pipelines/01](../pipelines/01-sample-contract.md)).
★ `[Freesound 2019, 1st]` used BCE on curated and **Lsoft β=0.7** on noisy
([kaggle/05 F1](../kaggle/05-transferable-playbook.md)), but publishes **no ablation**.

The one real audio ablation — ★ Fonseca et al., ICASSP 2019
([arXiv:1901.01189](https://arxiv.org/abs/1901.01189)), FSDnoisy18k, accuracy %:

| Loss | All data | Noisy only |
|---|---|---|
| CCE baseline | 71.6 ± 0.4 | 66.5 ± 0.6 |
| L_soft β=0.3 | **73.1 ± 0.6** | 66.8 ± 0.6 |
| L_q q=0.7 | **74.3 ± 0.7** | 66.7 ± 1.2 |

+1.0–2.7 **accuracy** points at an estimated **~40% label error rate** — far dirtier than our
scraped tier, and on a thresholded metric. Their useful structural finding: applying the robust loss
**only to the noisy tier** beat applying it everywhere.

⚠️ **Do not copy `LqLoss` from that repository.** Its form `loss = y_pred * y_true;
(1 − (loss+eps)**q)/q` is a constant when `y_true = 0` ⇒ **zero gradient on every negative**. Benign
in an 80-class sparse multi-label setting where it acts as a positives-only loss; **broken** in our
5-head binary setting. `l_soft` is safe — and its target blend must stay under `no_grad`, or it
degenerates into a self-consistency objective.

**Verdict**: worth one run at a single β, not more. Expected to be unmeasurable.

## 6. Reinforcement learning — ❌ rejected

Recorded because the rejection is 2nd-stage report material
([competition/03](../competition/03-evaluation.md)), not because it was close.

The instinct is sound: EER and AUC are non-differentiable, and BCE is a proxy. But the remedy for a
non-differentiable ranking metric is a **differentiable ranking surrogate**, and §1 above shows even
that is worth ~0 at our class balance. RL would be a **higher-variance estimator of the same
objective**:

| RL requires | We have |
|---|---|
| A sequential decision process | One forward pass, one score per file |
| An action space worth exploring | A scalar output already trained by gradient descent |
| A reward not expressible as a differentiable loss | A reward (EER) with known differentiable surrogates |
| Sample budget for policy-gradient variance | ~10 engineer-days total, no corpus yet |

Policy gradient would estimate, at far higher variance, a quantity that pairwise ranking already
estimates cheaply — and pairwise ranking measures **0.00 EER improvement** (§1). There is no
sequential structure, no exploration problem, and no environment.

🔷 The only RL-shaped framing with any merit is a **policy over augmentation or domain sampling**
(learned augmentation policy, or a bandit over hard domains). Even that is almost certainly
dominated by DOSS-style per-domain capping ([pipelines/02 §2](../pipelines/02-sampler.md)), which is
one line of sampler config with a **measured** result behind it (0.2k h balanced → 2.77% EER vs
6.4k h naive → 3.29%). **Decline, and record the reasoning.**
