# 09 — Training Losses

Card-level entries for this axis live in the flat table at
[INDEX.md](INDEX.md) — search this file's section heading there.

✅ **The deep-read promotion criterion has fired and been discharged.** This file previously said
*"No papers on this axis were selected for deep reading… **Promote to deep read if** we commit to a
design decision that depends on this axis."* We committed to the objective, so four axes were read.
The design conclusions live in [`docs/training/`](../training/README.md); this file records the
**papers**.

🔴 **The headline finding is negative and it is structural.** This literature reports **minDCF,
Cllr, actDCF, Macro-F1 and accuracy** — all threshold- or calibration-sensitive. We are scored on
**EER and ROC-AUC**, which read only the ordering. When the *same ablation* also reports a ranking
metric, the effect is routinely an order of magnitude smaller and sometimes zero. ⚠️ **Check which
metric produced a number before adopting it** — that single habit is what separates a usable finding
from a worthless one on this axis.

---

## ⭐ D10 · TFPARN — Training-Efficient Transformer Anti-Spoofing (2026) — ★ full text read
[arXiv](https://arxiv.org/abs/2606.02980) · ASVspoof 5 Track 1 closed

**Contribution** — focal classification loss + pairwise ranking loss, explicitly aimed at
EER/threshold metrics, with attention pooling. RawBoost + TTA.

**🔴 Result — the ablation refutes the reason we adopted it.** Table VI, mean ± std over 3 seeds:

| ID | Loss | Pairwise | Pooling | **EER (%)** | minDCF | Cllr |
|---|---|:--:|---|---|---|---|
| 3 | CE | **No** | Mean | **12.91 ± 0.09** | 0.2662 | 1.8796 |
| 4 | CE | **Yes** | Mean | **12.92 ± 0.07** | 0.2561 | 1.6786 |
| 5 | Focal | Yes | Mean | 12.70 ± 0.36 | 0.2499 | 0.7232 |
| 6 | Focal | Yes | Attention | 12.52 ± 0.11 | 0.2430 | 0.9243 |

Isolated: **pairwise +0.01 EER (worse)** · focal −0.22 against a ±0.36 seed std · attention −0.18.
All three together move EER by 0.39. The paper states it: *"Adding the pairwise ranking branch
leaves the EER almost unchanged, from 12.91% to 12.92%, but lowers minDCF, Cllr and actDCF; the
ranking term acts on the decision cost and score ordering rather than on the equal-error point."*

⚠️ **The abstract disagrees with the table**, claiming the three components "all improve
performance". The table is the primary evidence.

**For us** — ❌ `ranking_weight = 0`, stage S4 dropped. Every gain here is calibration (Cllr
1.6786 → 0.7232), worth exactly zero under a ranking metric.

---

## ⭐ Zhang, Wang, Cooper & Yamagishi — Multi-Task Learning in Utterance- and Segmental-Level Spoof Detection (2021) — ★ full text read
[ASVspoof 2021 Workshop, pp. 9–15](https://www.isca-archive.org/asvspoof_2021/zhang21_asvspoof.pdf)

🔴 **The single most decision-relevant paper on this axis, and it is about *where* losses attach,
not which loss.** Table 5, PartialSpoof, utterance EER %:

| Model | Utt eval | Seg eval |
|---|---|---|
| **Utterance single-task** | **6.33** | 44.00 |
| Segment single-task | 7.69 | 15.93 |
| **UttU** — uni-branch MTL | **9.96** | 20.04 |
| **SegU** — uni-branch MTL | **7.04** | 17.75 |
| MulBS — binary-branch | 5.90 | 17.55 |

Supervising both levels through **one shared head** is 0.71–3.63 points *worse* than utterance-only.
§6.2: *"introducing another level's labels is meaningful but sharing the entire neural network can
reduce the original level of performance… conflict cases exist, such as how some bona fide feature
vectors in a spoofed trial might be updated by two gradients from opposite directions."* Fig. 4 is
titled *"Contradiction phenomenon in UttU/SegU"*.

⚠️ **Four transfer limits**, all read from the paper: their **footnote 6** scopes the mechanism to
*"models that use length-normalized vectors for classification, e.g. angular softmax"* (their
criterion is P2SGrad; ours is BCE on unnormalized logits); **no dispersion is reported** anywhere;
`UttBW`'s 5.66 used *"the best pre-trained model in the development set"* plus hand-pinned seeds
outside the standard range, so it is **not comparable** to MulBS; and their frontend is
LFCC + SELCNN + BiLSTM with no augmentation.

**For us** — 🔴 `clip_weight = 1.0` (clip-only) by default; our head is exactly the UttU/SegU
configuration. Ablation **T1**.

---

## ⭐ PartialSpoof database and countermeasures (TASLP 2023) — ★ read
[arXiv](https://arxiv.org/abs/2204.05177)

Six resolutions (20–640 ms), loss an **unweighted sum** across them, **no coefficient ablation**.
Table VIII, utterance EER %:

| Strategy | LA eval (out-of-domain) | PS eval (in-domain) |
|---|---|---|
| single-resolution, utterance-level | **0.77** | **0.64** |
| multi-resolution, all levels | **0.90** | **0.49** |

**+0.15 in-domain, −0.13 out-of-domain.** Multi-res is *worse* than single-res at fine segment
levels; their conclusion is that the countermeasure should be adapted to the goal.

🔴 **Two side findings matter more than the headline.** §VI-D / Fig. 5: EER by concatenation-boundary
count is **worst — above 14% — at zero boundaries**, attributed to overlap-add artifacts. That is
direct evidence for our composition trap ([data/02](../data/02-label-taxonomy.md#-the-composition-trap)).
And **footnote 11** names our exact situation as unsolved: training under *"parts of audio files in
the training database do not have segment-level labels"* is *"beyond the scope of this paper"*.

---

## OC-Softmax / one-class and margin losses — ★ read

**Zhang, Jiang & Duan, IEEE SPL 2021** ([arXiv](https://arxiv.org/abs/2010.13995)) — ASVspoof2019 LA
eval EER **4.69 (Softmax) → 3.26 (AM-Softmax) → 2.19 (OC-Softmax)**, LFCC + ResNet-18.
⚠️ **No cross-dataset evaluation**, and the gain is essentially **one attack** (A17: 23.48 / 13.45 /
9.22). AM-Softmax is *worse* than softmax on A18 (0.20 → 4.27).

🔴 **XMUspeech, ASVspoof 5** ([arXiv](https://arxiv.org/abs/2509.18102)) — the one genuinely-unseen
benchmark (new generators, adversarial attacks, codec variation, EERs 12–29%):

| System | Loss | Progress EER |
|---|---|---|
| B1 | **plain Softmax** | **16.32** |
| S2/S3 | OC-Softmax | 24.88 – 25.83 |
| S4–S11 | SAMO | 17.07 – 27.44 |

⚠️ **Confound**: B1 uses AASIST, S2–S11 use HM-Conformer, so loss and backbone are not separated.
The defensible claim is *"one-class did not rescue a different backbone"*.
⚠️ **Model-selection warning**: S11 has the best **dev** EER (12.32) and a **+9.24** dev→progress
collapse, against softmax's +1.12. We select on VAL.
⚠️ **Zero evidence in the music domain**, which carries 0.27 of our metric.

---

## Focal loss — ★ theorem, not an ablation

**Charoenphakdee, Vongkulbhisal, Chairatanakul & Sugiyama, CVPR 2021**
([arXiv](https://arxiv.org/abs/2011.09172)) — Thm 3 (classification-calibrated, its proof
establishing order preservation), Thm 5 (**not** strictly proper), Thm 11 (closed-form recovery
`Ψᵢ^γ`), Lemma 14 (`h^γ` **strictly increasing** on (0,1)).

🔷 Bridging their *across-class* statement to the *across-instance* ordering AUC reads is our own
step: in the binary case Thm 11 reduces to a fixed strictly-increasing scalar function of `η`.
⇒ **Focal and BCE have identical AUC/EER at the population optimum.**

---

## Label-noise-robust losses — ★ read

**Menon, van Rooyen, Ong & Williamson, ICML 2015** ([PMLR v37](https://proceedings.mlr.press/v37/menon15.pdf))
— §3 is titled *"BER and AUC are immune to corruption."* **Corollary 3**:
`AUC_{D_corr}(s) = (1 − α − β)·AUC_D(s) + (α+β)/2`, a strictly increasing affine map ⇒ **the argmax
over scorers is identical.** ⚠️ The guarantee is for **class-conditional, instance-independent**
noise; our contamination is plausibly instance-dependent, so this is a strong prior, not a proof.

**Fonseca et al., ICASSP 2019** ([arXiv](https://arxiv.org/abs/1901.01189)) — FSDnoisy18k, the one
real audio ablation: L_soft +1.0–1.5 and L_q +2.7 **accuracy** points at an estimated **~40%** label
error rate. Their structural finding: apply the robust loss **only to the noisy tier**.

⚠️ **Do not copy `LqLoss`** from the Freesound-2019 repository — its form gives **zero gradient on
every negative**, benign in an 80-class sparse multi-label setting and broken in a 5-head binary one.

---

## Multi-task loss weighting — ★ read, and the answer is "constants"

**Kendall & Gal, CVPR 2018** ([arXiv](https://arxiv.org/abs/1705.07115)) Table 1, CityScapes: equal
weights 50.1 IoU → **grid-searched constants 62.8** → learned uncertainty 63.4. ⚠️ The +12.7 comes
from mismatched loss *scales*; our five heads are all BCE on [0,1].

❌ **Negative results on adaptive weighting are strong**: Kurin et al. NeurIPS 2022
([arXiv](https://arxiv.org/abs/2201.04122)) — unitary scalarization *"matches or improves upon"*
complex multi-task optimizers; Xin et al. NeurIPS 2022 — MTO methods *"do not yield any performance
improvements beyond what is achievable via traditional optimization"*.

⚠️ **No study tests loss-weights ∝ evaluation-metric weights.** That question is settled by argument
([training/02 §4](../training/02-the-loss.md#4-per-head-weights--metric-proportional-by-argument)).

---

## Ranking / AUC surrogates — ★ read

*Understanding the Ranking Loss for Recommendation with Sparse User Feedback* (KDD 2024,
Criteo/DCNv2) — the gain from a RankNet term scales with positive sparsity: **+0.095% relative AUC
at 3.3% positives, +0.020% at 25.6%**. The mechanism is **vanishing gradients on negatives** under
BCE (gradient ∝ `p̂`).

🔷 The rule this yields, and it is narrower than "calibration losses are worthless": **losses that
reshape the score scale are worth zero; losses that repair BCE's imbalance pathology are worth
something proportional to the imbalance.** Genuine ranking gains do exist at extreme imbalance
(AUC-margin: PatchCamelyon AUC 0.8394 → 0.8896; pAUC-L: speaker-verification EER 3.04 → 2.23).
🔴 **We compose our corpus, so we decline the imbalance instead** — sampler constraint **C1**.
