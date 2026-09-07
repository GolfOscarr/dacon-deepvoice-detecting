# 02 — Pretrained Models We Can Adopt

**Adopting pretrained weights is the default, not the exception.** Rule 2.1 explicitly permits
사전학습 모델, and the whole field agrees on why: ★ ASVspoof 5's organizers found *"most of the
well-performing submissions use features extracted by pre-trained self-supervised learning (SSL)
models"*, and the survey rates **SSL frontend choice ⭐⭐⭐⭐⭐ against backbone architecture ⭐⭐**
([survey/01](../survey/01-sota-speech.md)). Training a frontend from scratch would be the single
worst use of our 22 days.

This file is the menu. Every entry is filtered by [01](01-design-envelope.md): loadable offline,
redistributable to DACON, inside the zip budget, and affordable on an L4 after the compression in
[06](06-compression.md).

⚠️ **Every licence below is unverified unless marked ★.** Check the model card before use.
A licence that forbids third-party provision makes an asset unusable *at all* here, not merely
unshippable ([01 §1.3](01-design-envelope.md#13-shipped-weights-are-part-of-the-deliverable-)).

---

## The case for two frontends

🔴 The decisive evidence is CompSpoof's stated limitation: *"environment anti-spoofing
consistently performs worse than speech anti-spoofing"* — their XLSR-AASIST detector, excellent on
the speech component, was measurably weaker on the non-speech component of the same mixture
([papers/04](../papers/04-component-partial.md)). A speech SSL encoder is trained to discard
exactly the non-speech structure the music head needs.

The AT-ADD Track 2 winner reached the same conclusion independently, using **wav2vec2-XLSR for
speech and EAT-large for non-speech** ([survey/09](../survey/09-challenge-playbooks.md)).

So: **one speech frontend for the voice branch, one general-audio frontend for the music branch.**
[01 §2](01-design-envelope.md#2-the-compute-ceiling) would have made that unaffordable at full
depth; layer truncation makes it affordable ([06](06-compression.md)).

🔷 The counter-consideration we accept: two frontends means two representations that the file
branch must reconcile, and 🔴 the pooled File EER punishes score-scale mismatch
([01 §3.4](01-design-envelope.md#34--overlapping-classes-and-one-pooled-ranking)). Our answer is
that the file branch reads *features* from both, not *scores* from both
([04](04-heads-and-pooling.md#5-the-file-head)).

---

### 🔴 The evidence asymmetry nobody should forget

The music branch carries **0.27 directly, plus most of the 0.45 file head** — music-fake is the
component our file score will most often hinge on. And essentially **every load-bearing technical
result in this directory was measured on speech**:

| Technique | Measured on | Used for |
|---|---|---|
| Layer truncation / probed-layer selection | XLS-R, **speech** anti-spoofing | Both frontends ([06](06-compression.md)) |
| Meta-learned LoRA, 8.84 → 5.30% EER | **speech** | Both branches |
| AASIST graph-attention backend | **speech** | Both branches |
| Stop-gradient distillation | bird audio (BirdCLEF) | Both branches |
| Joint component training (PC-Mix, CompSpoof) | speech + environmental sound | The whole design |

⚠️ None of it is measured on AI-music detection, and the one place our literature *does* apply a
speech-style recipe to music — MERT-AASIST — produced the **46.4% cross-generator EER** that is the
worst number in our survey.

🔷 This does not invalidate the design; it is the best evidence available and the alternative is
nothing. But it means **every one of these techniques must be re-validated on the music head
specifically**, not assumed to transfer from the voice head where it will be easier to demonstrate.
Where the two heads disagree, believe the music head — it is worth more.

---

## A. Speech SSL frontends → the voice branch

| Model | HF id | Params | Licence | Verdict |
|---|---|---|---|---|
| **wav2vec2-XLS-R 300M** | `facebook/wav2vec2-xls-r-300m` | 300M | ☆ Apache-2.0 (verify) | ✅ **Default choice.** The standard anti-spoofing frontend; 436k h / 128 languages. Truncatable. Also 1B / 2B variants for teacher use |
| **W2V-BERT 2.0** | `facebook/w2v-bert-2.0` | ~580M | ⚠️ **V3 — unverified**; Seamless components vary, some CC-BY-NC | ⚠️ AT-ADD Track 1 winner's frontend. Strongest candidate *if* the licence clears. Blocked on [survey/10 V3](../survey/10-open-questions.md) |
| **WavLM Large** | `microsoft/wavlm-large` | ~317M | ☆ MIT-family (verify) | ✅ Strong alternative; the MoE anti-spoofing work (D8) is built on it |
| **Alethia** | [arXiv](https://arxiv.org/pdf/2605.00251) | — | unverified | ⚠️ 2026, claims a foundational voice-deepfake encoder generalizing across tasks **including singing** — which matters because our rules class vocals as voice. Not read in depth; card only |

🔷 **Our ordering: XLS-R-300M first**, because it is the best-documented, the layer-selection
evidence in [06](06-compression.md) is measured *on it specifically*, and its licence is the least
likely to surprise us. Promote W2V-BERT 2.0 only if V3 clears **and** a probe shows it wins.

☆ A useful tiebreaker exists and we should use it: *A SUPERB-Style Benchmark of SSL Models for ADD*
(2026, [arXiv](https://arxiv.org/pdf/2603.01482)) is a leaderboard comparing SSL encoders for
countermeasures. It is card-level in our index — **read it before finalizing the frontend**
([09 B1](09-open-questions.md)).

---

## B. General-audio SSL frontends → the music branch

| Model | Source | AudioSet mAP | Licence | Verdict |
|---|---|---|---|---|
| **SSLAM** | [hf](https://huggingface.co/ta012/SSLAM) · [code](https://github.com/ta012/SSLAM) | **0.502** | unverified | ⭐ **Best fit on paper.** ICLR 2025, self-supervised on audio **mixtures** with a Source Retention Loss — explicitly built for polyphonic overlapping sources, which is our 혼합 case exactly. Drop-in for EAT weights |
| **EAT (large)** | [code](https://github.com/cwx-worst-one/EAT) | ~0.48–0.49 | unverified | ✅ AT-ADD Track 2 winner's **non-speech branch**. Proven on our exact task shape |
| **BEATs** | [unilm](https://github.com/microsoft/unilm/blob/master/beats/README.md) | ~0.48 | ★ **MIT** (`BEATs_iter3_plus_AS2M.pt`) | ✅ **The licence-safe default.** AT-ADD winner's router. If a licence question blocks everything else, this still ships |
| **OpenBEATs** | [arXiv](https://arxiv.org/pdf/2507.14129) | — | fully open | ✅ Explicitly a licence-safe reimplementation. The fallback behind BEATs |
| **MERT v1** | `m-a-p/MERT-v1-95M` / `-330M` | — (MIR, not tagging) | ⚠️ **V4 — likely CC-BY-NC** | ⚠️ Our only dedicated *music* encoder, but three warnings: trained at **24 kHz, untested at 16 kHz**; licence unverified; and **MERT-AASIST is the 46.4%-EER cross-generator failure** in MusicDET |
| **CLAP** | LAION / MS | — | unverified | ☆ TISMIR got F1 > 0.96 in-distribution from CLAP embeddings + SVM — but that result rides the production artifacts 16 kHz destroys ([survey/02](../survey/02-sota-music.md)) |

🔴 **The non-obvious call: prefer a general-audio encoder over the dedicated music encoder.**
G2 in [survey/10](../survey/10-open-questions.md) records this as open, but the evidence leans one
way — the AT-ADD winner chose EAT-large for non-speech rather than MERT, and MERT's only appearance
in our music literature is as the backbone of the worst cross-generator result we found.

🔷 **Our ordering: SSLAM, then EAT, with BEATs as the licence-safe floor.** SSLAM's mixture
pretraining is the property most specific to our problem, and it is the highest-mAP option; BEATs
is the one we can commit to today without a licence answer. Decide by probe, not by mAP
([09 B2](09-open-questions.md)).

---

## C. Presence heads

Combined weight **0.10**, metric ROC-AUC — an easier metric than 1−EER. ★ There is **no dedicated
2024–26 speech/music discrimination literature**, which the survey reads (correctly) as
reassurance that general AudioSet tagging is the right proxy
([papers/08](../papers/08-tagging-separation.md)).

| Option | Verdict |
|---|---|
| **PANNs CNN14** — **preinstalled** (`panns-inference==0.1.1`, `torchlibrosa==0.1.0`), mAP 0.439 | ✅ **Ship this on day one.** Zero packaging risk, ample for a 0.10-weight ranking task. ⚠️ Its checkpoint still downloads on first use — vendor it |
| **Two more SED heads on our own audio branch** | 🔷 **The intended end state.** We *compose* the corpus, so we know presence labels exactly for every training file — better supervision than AudioSet transfer, and near-zero marginal inference cost since the features already exist |
| BEATs / SSLAM as a dedicated presence model | ❌ Not worth a separate forward pass for 0.10. If we run one anyway as the music frontend, take the heads for free |

🔷 The sequencing matters more than the choice: **PANNs is how we get a valid, non-degenerate
submission on the board before the real model exists.** Keep it as a permanent baseline and as an
ablation row in the report.

### 🔴 Sung vocals are the failure case, and PANNs is likely bad at it

The rules class **vocals as voice** ([competition/01](../competition/01-overview.md)), so an
ordinary song must fire **both** presence heads and is 혼합 by definition — the single most common
mixed case there is.

⚠️ AudioSet tagging conflates singing with `Music` and does **not** reliably fire `Speech` on sung
vocals. So the day-one PANNs baseline probably fails on precisely the case the competition is built
around, and it will fail *quietly*: `MUSIC_PRESENT` will look fine while `VOICE_PRESENT` misses.

🔷 Mitigations, in order of cost: sum the `Speech` **and** `Singing` (and related vocal) AudioSet
logits rather than reading `Speech` alone; add a VAD prior; or train our own presence heads on
composed audio where we know the answer. **One song is enough to check** — do it before trusting
the baseline, not after.

⚠️ This changes what "shippable day-one presence" means, and it is a better argument for training
our own presence heads than the marginal-mAP argument above.

⚠️ Edge cases decide these heads more than model capacity does — applause/crowd is a classic
false positive for "music", and a 60 s file with 3 s of music must still score high on
`MUSIC_PRESENT`. That is a **pooling** requirement (max/attention, not mean), not a backbone
requirement ([04](04-heads-and-pooling.md)).

---

## D. Backends and heads

| Backend | Licence / risk | Verdict |
|---|---|---|
| **AASIST / AASIST3** | ★ open, pure PyTorch | ✅ **Default.** Graph-attention; used by *both* AT-ADD winners; 0.83% EER (19LA) as a standalone |
| **SED attention head** (GeM freq pool → attention over time) | our own code | ✅ **Adopted for all five heads** — see [04](04-heads-and-pooling.md). Matches our OR-over-segments labels and doubles as the interpretability artifact |
| **Bi-Mamba / Mamba-Attention** | ⚠️ **gated on V5** ([01 §1.1](01-design-envelope.md#11-the-10-minute-pip-budget-)) | ⚠️ Best published numbers, possibly uninstallable |
| **SLS classifier**, **Adapter-MFA**, **HA-MoE** | open | ☆ Multi-layer fusion over SSL layers. Cheap add-ons over a frozen frontend; worth an ablation, not a commitment |
| **MoE (D8)** | open | ❌ **Rejected on cost.** Macro EER 5.46 → 4.81% over 14 datasets, but parameters 178M → **329M (+85%)**. Wrong trade at our budget; the parameter-efficient alternatives below dominate it |

---

## E. Adapters and parameter-efficient tuning

These matter because they let one frozen frontend serve several branches without three copies of
its weights in the zip.

| Method | Result | Note |
|---|---|---|
| ☆ **Probing-guided layer selection** | 🔴 **4 layers of XLS-R-300M match the full model** at 1.34M trainable params | The cheapest large win available. Central to [06](06-compression.md) |
| ☆ **Meta-learned LoRA** | Avg EER **8.84 → 5.30%** vs full fine-tune, at **~1.1% trainable params** | Better than full fine-tuning *and* cheaper |
| ☆ Mixture of Low-Rank Adapter Experts | OOD EER 8.55 → 6.08% | Adapter-level MoE; keeps the frontend shared |
| ☆ Multi-Scale Convolutional Adapter | — | Cheaper than a full LoRA sweep |

🔷 **Read together, these say: freeze the frontend, and buy branch specialization with adapters
rather than with separate encoders.** That is what makes a two-frontend, three-branch design fit
in 10 GB and on an L4.

---

## F. Source separation — available, and we still say no

| Model | Availability | Verdict |
|---|---|---|
| **HT-Demucs** | **preinstalled** (`demucs==4.0.1` + `julius`, `diffq`) | ❌ **Not in the inference path.** See below |
| Mel-Band RoFormer / BS-RoFormer | weights would need shipping | ❌ Same reasoning, plus packaging cost |
| A separator **we train at 16 kHz** | our own | ⚠️ The only version the evidence supports, and only if trained *jointly* — see [03 candidate C](03-candidates.md) |

🔴 The preinstalled package list is a strong hint that the organizers anticipated a
separate-then-classify pipeline ([competition/02](../competition/02-submission.md)). **The
evidence says not to take the hint**, on four independent grounds:

| Ground | Evidence |
|---|---|
| Naive separate-then-detect is worse than not separating | CompSpoof **0.668 vs 0.827** F1 · PC-Mix speech **51.38% vs 29.35%** EER |
| Separation artifacts spread across *all* stems | Hybrid-stems: **38% FPR** (vocals), **94.7% FPR** (accompaniment) |
| No MSS method is confirmed native at 16 kHz | ★ Negative finding, [papers/08](../papers/08-tagging-separation.md). HT-Demucs is a 44.1 kHz model; our audio includes telephone-band |
| It costs runtime we would rather spend on coverage | [07](07-runtime-budget.md) |

⭐ **The one use that survives**: ArtifactNet distils **Demucs v4 residuals** as a Phase-1 *teacher*,
keeping separation entirely out of the inference path ([papers/02](../papers/02-music-detection.md)).
Since we have 8×H200 and separation costs us nothing at training time, this is genuinely
attractive — it is candidate **E** in [03](03-candidates.md).

---

## G. Off-the-shelf detectors — baselines to beat, not components

| Model | Verdict |
|---|---|
| `lofcz/ai-music-detector` | ⚠️ **The interesting one**: it implements "fakeprint" and **resamples to 16 kHz**, while the fakeprint paper itself excludes 16 kHz data as unsuitable. That contradiction must be resolved before trusting either ([09 A1](09-open-questions.md#a1---the-16-khz-survivability-probe-e-a1)) |
| `intrect/artifactnet` | ❌ **Not adoptable.** 44.1 kHz required and explicitly degraded at 16 kHz; training code and raw weights unreleased so we cannot adapt it; ⚠️ **KR + PCT patents pending** on the bounded-mask and codec-invariant methods, in a Korean government competition. Ideas are usable, the model is not |
| IRCAM Amplify | ❌ Commercial, and **misclassifies all Suno samples at 22.05 kHz** |

🔴 The general warning applies to this whole row: published AI-music detectors ride **production
artifacts** — Suno at 192 kbps, Udio at 320 kbps, 48 kHz upsampling — and our test set is
standardized to 16 kHz, which destroys every one of them ([survey/02](../survey/02-sota-music.md)).
**No off-the-shelf music detector should enter the design without passing `E-A1` first.**

---

## The shortlist

What we would build today, pending the probes in [09](09-open-questions.md):

| Role | Choice | Fallback |
|---|---|---|
| Voice frontend | XLS-R-300M, truncated | WavLM-Large · W2V-BERT 2.0 if V3 clears |
| Music frontend | SSLAM or EAT, truncated | **BEATs (MIT)** — the licence-safe floor |
| Branch heads | SED attention (GeM → attention → clip + framewise) | — |
| Backend | AASIST-style graph attention | plain attention pooling |
| Presence | SED heads on the music frontend | **PANNs, shipped day one** |
| Teachers (training only) | XLS-R-1B/2B, SSLAM/EAT-large, HT-Demucs residuals | — |
| Separation at inference | ❌ none | — |
