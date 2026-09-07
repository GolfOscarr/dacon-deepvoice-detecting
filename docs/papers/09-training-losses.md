# 09 — Training Losses

Card-level entries for this axis live in the flat table at
[INDEX.md](INDEX.md) — search this file's section heading there.

No papers on this axis were selected for deep reading in the first pass; the deep-read budget went
to [02 music](02-music-detection.md), [04 component-level](04-component-partial.md),
[05 generalization](05-generalization.md) and [06 robustness](06-robustness-channel.md), where our
open questions were concentrated.

**Promote to deep read if:** we commit to a design decision that depends on this axis.

---

### ⭐ D10 · TFPARN / Training-Efficient Transformer Anti-Spoofing (2026) — partial read
[arXiv](https://arxiv.org/abs/2606.02980) · ★ abstract-level

**Contribution** — Combines **focal classification loss + pairwise ranking loss**, explicitly aimed
at EER/threshold metrics, with attention pooling for utterance-level representations. RawBoost +
test-time augmentation.

**Result** — minDCF **0.2430**, EER **12.52%**; inference **0.79 ms/utterance**, 1.4 GB memory, vs
re-implemented AASIST and RawNet2 baselines. Ablations confirm pairwise loss, focal loss and
attention pooling each help — but the per-component numbers are not in the abstract.

**For us** — The *idea* is right and matches what our metric demands: **EER is pure ranking, so a
pairwise ranking loss optimizes it directly**, and [LLM-Detect-AI 2024, 1st] arrived at the same
conclusion independently ([kaggle/01](../kaggle/01-ai-content-detection.md)). But 12.52% EER is
weak in absolute terms against ~4–5% SOTA, so **adopt the loss, not the system**. The 0.79 ms
inference figure is notable against our 3.0 s/file budget.

See also, on the same idea: [Online AUC Optimization via Second-order Surrogate Loss](https://arxiv.org/abs/2510.21202)
and [Ensemble Learning for AUC Maximization](https://openreview.net/forum?id=kbxjkoF42x).
