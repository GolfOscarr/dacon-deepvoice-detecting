# PROGRESS

**DACON 236749 — 딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회**
Updated 2026-09-07 · **22 days to LB close** (2026-09-29 10:00 KST) · 2nd-stage materials 2026-10-05

Docs: 58 files under [`docs/`](docs/README.md) · Code: [`metrics/`](metrics/AGENTS.md) — 1,584 lines, **60 tests green** on `main`

---

## Phase status

| Phase | Topic | Status |
|---|---|---|
| **A** | Prior-art survey | ✅ done → [`docs/survey/`](docs/survey/README.md) |
| **+** | Kaggle intelligence | ✅ done → [`docs/kaggle/`](docs/kaggle/README.md) |
| **+** | Paper research | ✅ done → [`docs/papers/`](docs/papers/INDEX.md) — ~75 indexed, 12 deep-read |
| **B** | Data strategy | ✅ planned, ⬜ **not executed** → [`docs/data/`](docs/data/README.md) |
| **C** | Validation design | ✅ designed → [`docs/validation/`](docs/validation/README.md) · ✅ metric pipeline shipped → [`metrics/`](metrics/AGENTS.md) · ⬜ fold builder + gates |
| **D** | Model architecture | ⬜ |
| **E** | Score fusion & calibration | ⬜ |
| **F** | Engineering / submission | ⬜ |
| **G** | Report & compliance | ⬜ runs throughout |

---

## Done

**Competition understanding**
- [x] Objective, 5 output columns, 8-cell label taxonomy
- [x] Metric decomposed: `0.9·ADS + 0.1·CPS`; effective weights File .45 / Music .27 / Voice .18 / presence .05×2
- [x] Rules: code submission, offline L4, 60 min, 3/day, Private = Public
- [x] All 6 talkboard threads + official answers

**Survey (A)**
- [x] SOTA per head; realistic ceilings (speech ~4–6% EER, music **46.4% EER cross-generator**)
- [x] AT-ADD 2026 = closest challenge; winners' architectures
- [x] 🔴 16 kHz standardization destroys the shortcut published music detectors rely on
- [x] 🔴 Naive separate-then-detect fails (38–94.7% FPR)
- [x] Model / dataset / generator / augmentation catalogs

**Kaggle (+)**
- [x] API configured (`KAGGLE_API_TOKEN` in `.env`, `.venv/bin/kaggle`)
- [x] 8 top notebooks pulled; code-level recipes extracted
- [x] 🔴 SED attention head + `0.5·clip + 0.5·frame_max` = our short-component pooling fix
- [x] 🔴 Stop-gradient distillation (0.898 vs 0.876) = our runtime fix
- [x] 🔴 Pseudo-labeling the test set is **forbidden** (rule 2.3); cross-domain MixUp is the legal substitute

**Papers (+)**
- [x] 5-agent broad sweep across 12 axes; ~75 papers indexed with links
- [x] 12 deep reads from primary text
- [x] 🔴 **DOSS**: 0.2k h domain-balanced → 2.77% EER vs 6.4k h naive → 3.29%. Our ~240 h pool is *not* the constraint
- [x] 🔴 **BR vs AG**: generator-diverse 0.74 AUC · speaker-diverse 0.69 · **balanced 0.82**
- [x] 🔴 **PC-Mix vs CompSpoof**: never separate-then-detect; skip separation *or* train it jointly
- [x] 🔴 **Broadcast monitoring**: at 8 kHz+AAC, F1 0.992→0.186 but **AUC only 0.998→0.775** — we're scored on ranking
- [x] 🔴 **ArtifactNet read in full → NOT adoptable** (needs 44.1 kHz, weights unreleased, KR patents pending); ideas still usable
- [x] Negative findings recorded: no recent speech/music discrimination work; no MSS confirmed at 16 kHz

**Data plan (B)**
- [x] 🔴 Crawling verdict: **blocked** by rules, not copyright
- [x] 4 pools → 8 cells; composition trap identified
- [x] Tiered catalogs: synthesis, augmentation, EDA
- [x] Preprocessing / filtering / salvage + **review gates G1–G8**
- [x] ~70-source inventory with license verdicts
- [x] 🔴 CompSpoof V2 found — public dataset with our exact label structure
- [x] 🔴 AI-Hub blocked by default; NIA is a 주최기관 → inquiry worth making

**Validation design + metric pipeline (C)** — [`docs/validation/`](docs/validation/README.md) · [`metrics/`](metrics/AGENTS.md)
- [x] Split scheme: **artifact-family**-disjoint (not model-name), 4 slices, size floors from an EER-noise simulation
- [x] Four-tier **metric register** — official / decision / diagnostic / guardrail
- [x] Gates **VG1–VG6**; decision protocol with a paired-bootstrap promotion rule
- [x] 🔴 LB decomposition: **4 marginal submissions** recover all 3 fake EERs *and* CPS — proved minimal
- [x] `metrics/` shipped and merged ([#1](https://github.com/GolfOscarr/dacon-deepvoice-detecting/pull/1)); usage guide in [`metrics/AGENTS.md`](metrics/AGENTS.md), snippets executed by the test suite
- [x] 🔴 **Never pool raw OOF scores across folds** — measured 0.1705 for a true 0.100; use the mean of per-fold metrics
- [x] 🔴 EER invariant to monotone transforms and class prevalence, **not** to cell composition within a class (0.034 → 0.297) ⇒ freeze the eval composition by seed
- [x] 🔴 **Cell and artifact_family determine the label**, so those slices need a shared contrast pool — a within-slice EER on them is undefined
- [x] 🔴 **Rule 2.4 forbids rank calibration** across the cohort ⇒ tie-freedom from float64 logits, never rank normalization
- [x] Saturation, not rounding, is the output risk: rounding to 2 dp is harmless, saturating the operating point took EER 0.0950 → **0.3017**
- [x] 3 submission-contract defects found by audit and fixed (column order; order hardcoded instead of read from `sample_submission.csv`; validator comparing a re-parsed approximation)

---

## Next — do in this order

**Now (blocking, days 1–2)**
- [ ] Post `[DACON 답변 요청]` — **only 2 left**: commercial-API terms, and **AI-Hub/NIA**. Q1–Q7 of ours were asked by another participant in [#417333](https://dacon.io/competitions/official/236749/talkboard/417333) on 2026-09-05 (⏳ unanswered) — watch, don't duplicate
- [ ] Email AI-Hub (safezone1@aihub.kr) re: NIA / competition use
- [ ] **G1** dummy-file forensics → `signal_chain.yaml`
- [ ] **G2** license audit of `docs/data/11-source-inventory.md` (top 10 first)
- [ ] `submit.zip` skeleton + trivial model → validate I/O, runtime, offline packaging (`metrics/submission.py` is ready to vendor)
- [ ] LB probe: all-constant 0.5 submission → **must score exactly 0.5000**. ⚠️ P0 is deliberately degenerate, so it must be written with `validate=False` — the VG5 resolution gate rejects a constant column and would otherwise block the first submission

**Highest-value single experiment**
- [ ] 🔴 `E-A1` **16 kHz survivability probe** — ArtifactNet's Table XI shows AI residual bandwidth ~291 Hz vs human ~1,996 Hz, which *looks* like it should survive an 8 kHz Nyquist, yet the paper insists 44.1 kHz is required. Settling this decides the whole music-head approach

**Then (C — validation, now design-complete → [`docs/validation/`](docs/validation/README.md))**
- [x] Split scheme: artifact-family-disjoint + source-disjoint, 4 slices (TRAIN/VAL/SHADOW/PROBE)
- [x] Metric harness spec + verified properties (all-constant = exactly 0.5000; saturation risk)
- [x] Decision protocol: 3 speeds, paired-bootstrap promotion rule, experiment ledger
- [x] Gates **VG1–VG6** defined with pre-committed thresholds
- [x] 🔴 LB decomposition solved: **4 marginal submissions** recover all 3 fake EERs *and* CPS
- [x] **Shipped `metrics/`** — `dacon` · `submission` · `aggregate` · `breakdown` + [`AGENTS.md`](metrics/AGENTS.md) usage guide. **60 tests** under the server's scikit-learn 1.8.0. Merged to `main` in [#1](https://github.com/GolfOscarr/dacon-deepvoice-detecting/pull/1)
- [x] **Audited and verified end-to-end** — `eer` fuzzed against exact rational arithmetic over 2,991 cases (**zero** genuine disagreements); a synthetic 1,200-file run in the real `submit.zip` layout scores the constant probe at **exactly 0.5000** and recovers the 4-way LB decomposition to ~1e-16
- [ ] **Implement** `folds.parquet` builder + VG1 assertions ← next code
- [ ] Wire VG2/VG3 (`E-S2`, `E-A2`) to run per-experiment, not ad hoc
- [ ] Close the **version skew**: suite runs under scikit-learn 1.8.0 but Python 3.12 / numpy 2.5.3 / pandas 3.0.5, against the server's 3.11.15 / 1.26.4 / 2.0.3
- [ ] 🔴 Resolve the **music-family shortfall** below before any fold is built

**Then (B execution — days 3–12)**
- [ ] Acquire top-5 sources: CompSpoof V2, MLAAD, MUSDB18-HQ, Codecfake, ASVspoof21 LA
- [ ] 🔴 Apply **DOSS**: use `source × generator` as domain key, **cap per fake domain** (start N_c≈500) instead of taking everything
- [ ] 🔴 Scale **real-source diversity alongside** generator count (BR/AG balance), not after it
- [ ] Test-chain normalizer + curation sidecar layer
- [ ] `E-A1` **16 kHz survivability probe** ← tests the premise the music head rests on
- [ ] T3 resynthesis twins; generator breadth (≥20 voice, ≥5 music families)
- [ ] 4-way voice×music grid; REAL-processed slice
- [ ] Composition + augmentation pipeline (on-the-fly, seeded)

**Then (D/E/F)**
- [ ] Presence heads first (PANNs, cheap, 0.10 weight) → ship a real submission
- [ ] 🔴 **Three parallel branches on the mixture** (PC-Mix layout) — voice / music / file — with **joint training across branches** (the largest single gain in both component papers)
- [ ] Specialist frontends: speech SSL for voice, general-audio SSL for music (CompSpoof: don't reuse the speech encoder on the non-speech head)
- [ ] Try **codec-aware training phase** (ArtifactNet: FMA hard-negative FPR 98.7% → 8.0%)
- [ ] If a residual/denoise channel is used, **bound the mask** (unbounded degenerates to passing the input through)
- [ ] SED attention head, GeM freq pooling, clip+frame-max loss
- [ ] `FILE_FAKE_PROB`: noisy-OR vs learned head vs confidence-gated aggregation
- [ ] Cross-condition score calibration (pooled-EER comparability)
- [ ] Runtime budget: ≤3.0 s/file, offline weights, per-file try/except fallback

**Throughout (G)**
- [ ] Re-run `dacon-talkboard-sync` skill every 1–2 days
- [ ] Provenance ledger from the first download
- [ ] Ablation log + interpretability artifacts (70 of 100 2nd-stage points)
- [ ] Training code reproducing the Private score

---

## Open decisions — need your call

- [ ] **DACON questions** — confirm the 5 before posting (4 in [`survey/10`](docs/survey/10-open-questions.md) + AI-Hub)
- [ ] **CtrSVDD** CC BY-NC-**ND** — accept the risk or self-generate only? (307 h singing @16 kHz at stake). ⏳ #417333 Q5 asks exactly this — may resolve itself
- [ ] ⚠️ **#417333 Q1/Q6 could invalidate planned work** — codec-resynthesis labelling (T3 twins) and whether mixed public data must be shipped as files (on-the-fly composition). Monitor before building either
- [ ] **G6** preprocessing policy: channel handling, loudness normalization, silence trimming
- [ ] 🔴 **Music generator families: raise the floor from ≥5 to ≥8** — at 5, a family-disjoint 5-fold puts *one* family per validation fold and the sealed PROBE slice cannot be carved out at all, on the highest-weighted component head (0.27). Costs more Phase-B synthesis. [`validation/01`](docs/validation/01-split-scheme.md#-the-music-head-cannot-support-the-planned-split)
- [ ] **Korean slice size** — unknown test-set language composition
- [ ] **Review packs** — build the listenable HTML threshold-review generator now, or at first threshold?
- [ ] ⚠️ **ArtifactNet patents (KR + PCT)** cover bounded-mask residual extraction and codec-invariant training. Reimplementing those specific methods in a Korean government competition may warrant a legal look — or we simply avoid that exact formulation

---

## Pinned facts

| | |
|---|---|
| Metric | `Score = 0.9·ADS + 0.1·CPS`; all-constant submission = **exactly 0.5000** |
| Weights | File .45 · **Music .27** · Voice .18 · V-present .05 · M-present .05 |
| Private | = Public at close. No shakeup, no protection against LB overfit |
| 2nd stage | Top 15 advance; LB worth only **30 of 100** final points |
| Runtime | 1,200 files ≤ 60 min on one L4 = **3.0 s/file**, offline |
| Submissions | 3/day; runtime errors count, install errors don't |
| Test audio | 4–60 s, 16 kHz, mono+stereo, MP3/WAV/FLAC, some telephone-channel |
| Forbidden | Pseudo-labeling test data · cross-file statistics · non-redistributable data |
| Data volume | **Domain balance beats hours** — 3% of data can win (DOSS) |
| Separation | Never as a frozen preprocessor. Skip it, train it jointly, or use it as a residual *teacher* |
| 16 kHz | Kills most published music detectors. But **ranking metrics degrade far less than thresholded F1** |
| LB noise floor | 1,200 files ⇒ **±1.7 pts** EER on the File head, **±2.5** on each component head. Sub-1-point LB moves are not evidence |
| VAL size floor | **≥1,200 per class per masked pool** to resolve a 1-pt gap at 95% (paired) |
| Aggregation | **Mean of per-fold EER.** Pooling raw OOF scores across folds gave 0.171 for a true 0.100 |
| Invariance | EER is invariant to monotone transforms and to class prevalence — **not** to cell composition within a class (0.034 → 0.297) |
| Rule 2.4 | Per-file independence — cross-file normalization and **rank calibration are forbidden** in `script.py` |
