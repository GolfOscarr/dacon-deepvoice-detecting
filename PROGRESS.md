# PROGRESS

> **Historical (last updated 2026-09-11).** The competition closed 2026-09-29: **28th place, top 3 %,
> score 0.86669**. The final state is in [README.md](README.md) and [docs/training/17-final-runs.md](docs/training/17-final-runs.md).

**DACON 236749 — 딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회**
Updated 2026-09-11 · **18 days to LB close** (2026-09-29 10:00 KST) · 2nd-stage materials 2026-10-05

Docs: 87 files under [`docs/`](docs/README.md) · Code: [`metrics/`](metrics/AGENTS.md) + [`models/`](models/AGENTS.md) + [`training/`](training/AGENTS.md) — **927 tests green** on `feat/smoke-training-verification` (898 on `main`)

---

## Phase status

| Phase | Topic | Status |
|---|---|---|
| **A** | Prior-art survey | ✅ done → [`docs/survey/`](docs/survey/README.md) |
| **+** | Kaggle intelligence | ✅ done → [`docs/kaggle/`](docs/kaggle/README.md) |
| **+** | Paper research | ✅ done → [`docs/papers/`](docs/papers/INDEX.md) — ~75 indexed, 12 deep-read |
| **B** | Data strategy | ✅ planned · 🟡 **executing** — 9 sources / ~239 GiB in S3, 20 more queued → [`docs/data/12`](docs/data/12-acquisition-status.md) · 🔴 **pool D (fake music) has no acquisition path** |
| **C** | Validation design | ✅ designed → [`docs/validation/`](docs/validation/README.md) · ✅ metric pipeline shipped → [`metrics/`](metrics/AGENTS.md) · ✅ fold builder + VG1–VG6 wired → [`training/`](training/AGENTS.md) |
| **D** | Model architecture | ✅ designed → [`docs/architecture/`](docs/architecture/README.md) · ✅ implemented → [`models/`](models/AGENTS.md) · ✅ **candidate A trains on real BEATs weights** · ⬜ candidate B unbuilt (C1 licence), truncation depth unmeasured |
| **D+** | Training & data pipeline | ✅ designed → [`docs/pipelines/`](docs/pipelines/README.md) + [`docs/training/`](docs/training/README.md) · ✅ implemented → [`training/`](training/AGENTS.md) · ✅ **S1→S2→S3 run on real audio, measurement chain verified** · ⬜ no entrypoint; corpus is a placeholder |
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

**Architecture design (D)** — [`docs/architecture/`](docs/architecture/README.md)
- [x] Design envelope: the hard gates (10-min pip, offline weights, shippable licences, rules 2.3/2.4) that eliminate architectures before any accuracy argument
- [x] 🔴 **Training compute is free, inference is scarce** (8×H200 → one L4 at ~10× real-time). Every large model becomes a *teacher*, never a shipped component
- [x] Pretrained catalog per role, with licence / size / 16 kHz verdict → [`02`](docs/architecture/02-pretrained-catalog.md)
- [x] Candidates A–G ranked; **chosen: 3 branches on the mixture, jointly trained, two truncated specialist frontends** (PC-Mix layout) → [`03`](docs/architecture/03-candidates.md)
- [x] 🔴 **Hard type routing rejected** despite winning AT-ADD Track 2 — their macro-F1 used per-type thresholds; our File EER pools every cell into one ranking, so a hard switch creates a score-comparability problem theirs never had
- [x] 🔴 **Rank averaging across the test set is forbidden** (rule 2.4) — the standard Kaggle ensemble is illegal here; and probability averaging across models is unsound under a ranking metric until each model is calibrated offline on our own data
- [x] 🔴 **Combine at training time, not inference time**: teacher ensemble + stop-gradient distillation, plus checkpoint soup — ensembling at zero inference cost → [`05`](docs/architecture/05-multi-model.md)
- [x] 🔴 **Layer truncation** (4 probed layers of XLS-R-300M match the full model at 1.34M trainable params) is what makes two specialist frontends affordable at all → [`06`](docs/architecture/06-compression.md)
- [x] 🔷 **Full temporal coverage beats multi-crop sampling** — 4 crops of a 60 s file cover 33%, and `frame_max` cannot recover what was never seen
- [x] 🔷 Predicted binding constraint is the **6 vCPU decode path, not the GPU**; measurement protocol written → [`07`](docs/architecture/07-runtime-budget.md)
- [ ] ⚠️ Every runtime number is an unmeasured extrapolation until [`07 §4`](docs/architecture/07-runtime-budget.md) runs

**Model implementation (D)** — [`models/`](models/AGENTS.md), branch `feat/model-architecture`
- [x] `config` · `heads` · `frontends` · `model` · `losses` · `outputs` + [`AGENTS.md`](models/AGENTS.md) whose snippets the suite executes
- [x] **A and B are one class**, differing only by config — a test asserts they share branch structure and differ only in frontend count
- [x] 🔴 **Three rule-2.4 tests pass rather than being asserted in prose**: a file scored alone matches the same file inside a batch of longer, louder files; ranking over a canned set is identical solo and batched; two different pad fillings give the same score
- [x] Frontends normalise two genuinely different output shapes — `(B,T,D)` and an `(F',T')` patch grid — to one contract
- [x] 🔴 **Default squash changed sigmoid → softsign.** At logit scale 30 a float64 sigmoid keeps 314/500 distinct values and loses the ordering; softsign keeps 500/500 and preserves it
- [x] 🔴 **Measured the duration bias and it corrected the docs**: spread over 1–12 windows is `max` 1.63 · `top-k mean` 1.18 · `quantile` 1.08 · `mean` 0.004. Top-k mean was recorded as "mild" and is only ~28% better than max — every order statistic inherits the bias. `confidence_gated`'s good showing is an artifact: it equals `mean` on 99.7% of files
- [x] 🔴 **Fixed a third rule-2.4 violation — one I introduced while fixing the config sweep.** `bandpass` ran the rFFT over the *padded* batch tensor, so the frequency grid depended on padding width: **one sample** of padding moved the filtered waveform by up to 1.0 on unit-variance audio. The docstring asserted the opposite ("introduces no cross-file dependence"). Now filtered per row over its valid prefix; bitwise identical across padding
- [x] 🔴 **Fixed six more review findings**: GeM overflowed in fp16 (the shipped inference precision) above a feature scale of ~40, producing NaN attention; the learnable exponent could never recover from below 1 (clamp has zero gradient); the loss and inference had *separate* knobs for the same clip/frame blend, with the loss-side one mis-documented as frame supervision; the ranking loss ranked `clip` while inference ranks the blend; and `frame_resolutions_ms` was read nowhere — it escaped the ignored-field guard because that guard only walked `ModelConfig`
- [x] 🔴 **Fixed two head defects found by review** — both inherited unexamined from the BirdCLEF snippet, both correct in *its* regime and wrong in ours. `tanh` on the attention logits capped any single frame's weight at ~7.4/T, so at T≈3000 `clip_logits` was a **mean pool** — the exact failure the SED head exists to prevent — and `norm_att` was flat, so the interpretability artifact worth 15 report points did not exist. ⚠️ **Our own `whole_file` choice made it ~12× worse** (the notebook's clips were 5 s, T≈250). And GeM's clamp discarded **50.7%** of SSL features with zero gradient; I introduced that clamp to stop a NaN and fixed it by destroying information
- [x] 🔴 **Fixed a second rule-2.4 violation, found by independent review**: `align_time` interpolated over the *padded* axis, so the source→target mapping was a ratio of two padded frame counts and moved with the batch. Submitted `FILE_FAKE_PROB` drifted **3.5e-4 at 9 s of padding**, monotonically — latent only because both shipped configs use 50 fps for both frontends, which short-circuits the function. Now mapped in absolute time and clamped per sample. Residual 1e-8 is float32 kernel selection
- [x] 🔴 **Fixed a rule-2.4 violation in the submission path**: `clip_logits` were padding-safe but `frame_max` was not, so the *submitted* probability moved 0.519 → 0.847 for the same file depending on what shared its batch. The mask now travels with the head output. The earlier test passed because it asserted on `clip_logits` rather than on what we upload
- [x] 🔴 **Fixed 19 silently-ignored config fields** — `freeze` did nothing (the encoder was fully trainable) and all three `file_head.mode` settings produced identical output. Guarded by `test_no_config_field_is_silently_ignored`
- [x] 🔴 Fixed a stub bug found by writing the batch tests — adaptive pooling *stretched* short files across the padded width, so the frame mask described the wrong frames. Frames are now absolutely positioned
- [ ] ⚠️ **Real frontends are deliberately not wired** — gated on the licence verification in [`architecture/09 C1–C3`](docs/architecture/09-open-questions.md). `build_frontend` raises with that reason
- [x] Training loop and data loading **implemented** → [`training/`](training/AGENTS.md). Teacher wiring still needs the corpus
- [ ] ⚠️ **Six defects were found this session, none by a green test suite.** Two shapes recur: a test asserting an *adjacent* quantity (`clip_logits` instead of the submitted probability), and a component inherited from a source recipe that was correct in *its* regime and silently wrong in ours after a later decision changed the regime. Both are worth checking for deliberately rather than trusting coverage
- [x] **Reviewed by a separate fact-check and critique pass**; corrections recorded in the [`architecture README`](docs/architecture/README.md) rather than silently applied. The hard-routing rejection was re-argued from scratch, the inference blend was defeating our own saturation finding, `P0` had to split in two, and a claim about noisy-OR monotonicity was simply false
- [ ] 🔴 Open from review: tile vs **whole-file single pass** (deletes the duration-bias problem at ~1.6× cost); does distillation still work with a **frozen** frontend; add a binned **duration stratum** to `metrics/breakdown.py`
- [ ] 🔴 **`G1` dummy forensics now has a second, independent reason to be first**: per-file metadata (container, bitrate, channels, duration, encoder fingerprint) is legal under rule 2.4 and may separate REAL/FAKE almost for free — or be a pure CV mirage. If the leak is real it dominates every architecture decision ([`architecture/09 A5`](docs/architecture/09-open-questions.md))
- [ ] 🔴 **Build in the stated drop order** — P0-a → P0-b → **candidate A** → candidate B → codec stage → distillation → extras. ~10 days of modelling remain after the corpus build ([`architecture/08 §4b`](docs/architecture/08-training-recipe.md#4b--the-budget-nobody-costed-engineer-days))
- [x] 🔴 **A is stage one of B, not a fallback.** B strictly contains A — same SED heads, same masked losses, same joint multi-task training — so the A→B increment is one extra encoder, per-branch adapters, and a time-base alignment. "A then B" is strictly cheaper than "B, and A if B fails". And since A is *already* jointly trained, **B's marginal claim over A is frontend specialization alone, resting on a single citation** ([`architecture/03`](docs/architecture/03-candidates.md#-a-is-stage-one-of-b-not-a-fallback))
- [ ] ⚠️ **Promote B over A only under P1–P6** ([`validation/03 §3`](docs/validation/03-decision-protocol.md)). "B failed" is undefined against a ±1.7/±2.5 pt noise floor; on an inconclusive result the pre-committed tiebreaker favours **A**
- [ ] 🔴 **Check PANNs fires `VOICE_PRESENT` on a sung song** before trusting the day-one presence baseline — one song answers it; AudioSet conflates singing with Music
- [ ] ⚠️ **Re-validate every technique on the music head specifically.** Layer truncation, meta-LoRA, AASIST and the distillation recipe were all measured on *speech*, while the music branch carries 0.27 plus most of the 0.45 file head — and the one time our literature applied a speech recipe to music (MERT-AASIST) it produced the 46.4% cross-generator EER

**Training & data pipeline (D+)** — [`training/`](training/AGENTS.md), branch `feat/training-pipeline`
- [x] `spec` · `manifest` · `sampler` · `audit` · `folds` · `foldcheck` · `registries` · `render` · `collate` · `dataset` · `stages` · `checkpoint` · `loop` · `validate` · `synthetic`, plus [`AGENTS.md`](training/AGENTS.md) whose snippets the suite executes
- [x] 🔴 **`sample_spec()` is pure and label-free.** `SampleSpec` has no label fields — labels are derived properties of the cell, so no transform can reach them. Auditing a stream costs no audio decode
- [x] 🔴 **The reference cell mix was itself trapped when first published**: `P(mixed|FAKE)=0.667` vs `P(mixed|REAL)=0.300`, so "is a mixed file" predicted FAKE at 0.769. Re-solved; the shipped mix satisfies C1 and C3 simultaneously
- [x] Composedness is **one knob** (`f8`), conditional (`f8=0`) primary and strict (`f8=1`) its endpoint — not two code paths
- [x] 🔴 **Registries measure time-invariance at registration** rather than trusting a declaration: each step runs on fixed-seed broadband noise correlated in two windows, and a shift refuses registration *whether undeclared or wrongly declared*. A chirp probe wrongly accepted `pre_emphasis`; the two-window noise probe is required
- [x] 🔴 **The resume guarantee is bitwise and the state list is complete** — no hidden generator anywhere (`render` draws from `spec.rng` keyed on `(sample_id, epoch, seed)`, `bucket_batches` from a local seeded generator), no LR schedule, no gradient accumulation
- [x] S1→S3 stage runner, EMA, checkpoint soup, VG1–VG6 wired, leak tripwires. `rank_polish` **raises** — S4 is dropped, and a knob that validates and silently does nothing is its own defect
- [x] 🔴 **`quotable` cannot be earned by SKIPs.** The ledger row carries a `vg1..vg6` tri-state where **fail beats na beats pass**, so a gate reads `pass` only if every sub-check actually ran
- [x] **Refactor**: `loop.py` 1,456 → 386 lines split into `stages` / `checkpoint` / `loop` / `validate`; `folds.py` split builder-from-judge. `registries.py`, `audit.py`, `render.py` deliberately left whole — `registries` is the one module vendored into `submit.zip`, and splitting `audit` would separate each estimator from the invariant whose tolerance it justifies. All 283 emoji removed from the Python sources
- [ ] ⚠️ **Nothing here has run on real audio.** Every number is against synthetic corpora and stub models
- [ ] ⚠️ **Four shipped model configs say `runtime.precision: fp16` while `configs/train_joint.yaml` says `bf16`** — which precision a real run trains at is not currently obvious from the configs. Settle before T1
- [ ] **VG3 reports SKIP by name.** Unguarded until a corpus exists: label-independent TRAIN/VAL domain drift, and a corpus edit silently breaking ledger comparability (only the VG2 half of the re-trigger is detectable). VG1 A1–A7, the tripwires and VG4's T3 gap cover part of the same ground

**Acceptance check of the training pipeline** — five axes, audited against the code
- [x] **Objectives** — metric-proportional weights reach the loss, masks mirror the masked EER pools, `ranking_weight` 0, `clip_weight` read from the same field inference blends with
- [x] Critical: **the head-to-metric-weight mapping was untested.** Mapping both 0.05 presence heads to `file` (0.45) passed **749/749**. The File-Voice swap was caught only by an fp16 `GradScaler` backoff tripping an unrelated test's non-vacuity guard — so which mis-assignments the suite caught was decided by numerical overflow, not intent. Now asserted as the weight and target *in effect* per head
- [x] A partial `loss.weights` dict silently defaulted missing heads to **1.0** — `weights: {file: 0.45}` loaded clean and trained both presence heads at 20x their metric weight. The fallback is gone, and the check sits on `LossConfig.__post_init__` so `dataclasses.replace` and checkpoint round-trips cannot route around it
- [x] `p_c` and `w_eff` now logged per head — `docs/training/02 §4` forbids tuning `w_c` without them, and T2 could not be read as specified
- [x] **Data pipeline / fixed format** — one batch shape, enforced; `validate_manifest` now checks all four labels, the domain key and the family key. Zeroing `label_voice_fake` corpus-wide was accepted and took the voice family count **24 → 0** while `build_folds` emitted a full table
- [x] **YAML configuration** — `training/` had none; four run-shaping dataclasses (`SamplerConfig`, `RenderConfig`, `FoldConfig`, `LoopConfig`) now load from `configs/run_*.yaml` with exhaustive knob tables. `f8` is a config value at last, which is what "one knob, not a second code path" required
- [x] Critical: **validation had to land before YAML, not after.** A legal-looking cell mix put three of five heads outside C1 (0.890 / 0.918 / 0.818 against `[0.2, 0.8]`) and opened a C3 gap of 0.232, with nothing to say so — `audit_specs` is the only thing that checks, and the training path never ran it. C1/C3 are now enforced at construction
- [x] **Training algorithms** — clean, no caveats. S1 selects *modules* rather than zeroing loss terms, so weight decay and Adam momentum cannot move an untrained branch; `leak_tripwires` consumes the `MetricSet` rather than the frame so it cannot disagree with the number it guards; there is no EER implementation anywhere in `training/`
- [x] **Code structure** — `pass_plan` gives the loop and the audit one batch plan (they had diverged for every pass but pass 0 at seed 0, and by a different *list* under S3); `MANIFEST_SLICES` / `FOLD_SLICES` no longer collide, and `folds.py` no longer imports one and shadows it with a redeclaration
- [ ] ⚠️ Remaining: `training.loop` re-exports 28 names of which it owns 4, and `training/AGENTS.md` teaches 12 examples through it. `validate.py` at ~890 lines now carries three concerns — the shape `loop.py` had before it was split
- [ ] Critical: **a green suite is not evidence a check can fail.** An adversarial pass left **28 of 85 mutants alive** under 101 passing tests. Of the defects above, several were found by an agent asking whether it had tested something — including one where the fix's own docstring claimed an assertion could not fail, written without checking, and wrong

**🔴 What the pipeline sessions cost, and what actually found it** — none of the following was found by a passing test
- [x] **A trapped spec stream passed the entire audit clean**: transform *parameters* scored AUC 1.000 while I1 counted only names, and the whole `normalize` draw was read by nothing. Three checks could not fail at all, and I7 printed PASS for size floors implemented nowhere
- [x] **The pad *value* reached the submitted probability.** `frames_for` rounds up, so a row whose length is not a multiple of the 320-sample hop has a final frame that is part padding — and it is masked *in*. The real finding was the fixture convention: every padding test used `lengths = SR*4`, an exact multiple of the hop, so the four tests guarding this repo's most-repeated defect class had **never once exercised a partial boundary frame**
- [x] **A frozen epoch key made every training pass replay epoch 0** while all 97 loop tests stayed green. Found by hand-running a mutation, not by the suite
- [x] **The mid-epoch checkpoint dropped the fp16 loss scale**, so a resumed run replayed the scaler warm-up. With a backoff forced, max weight divergence **5.476e-07**. Invisible because no test set `checkpoint_every` non-zero *and* no test trained at fp16 — two independent holes, either of which alone would still hide it
- [x] **A perfectly separating file head tripped no tripwire** on the generator-disjoint split we actually validate on. L3 skipped claiming "L1/L2 cover this one"; L1 reads `eer_music`, L2 reads `eer_voice`, and neither reads `eer_file` — the 0.45-weight head
- [x] **The predictions-to-specs join was asserted by nothing.** Reversing each eval batch moved `score` 0.5548 → 0.4492 with **every gate green**, because `prediction_frame` paired predictions to specs positionally
- [x] **`test_no_loop_config_field_is_silently_ignored` was a textual grep** for `loop_cfg.<name>` — the guard against dead config knobs was satisfied by a mere mention, and 2 of 6 knobs slipped through it
- [x] **`synthetic.py` built its digest from Python's salted `hash()`**, contradicting its own "byte-identical corpora" claim — the exact hazard `spec.py` documents and avoids two files away
- [x] 🔴 **The lesson, stated once**: an adversarial mutation pass on `loop.py` left **28 of 85 mutants alive** under a green 101-test suite. Three of the defects above were found by an agent *asking itself whether it had tested something*, not by running anything. Budget a review pass per module; a green suite is not evidence that a check can fail


---

## 🔴 2026-09-08 — talkboard #417333 answered all seven questions

The single most consequential external update so far. Full text and quotes:
[`competition/05`](docs/competition/05-talkboard-qa.md).

**Against us — one planned source is retired**
- ❌ **S-S1 / T3 resynthesis twins are dead.** A1: *음성·음악 성분을 새로 생성하지 않는 후처리만
  적용된 경우 REAL로 간주합니다.* A codec round-trip, neural denoise or source separation of real
  audio **reconstructs** — it does not generate — so the output is **REAL**. We had this listed as
  the "cheapest family multiplier" for Pool B, which is exactly what made it dangerous: labelling
  it FAKE would have trained the model to invert the target on the slice the test set most likely
  contains ([`data/04`](docs/data/04-sources.md), [`data/05`](docs/data/05-synthesis-plan.md))
- ✅ The same answer **promotes the REAL-processed slice from optional to required** (S-S1R). The
  REAL class contains codec-round-tripped, enhanced and separated audio; a detector that never saw
  processed REAL will false-positive on it. ⚠️ This makes REAL the *harder* class — the naive
  reading "neural artifact ⇒ FAKE" is now wrong

**For us — two open bets settled**
- 🔴 **A5: ND data is usable**, augmentation included, if reproducible from 원본 파일 + 코드.
  Unblocks **Codecfake** (crown jewel #4, 32 GB), **ST-Codecfake** (39 GB), **CtrSVDD** (31.6 GB,
  260 h sung fake at 16 kHz), **SceneFake**, and ~63,000 ND tracks in FMA/Jamendo. Closes
  [`survey/10 V2`](docs/survey/10-open-questions.md)
- 🔴 **A6: on-the-fly composition is legal.** *가공 데이터는 원본 데이터와 재현 가능한
  코드·설정값·seed 등을 제출하면 됩니다.* The entire [`data/06`](docs/data/06-augmentation-spec.md)
  design rested on this. Consequence: `render(spec) == render(spec)` stops being an internal
  nicety and becomes what the 2nd-stage submission is built on
- ✅ A2 confirms the label taxonomy; A3 confirms `PRESENT=1` at **any** duration (the strongest
  external support yet for the SED head and `0.5·clip + 0.5·frame_max`); A4 confirms masked EER on
  ground-truth PRESENT, as [`metrics/`](metrics/AGENTS.md) implements
- ⚠️ **A7: DACON will not adjudicate any dataset's licence.** The G2 gate in
  [`scripts/sources.yaml`](scripts/sources.yaml) is the only licence check anyone will run

**Still open**
- ⚠️ **Cell 9 was NOT settled.** A3 answered only the duration half. Whether files exist with both
  `PRESENT=0`, and whether AI-generated environmental sound is `FILE_FAKE`, remain unanswered

**#417344 — row independence is spot-checked, not automatic**
- 🔴 *운영진이 대회 기간 중 불시에 점검... 자동으로 이루어지는 방식이 아닙니다*, with rigorous
  verification after the competition on award candidates. **A valid leaderboard score is not
  evidence of compliance.** This is the strongest justification yet for the four rule-2.4
  batch-dependence fixes in [`models/`](models/AGENTS.md) — each would have scored normally and
  failed the post-hoc review

**#417336 — 2차 평가 delivery: no size cap, Google Drive confirmed**

---

## 🟡 2026-09-10 — corpus acquisition running

**7 sources, 122.6 GiB in `s3://<bucket>/dacon-deepfake-detection/data/raw/`**, each with a
provenance record carrying its licence verdict and a sha256 per artifact.
Full state: [`data/12`](docs/data/12-acquisition-status.md).

- ✅ **Pools A and E are covered** — Korean (Zeroth-Korean + Common Voice ko), English read
  (LibriTTS-R, LJSpeech), English crowd-sourced (Common Voice en, 88.1 GB), plus MUSAN noise and
  RIRS impulse responses. Enough for the presence heads and the augmentation chain
- ⬜ **Pools B, C and D remain empty.** 22 sources / 367 GB are cleared and queued; MLAAD is
  in flight
- 🔴 **Pool D (fake music) has no acquisition path at all.** It carries **0.27** of the metric —
  more than voice — and nothing in the queue fills it. That is ACE-Step generation on a GPU, and
  it is now the largest gap in the plan

**Tooling shipped** — [`scripts/fetch_to_s3.py`](scripts/fetch_to_s3.py) +
[`scripts/sources.yaml`](scripts/sources.yaml) (the G2 gate as data: the fetcher refuses any source
whose verdict is not `ok`, and refuses an `ok` verdict not marked `verified_at_origin`) +
[`scripts/filter_track_licences.py`](scripts/filter_track_licences.py) (per-track allowlists for
FMA and MTG-Jamendo). 42 tests in
[`tests/test_fetch_sources.py`](tests/test_fetch_sources.py).

**🔴 Five defects, none found by a test going red.** Every one surfaced from watching a transfer
run — the same pattern [`pipelines/05`](docs/pipelines/05-invariants.md) records for the training
loop:

- `curl --retry` with `-C -` re-requests from the offset fixed at process start, so Common Voice
  English went **backwards 74 GB → 45 GB** and could never converge
- `pkill -f` matched only the Python parent, leaving an **orphaned curl at `PPID=1`** writing into
  the same path as its replacement — two writers, different offsets
- those two together produced **118.3 GB against an expected 94.6 GB (125%)**: corrupt, and
  unresumable because `-C -` then asks past EOF. ⚠️ **Size was the only visible symptom** — which
  is why publisher checksums are now verified where they exist, and CV-English's final
  `sha256 6809228e…` matched MDC's published value
- a `finally` block deleted staging on failure, restarting a 66%-complete 88 GB transfer four times
- HF datasets resolved one URL at a time, flattening `fake/<lang>/<generator>/<file>` and
  destroying the **generator-disjoint split axis**

⚠️ **The HF API's `siblings` field undercounts by 5×** — 99,411 reported for MLAAD, 534,539 actual.
The gap was 174 GB against an assumed 30.

🔴 **MLAAD is capped at 30 files per generator directory ≈ 30.1 h**, matching the
[`data/04`](docs/data/04-sources.md) budget. The first value shipped was 300/dir = **301 h**, 10×
the budget and enough to make MLAAD ~80% of a 70 h Pool B — the exact imbalance DOSS argues
against. Selection is seeded and recorded in `_meta/selection.json`, because
[#417333 A6](docs/competition/05-talkboard-qa.md) makes reproducibility from originals + code +
seed what the 2nd-stage submission rests on
([`data/12 §4b`](docs/data/12-acquisition-status.md#4b--the-mlaad-cap-a-corpus-decision-not-a-download-setting)).

⚠️ **Confirm the bucket's region matches the training machine's.** Ingress was free; every read
back out is billed as egress, and training reads the corpus repeatedly. Retargeting is one env var
(`DACON_S3_BUCKET` / `DACON_S3_PREFIX`) and is cheaper before the corpus grows further.

ℹ️ Storage and compute are **personal resources** provisioned for this competition.

---

## ✅ 2026-09-11 — first real-audio training, and the measurement chain verified

Branch `feat/smoke-training-verification`, 6 atomic commits, 927 tests. The pipeline now trains end
to end on pretrained weights and the chain from manifest to submission CSV has been exercised.
**No competition-relevant model exists**: see the limits below before reading any number.

**What became possible**
- [x] 🔴 **BEATs wired** → candidate A trains. 71.5M params, 2.4M trainable (3.4%) via LoRA on 9
  truncated layers. `build_frontend` accepted only `stub` before this, so nothing in this repo had
  ever trained on a pretrained encoder. BEATs needed no licence answer — MIT, read at origin — which
  is exactly why [`architecture/02`](docs/architecture/02-pretrained-catalog.md) calls it the
  licence-safe floor. `LoRALinear` is new; `AdapterConfig` had said "nothing implements adapters yet"
  while a shipped config asked for `kind: lora`.
- [x] **A 2,177-row smoke corpus**, reproducible via [`scripts/build_test_corpus.py`](scripts/build_test_corpus.py):
  8 real sources across pools A/B/C/E, placeholder D, 28 grouping atoms, 17 artifact
  families, 64 speakers, 2 folds. All 13 audit checks and both folds' eval-side
  `I21`/`I7` pass.
- [x] **S1 → S2 → S3 all run on real audio** (S3 only after the codec fix below).
- [x] **Cross-fold aggregation, gates, tripwires and a submission CSV** all executed for the first time.

**Defects found — none by a test going red**
- [x] 🔴 **S3 was unrunnable on real audio.** `_codec_roundtrip` required exact length equality, and
  LAME leaves trailing padding on ~8% of real sample counts, so `codec_aware` raised on its first mp3
  draw (7.5% of samples). **The docstring's diagnosis was wrong** — it blamed a lost gapless header
  and a shifted waveform; cross-correlation peaks at lag **+0** and the header is present. Measured:
  header present 0–42 extra samples, header absent (pipe-encoded) 1169–1708. `_PAD_TOL = 576` sits in
  that gap.
- [x] 🔴 **`fps` was wrong by 8×** — configs said 50.0, copied from the wav2vec2 family; BEATs emits
  6.25. Nothing would have failed: the file branch would have concatenated misaligned evidence.
- [x] 🔴 **The obvious batched BEATs encode is a rule-2.4 violation** — it flattens the patch grid to
  one `T'·F'` sequence, so valid tokens are strided runs, not a prefix. One padded pass moves a 4 s
  file's features by **0.398**. Encoding per row is exact.
- [x] 🔴 **Upstream layerdrop is a generator outside our seeding** (`np.random.random()` per layer),
  which would have silently falsified the bitwise-resume claim. Forced to 0.
- [x] 🔴 **Bitwise resume had never worked on a GPU.** `train_stage` loads with
  `map_location=device`, so `torch.load` moved the RNG ByteTensors onto CUDA and `set_rng_state`
  rejected them. Invisible because `LoopConfig.device` defaults to `"cpu"` and the suite never probes
  cuda. Now verified bitwise on an H200, interrupted mid-pass.
- [x] 🔴 **The corpus had a split leak, and no check in the pipeline could see it.**
  `RIRS_NOISES/pointsource_noises/` is MUSAN's `free-sound` noise set **redistributed**: 88
  byte-identical pairs, same basename, different `file_id` and different `source_name` — so
  `grouping_atoms`' union-find could not link them and the same recording sat in TRAIN on the fold
  where its twin was in VAL. `validate_manifest` only tests `file_id.duplicated()`, never the hash,
  and the builder was computing every sha256 already while hardcoding `dup_group = None`. Now
  populated from those hashes; verified **0 shared hashes across TRAIN/VAL in either fold**.
  ⚠️ **The cascade is the lesson**: linking them correctly collapsed pool E from 4 groups to 2 and
  `build_folds` immediately refused — revealing that the *earlier* 2-fold table had been feasible
  **only because of the leak**. Fixed with genuinely independent audio already in the store
  (CompSpoof `env_sources/*/bonafide/`: EnvSDD, VGGSoundEnv), and `rirs_isotropic` was dropped
  outright: 79 of its 90 rows fell below the 4 s sampler floor, and real RIRs are convolution
  kernels, which is the same reason `simulated_rirs` was never taken.
- [x] 🔴 **A manifest missing a whole pool passes `validate_manifest`** — 1687 rows with no real music
  validated clean. And `source_name` at corpus granularity collapses every generator family into one
  fold group (9 → 1), which also validates and then makes `build_folds` infeasible at every fold
  count. Both now guarded in the builder.
- [ ] 🔴 **The pattern worth carrying forward: synthetic corpora, stub models and CPU are three axes
  of one blind spot.** Four of the above were unreachable by an 898-test suite *because* of what it
  runs on, not because of what it asserts.

**Measured, where the docs had extrapolations**
- [x] 🔴 **Throughput is stage-dependent, and [`architecture/07 §4`](docs/architecture/07-runtime-budget.md)
  is half right.** S2 `joint` is **GPU-bound** (47.7 samp/s; render 24%). S3 `codec_aware` is
  **decode-bound** (14.4 samp/s; render 75%, in `ffmpeg` subprocesses). Rendering is inline and
  single-threaded by design (`training/loop.py:193` names the `DataLoader` alternative). 🔷 A worker
  pool should recover most of the 3.3× on S3 — an estimate from this measurement, not a measurement.
- [x] **`I7` eval size floors PASS** at 6000 drawn val specs (1200/class/head; 600 specs gave ~200).
  First time that gate has returned anything but FAIL.
- [x] **`quotable` discriminates.** The run came out **NOT QUOTABLE** for a correct reason: a leak
  tripwire fired on fold 1. Injecting a `NOT_QUOTABLE` caveat voids a run regardless of gates, and a
  single fold reports `score_sd = 0.0` with `sd_is_a_measurement = False`.
- [x] **Rule 2.4 holds on the uploaded CSV.** Different neighbour content at fixed batch size moves a
  prediction by **0.0**, bitwise. ⚠️ But **batch size 8 → 32 moves it 6.8e-04** with identical
  neighbours — cuBLAS reduction choice, not a leak, and larger than the ranking resolution protected
  elsewhere. The submission must be written at a pinned batch size.
- [x] **Pass digests are distinct** across every pass of every stage (6/6, 10/10, 2/2); `set_epoch(0)`
  collapses them to 1.
- [x] **Version skew narrowed** (C5): the training venv is CPython **3.11.15**, the server's exact
  version, torch 2.7.1+cu128.

**Why no number here is a result**
- [ ] 🔴 **Every score measured before 2026-09-11's review pass came from the leaking corpus** — the
  split where 88 recordings could straddle TRAIN and VAL. They were already NOT QUOTABLE, so nothing
  claimed is retracted, but the 0.7776 two-fold figure should be discarded rather than carried
  forward; it was measured against a compromised split.
- [ ] ⚠️ **Pool D is a placeholder.** Nothing acquired contains fake *music*. CompSpoof V2 is not the
  answer — its second component is environmental sound, which [`data/11`](docs/data/11-source-inventory.md)
  states and [`data/12`](docs/data/12-acquisition-status.md)'s summary did not. The smoke corpus
  substitutes spoofed environmental audio, explicitly labelled; the leak tripwire caught it unaided
  (music EER 0.0201 vs a 0.03 floor). This is now **C7** in [`architecture/09`](docs/architecture/09-open-questions.md).
- [ ] ⚠️ **Durations reach 4–10 s, not 4–60 s.** CompSpoof ships fixed 4.00 s clips and the sampler
  filters components to `duration_s >= duration_range[0]`, so a 20 s draw has no fake music: 0/400
  pool-D and 4/600 pool-A rows reach 20 s. **`segmentation: whole_file` at 60 s is still untested on
  real audio** — which is the setting the `fps` work, the frame arithmetic and the duration-bias
  analysis all exist for.
- [ ] ⚠️ **TRAIN and VAL share one compositor.** Family-disjointness does not touch this: the model
  can key on our mixer's fingerprint, present in both halves and in none of DACON's 1200 test files.
  `shadow_of` / `shadow_b` are the repo's instrument for it and have never been built.
- [ ] ⚠️ **`build_folds` still refuses 5/4/3 folds** on real-voice and real-noise coverage, so the
  table is 2 folds and carries a 1-family-EER variance caveat on the music head. Pool D now has 8
  spoofed generators, which meets C6's floor in *count* — but they are environmental, so the count is
  satisfied and the domain is not.
- [ ] ⚠️ A 2-fold `score_sd` is a 2-sample standard deviation. Reported; not a confidence interval.
- [ ] 🔷 The repo's *"never pool raw OOF scores"* rule (0.1705 vs a true 0.100) **could not be
  reproduced here** — both folds landed on near-identical score scales, so raw concatenation cost
  0.0005. The correct path is provably used; this corpus cannot validate that particular rule.

**The entrypoint — [`scripts/train.py`](scripts/train.py)**
- [x] 🔴 **`run_schedule()` has a caller.** Folds -> schedule -> weight selection -> frozen eval ->
  gates -> tripwires -> `aggregate_folds` -> `ledger_row.json`, from one command. Every run before it
  came from a script outside the repo, which made the thing this project measures most carefully the
  one thing it could not reproduce.
- [x] 🔴 **The weight-selection step now exists, and it was missing rather than unused.**
  `EMA.state_dict_for()` and `checkpoint_soup()` were implemented and unit-tested with **no caller
  outside `tests/`**, while `LoopConfig.ema_decay` defaults to 0.999 — so every run in this pass
  maintained an EMA every step, discarded it, and scored raw weights, with
  [`architecture/05`](docs/architecture/05-multi-model.md) meanwhile calling the soup "⭐ free, do it
  by default". `--select raw|ema|soup` is that choice, recorded in the ledger row. Measured on one
  smoke fold: raw 0.7136 · ema 0.7096 · soup 0.7137.
- [x] **The caveat union is the caller's job and is now done.** `aggregate_folds`' docstring says
  caveats "do not travel on their own"; this collects them across every stage of every fold, so
  `--max-steps` reaches the `quotable` column instead of only the prose beside it.
- [x] **Exit status is the gate.** 0 only when the run is quotable; a truncated stage or a red gate
  returns 1, so a job array cannot bank an unquotable number.
- [x] 🔴 **VG1 A10 was reporting SKIP on every run.** `run_default.yaml` leaves
  `folds.scheme_version` null, so "the run's scheme_version matches folds.parquet's" compared nothing.
  The entrypoint defaults it from the manifest, which carries exactly one by construction.
- [x] **Verified end to end**: all three selections, the VG tri-state (`vg1=fail` on a deliberately
  undersized eval set, `vg2=pass`, `vg3..vg6=na` — the documented `fail > na > pass` ordering), and
  `--dry-run`.
- [ ] ⬜ No Slurm array wrapper yet. One GPU per invocation is the design (folds are independent, so
  N folds is an array rather than a distributed run); DDP would need the seeded sampler reworked.
- [ ] ⚠️ Cluster note: `/tmp` is **node-local**; `/data/project/private` is wekafs and shared. A job
  that reads a script from `/tmp` fails on any node but the one that wrote it.

---

## 🟡 2026-09-11 — EDA plan, and Phase 0 built

**Plan**: [`docs/EDA/`](docs/EDA/README.md), 9 files — one per pool, plus the shared harness, the
cross-pool analyses, and the phase order with seven pre-committed gates. It is the *execution* half
of [`data/07`](docs/data/07-eda-plan.md)'s method catalog and [`data/10`](docs/data/10-preprocessing-and-filtering.md)'s
processing catalog, neither of which is organized by pool or can be run.

**Code**: [`eda/`](eda/AGENTS.md) — Phase 0, everything answerable without decoding a sample.
41 tests, **19 of 19 mutants killed**. Driven by [`configs/eda.yaml`](configs/eda.yaml).

- [x] 🔴 **Pool D is no longer empty, and this file said otherwise until now.** `fakemusiccaps`
  (zenodo-15063698, 12.0 GiB, 27,605 clips from 5 TTM models, CC BY-NC 4.0 verified at origin) landed
  2026-09-10 with a complete provenance record. The bucket holds **13 sources / 291.0 GiB**, all with
  `_meta/DONE`, against the 9 / ~239 GiB [`data/12`](docs/data/12-acquisition-status.md) still
  reports. `fma`, `mtg-jamendo` and `wavefake` also landed unrecorded.
- [ ] 🔴 **`fakemusiccaps` is not in [`scripts/sources.yaml`](scripts/sources.yaml)**, so the
  fetcher's licence gate — the G2 gate expressed as data — never saw the one source carrying **0.27**
  of the metric. Reconcile the registry before the next fetch run.
- [x] 🔴 **The framing correction**: [`data/07`](docs/data/07-eda-plan.md) Tier X already rejects
  handcrafted acoustic features as model input (CtrSVDD: raw waveform **13.75%** EER vs MFCC
  **26.67%**). So "feature engineering" here is the seven manifest columns the pipeline actually
  consumes — and measured on the smoke corpus, `pair_id` is null on 2,177 of 2,177 rows,
  `validity_mask_ref` likewise, `aug_strength` is 0.0 everywhere, and `label_confidence` is assigned
  per *source* rather than from any per-file evidence.
- [x] **Two measurement planes** (native / post-16 kHz chain) are the plan's organizing idea: every
  band-limiting question this project has is a *difference* between them, and a single-plane EDA
  answers none. 🔷 The native plane needs no bypass — `resample_poly_to` returns its input unchanged
  when the rates match (`training/render.py:139`), so `load_audio(path, sample_rate=orig_sr)` is the
  undisturbed file through the same decode path, same ffmpeg fallback, same `DecodeError`.
- [x] **Three tables, not one.** M tier is 100% of ~1.7 M files and scalars; S tier is sampled and
  carries three `[128]`-vectors per plane. One table forces a choice between 1.7 M mostly-null vector
  columns and silently sampling the thing the shortcut audit needs the *population* of.
- [x] **Contracts, enforced rather than documented**: an extractor declares its columns and the
  declaration is checked on every call; a signal extractor's arity makes it unable to see which plane
  it is on; a decode failure is a row with `probe_ok=False`, never a dropped file; gates are a
  tri-state with **`fail > na > pass`**, so a check whose input is absent reads `na`. All four aim at
  defects this repo has already paid for — 19 ignored config fields, `dup_group` computed and
  hardcoded to `None`, VG1 A10 comparing nothing and reading as fine.
- [x] **MUSAN and RIRS are declared one source per partition** (`musan-speech`→A, `musan-music`→C,
  `musan-noise`→E). `sources.yaml` marks "never feed MUSAN's music in as noise" Critical; splitting it
  in config makes that structural instead of a rule somebody has to remember.
- [x] **Verified end to end on a fixture** with a planted format confound and a planted cross-source
  duplicate: `G-EDA2 fail` (AUC **1.000** on `voice_fake`, carried by `bit_rate`/`file_bytes`/
  `orig_sr`), `G-EDA5 fail`, exit 1. The gate can fire.
- [ ] ⚠️ **Nothing has run on the real corpus.** Every artifact in S3 is a publisher archive and
  **nothing is extracted**; `scripts/fetch_from_s3.py --extract` is the prerequisite, and egress is
  billed per read. 🔷 The ffprobe pass is ~16 minutes on 32 workers and ~8 hours on one.
- [ ] 🔴 **The prediction to test first**: MUSAN ships its music partition as `music/fma/`,
  `music/jamendo/` and `music/hd/`, and we separately acquired FMA and MTG-Jamendo. That is the
  RIRS/MUSAN redistribution pattern again, on the **0.27**-weight music head instead of on noise.
  One hash sweep to check; a corpus rebuild not to.
- [ ] ⬜ Phase 1 (the S tier: `planes.py`, level/timing/spectral/content extractors) and the manifest
  projection are not built. `gates.py` reports `na` for G-EDA4/6/7 with the reason, which is the
  correct verdict for a check whose input is absent.

**🔴 Review pass, same day — three real defects, and none of them was found by a test**

- [x] 🔴 **The exclusion matcher was substring-based, in two copies.**
  `rel.startswith(x) or f"/{x}" in f"/{rel}"` reads correctly and silently excludes `real_half/` and
  `a/really/` for `exclude: ["real"]`. It is the **licence** gate on pool D. `SourceSpec` now owns
  the matching and `eda/ids.py` implements it on path components, so `eda.driver` and `eda.gates`
  cannot disagree about what an exclusion means.
- [x] 🔴 **`na` meant "found nothing" as well as "did not run"** — the exact confusion the tri-state
  exists to prevent, reintroduced one layer up. The duplicate sweep wrote its detail table only when
  it found something, so a **clean** corpus left no artifact and a later `eda gates` reported `na`
  for a pass. Three paths could confuse the two and all three are now explicit: the summary is
  written unconditionally, a probe that skipped `identity` raises `NoHashes`, and a declared-but-
  unconsolidated pool lands in `load_files(...).attrs["missing_pools"]`.
- [x] 🔴 **`configs/eda.yaml` failed `test_every_shipped_config_loads`** — a pre-existing guard that
  sweeps `configs/` and asserts every file is accepted by **exactly one** loader. It was doing its
  job; `load_eda_config` is now registered. Resolving the resulting three-way tie surfaced a second
  defect: an EDA config with **no sources** loaded cleanly and then did nothing, so it is refused.
- [x] **Hardcoding removed**: `VALID_POOLS` now derives from `training.manifest.POOLS`; `HEADS`
  derives from the manifest's own label-column order rather than literal tuple indices (the shape of
  the File/Voice swap that once passed 749 of 749 tests); `n_splits` / `top_k` / `max_depth` moved
  into a new `AnalysisConfig`, whose seed is **deliberately separate** from `SampleConfig.seed` —
  that one picks which files are measured and is part of the corpus definition.
- [x] **Duplication removed**: `eda/ids.py` holds `file_id` construction, its decomposition and
  path-component matching, each of which had been written twice.
- [x] ⚠️ **One mutant was a false kill.** `test_the_cli_records_a_clean_sweep` died against its
  mutation *and* passed nothing clean, because it asserted the verb's exit code while the fixture's
  flat directory trips G-EDA3 at any floor. A mutant that dies against an already-red test is not
  evidence. Check the test is green before counting the kill.

---

## 🔴 2026-09-11 — SONICS and CtrSVDD registered, and one of them was in the wrong pool

Bucket is now **15 sources / ~340 GiB**. Both landed described as fake-music sources; **neither is
one**, and getting that wrong would have been expensive in different ways.

- [x] 🔴 **SONICS was acquired as `pool: D`, and that is wrong for 100% of it.** Its own
  `fake_songs.csv` reports `no_vocal = False` for **all 49,074 rows** — every song has vocals.
  `POOL_LABELS["D"]` asserts `voice_present = 0`, so as a pool-D component it would have told the
  model there is no voice in 49k AI songs that all sing, routed sung vocals into the **0.27**-weight
  music-fake head, and left `voice_fake` null on the strongest fake-vocal evidence in the corpus.
  It is **`row_kind: whole_file`, `cell: 8`** — and `validate_manifest` requires `pool = null` on a
  whole-file row, so the fix is a change of row kind, not a relabelling.
- [x] ⚠️ **CtrSVDD is fake *sung voice*, so it is pool B, not D.**
  [`data/02`](docs/data/02-label-taxonomy.md): *"Voice includes spoken and sung"*. Registered as B,
  which is what its acquisition record already said. **Pool D is therefore still FakeMusicCaps
  alone, five families, below the ≥8 floor** — SONICS does not fill it.
- [x] 🔴 **The corpus gains its first whole-file rows, and a mirrored trap with them.** The smoke
  corpus was 2,177 rows, *all* components. [`data/02`](docs/data/02-label-taxonomy.md)'s `f_c` rule
  is an equality: 49k whole-file cell-8 rows with no whole-file cell-5 counterpart makes **"is a
  whole song" predict FAKE**. The counterpart is FMA/Jamendo tracks *with vocals*, which is what
  [`EDA/03`](docs/EDA/03-pool-c-real-instrumental.md) C2's VAD sweep finds — so **C2 is now
  load-bearing rather than hygiene**. MUSDB18-HQ would close it properly and is cleared, unacquired.
- [x] **Schema**: `SourceSpec` gains `row_kind` / `cell`, `pool` becomes optional, and the two
  branches mirror `validate_manifest` exactly — including its refusal of cells **6 and 7** as
  whole-file rows, since they hold one real and one fake component. `head_labels` now reads
  `POOL_LABELS` for components and `CELL_TABLE` for whole files. The on-disk directory is a
  **partition** (`A`–`E`, `mixed`, `cell8`) because a whole-file source has no pool to partition by.
- [x] **Verified**: 46 EDA tests, **25 of 25 mutants killed**, and an end-to-end run over a
  five-source fixture spanning both row kinds. `voice_fake` now scores **17** rows where pool-D
  assignment gave it 12 — SONICS feeds both fake heads instead of one.
- [ ] ⚠️ **Numbers that will shape ingest**: 1,971 h against a ~45 h budget (**44×**, so DOSS-cap on
  `source × algorithm`); 2 artifact families, not 5 (`chirp-v2/v3/v3.5` are all Suno); duration
  32.9–240 s, median **131 s** — the first source reaching the competition's upper range and mostly
  **above** its 60 s ceiling; and ~**37 kbps** VBR mp3 against FMA's 128–320, which is a sharper
  confound than the acquisition caveat's "mp3 vs WAV" and which `bit_rate` hands to X1 directly.
- [ ] ⚠️ **CtrSVDD is already 16 kHz**, so it carries no resampler transition band while MLAAD and
  WaveFake do. Within pool B, **"sung" and "16 kHz native" are perfectly confounded** — new task
  [`EDA/02`](docs/EDA/02-pool-b-fake-voice.md) B7.
- [ ] 🔴 **`sonics` and `fakemusiccaps` are still absent from
  [`scripts/sources.yaml`](scripts/sources.yaml)** — two sources outside the G2 licence gate, one of
  them the whole of pool D. CtrSVDD *is* registered there with the ND reproduce-from-originals duty
  as `caveat_compliance`, but its own `acquisition.json` has an empty `verified_note` and `caveat`,
  and its `test_set.zip` comes from Zenodo record `10742049` while the licence was read on
  `10467648`.

---

## 🟡 2026-09-11 — the real-run plan, and four defects found checking it

[`docs/EDA/08`](docs/EDA/08-real-run.md). Every claim verified against the bucket and the code
rather than assumed; the checking is what found these. **None of the four would have raised.**

- [x] 🔴 **`extract_archives` was not recursive.** SONICS nests its ten zips under
  `payload/fake_songs/`, so `iterdir()` yielded the directory, `is_file()` was False, and
  **30.4 GiB would have extracted to nothing while the run reported success.** Now `rglob`.
- [x] 🔴 **A file that looks like an archive and is not unpacked was skipped silently** — CompSpoof's
  `development.tar.gz.part_aa..ae` is the live case. Now warned by name; docs and CSVs deliberately
  not, because a noisy warning trains the reader to ignore it.
- [x] 🔴 **`configs/eda.yaml` named the allowlist column `track_id`; both shipped CSVs call it
  `id`.** A mis-named column rejects every row, so **G-EDA1 would have FAILed all of pool C** for a
  reason that is not a licence. A mis-named column now reports `na`.
- [x] 🔴 **MTG-Jamendo's allowlist ids are *paths*** (`14/214.mp3`) while FMA's are integers (`2`).
  Stem-matching rejected **every** Jamendo track. Third key `relpath` added.
- [ ] ⚠️ **`fetch_from_s3.py --pool` filters on the bucket's acquisition record**, which still says
  SONICS is `pool: D`. `--pool D` pulls SONICS as well as FakeMusicCaps. **Select by name** until the
  acquisition records are reconciled.
- [x] **Archive layouts verified by reading headers out of S3**, since a wrong internal prefix makes
  `enumerate_source` raise: `musan/music/fma/music-fma-0001.wav` and
  `RIRS_NOISES/pointsource_noises/noise-free-sound-0423.wav`. 🔴 Both confirm E1's redistribution
  prediction **in the filenames themselves**.
- [x] **Wave 1 is 7 sources / ~56 GiB — 16% of the store — chosen for label coverage, not size.**
  X1 is a per-head population statement and a one-class head reports `na`, so no gate can fire until
  both sides of all four heads are on disk.
- [x] Verified: **105 tests** across the three touched files, **28 of 28 mutants killed**.
- [ ] ⚠️ Extract to `/data/project/private` (weka, **93 T** free), not `/` (456 G). `--dest` and
  `--extract` both retain their output, so budget 2x per source.

---

## 🔴 2026-09-12 — wave 1 probed and audited: the corpus is separable from metadata alone

Full state and the next session's instructions: [`docs/HANDOFF_EDA_REAL_RUN.md`](docs/HANDOFF_EDA_REAL_RUN.md).

**Measured**: 7 sources fetched (50 G raw / 55 G extracted, all sha256-verified), `eda probe`
10 probed / 7 absent / 1 blocked over **90,707 files**, 5 partitions consolidated, `eda analyze`
**aggregate: fail**.

- [x] 🔴 **`music_fake` is predicted at AUC 1.000 from `duration_s` alone.** FakeMusicCaps is
  10.0-10.2 s for all 27,605 files (MusicCaps is a 10-second corpus); `fma_small` is 30.0 s at both
  the 5th and 95th percentile. On the head carrying **0.27**. This was hypothesis D2, marked 🔷 in
  the plan; it is now measured. Removing every codec column changes nothing.
- [x] 🔴 **The audit is complete, not merely alarming.** Ablation: remove duration, format and
  channel count and all four heads sit at **exactly 0.500**. There is no fourth confound. The
  residual after duration is **channel count** -- `fma` is the corpus's only stereo source.
- [x] 🔴 **Source-grouped AUC collapses** (voice_present 0.500, music_present **0.222**), which is
  the diagnosis: the shortcut is **corpus identity**, not a property of the pools. A grouped AUC of
  0.222 is a classifier that anti-transfers.
- [ ] 🔴 **X1 cannot verify its own fix, and must not be asked to.** It reads the corpus as it sits
  on disk; the three fixes are **render-time transforms**. `G-EDA2` will keep reporting 1.000 no
  matter what is fixed. Confirming closure belongs to `audit_specs` over the spec stream (X5).
- [x] 🔴 **`rirs-pointsource` is 843 of 843 byte-identical MUSAN copies** -- 100%, not the "88 of
  90" [`data/12`](docs/data/12-acquisition-status.md) records from a 90-file sample. With it dropped
  and `rirs-isotropic` (8-channel, 1.4 s median = impulse responses) re-routed, **pool E has one
  independent source**, which `build_folds` cannot rotate on. CompSpoof `env_sources` is promoted
  from wave 2 to blocking.
- [x] ✅ **C1 answered**: **5,093 of 8,000** FMA tracks are allowlisted (63.7%) ~ 42 h against the
  ~45 h pool-C budget. No headroom.
- [x] ℹ️ **5 MLAAD files under `fake/lb/VITS2-Claude/` are byte-identical to each other** from five
  different utterances -- a TTS generation failure the hash sweep found for free.
- [x] **Six tool defects fixed while running**, three from checking the plan and three from the run
  itself; all in [`docs/EDA/08`](docs/EDA/08-real-run.md). The run-found three:
  `eda probe` died on the first unfetched source, `eda consolidate` returned 1 on an unprobed
  partition, and **a gate passed on an empty set**.
- [x] `eda/` is **56 tests / 37 of 37 mutants killed**; the harness is now
  [`scripts/mutate_eda.py`](scripts/mutate_eda.py) rather than a scratch file.

---

## Next — do in this order

**Now (blocking, days 1–2)**
- [ ] Post `[DACON 답변 요청]` — **only 2 left**: commercial-API terms, and **AI-Hub/NIA**. Q1–Q7 of ours were asked by another participant in [#417333](https://dacon.io/competitions/official/236749/talkboard/417333) on 2026-09-05 (⏳ unanswered) — watch, don't duplicate
- [ ] Email AI-Hub (safezone1@aihub.kr) re: NIA / competition use
- [ ] 🔴 **Record `source_name` at track/artist/speaker granularity from the first download.** A 5-fold needs ≥6 real corpora per role otherwise, and it **cannot be retrofitted** for generated audio — this is the one Phase B decision with no second chance
- [ ] **G1** dummy-file forensics → `signal_chain.yaml` (🔴 `normalize` in the render path is unparameterized until this lands, so A-S1 — the highest-leverage augmentation step — is structurally present but doing nothing)
- [ ] **G2** license audit of `docs/data/11-source-inventory.md` (top 10 first)
- [ ] `submit.zip` skeleton + trivial model → validate I/O, runtime, offline packaging (`metrics/submission.py` is ready to vendor)
- [ ] LB probe **P0-a**: all-constant 0.5 submission → **must score exactly 0.5000**. ⚠️ P0-a is deliberately degenerate, so it must be written with `validate=False` — the VG5 resolution gate rejects a constant column and would otherwise block the first submission
- [ ] Then **P0-b**: PANNs presence heads + constant fake columns. 🔴 **Expect ~0.535, not 0.5000** (`Score = 0.45 + 0.1·CPS`) — do not let this trip the P0-a stop rule ([`architecture/03`](docs/architecture/03-candidates.md#p0--the-baselines-that-are-not-models))

- [ ] 🔴 `pip download mamba-ssm causal-conv1d` against torch 2.7.1+cu128 / py3.11 / CUDA 12.8 — **one command**, and it decides whether the best published speech backbones (Fake-Mamba, 5.85% ITW EER) exist for us at all (V5)

**First training experiment — ready, blocked only on the corpus**
- [x] Pipeline is ready: `configs/run_default.yaml` (conditional, `f8=0`) and `configs/run_t1_strict.yaml` (strict, `f8=1`) ship, so the composedness policy is a config line rather than a code change
- [ ] 🔴 `T1` **`clip_weight` 1.0 vs 0.5** at Medium speed (⚠️ never Replay — it systematically favours ideas that help early, and a loss-structure change is exactly that shape). One config value, the correct control, and we currently ship a value chosen by symmetry against evidence that it may cost up to 3.63 EER points. Read its **sign** before its size ([`training/04 §6`](docs/training/04-schedule.md))
- [ ] ⚠️ Two pre-existing thin single-seed margins in `tests/test_aggregate.py` / `test_breakdown.py` would gate T1

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
- [ ] 🔴 **Three parallel branches on the mixture** (PC-Mix layout) — voice / music / file — with **joint training across branches**. ⚠️ "Largest single gain" is a *thresholded-metric* claim (ACC/F1); the EER deltas from the same ablation are 0.47–0.86 pts, below our resolution threshold ([training/04 §2](docs/training/04-schedule.md#2--s2s-evidence-is-a-thresholded-metric-result--keep-the-stage-drop-the-claim))
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
