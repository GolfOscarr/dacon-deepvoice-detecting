# Run 2 design

*Draft 2026-09-26 by the lead session. Inputs: run 1 (`docs/training/07` §7), its diagnosis
(`docs/training/10`), the round-2 data (`docs/training/09`), and the owner's decisions:
the highest score; English + Korean only; hyperthreads allowed only inside one exclusive Slurm
job that holds the node. Numbers are measured unless marked **ESTIMATE** or **PENDING**.*

**Deadline.** The leaderboard closes 2026-09-29 10:00 KST. Run 2 must leave time for run 3,
the final model.

## 0 · What run 2 has to answer

1. **Does the round-2 data remove the two diagnosed failures?**
   - F1: held-out clean read speech called fake. Real LJ false-alarm rate 0.52 on fold 0;
     real musan-speech 0.49 on fold 1.
   - F2: unseen cloners missed. Korean XTTS miss rate 0.31 on fold 1.
2. **Is starting from run 1's weights (B) as good as training from scratch?** It is much
   cheaper.
3. **Which models become the next submissions,** and which recipe becomes run 3.

## 1 · Data and folds — `strategy-v4`

### 1.1 Mixture

| side | v3 (run 1), ko/en only | added in v4 | source |
|---|---|---|---|
| real voice, ko | Zeroth 52.9 h, CV-ko 2.7 h | **Emilia-ko 40 h + Emilia-YODAS-ko 40 h + FLEURS-ko 8.4 h**; 19,968 speakers | `interim/emilia-ko` |
| real voice, en | musan-speech 60.4 h, LibriTTS-R 49.6 h, LJ 23.6 h | **Emilia-en 40 h + YODAS-en 40 h**; 11,285 speaker keys | `interim/emilia-en` |
| fake voice, ko | ko-synth 39.6 h (xtts, melo, openvoice, bark, knnvc, mms) + MLAAD-ko 0.9 h | **ko-synth2**: seedvc 8.2, fishspeech 8.0, maskgct 7.5, cosyvoice ≥ 2.8, chatterbox ≥ 2.6 h (at 01:00; final at 04:50) | `interim/ko-synth2` |
| fake voice, en | WaveFake 181.5 h (LJ speaker only) + MLAAD-en 8.5 h | **en-synth2**: the same 5 cloners, prompted by Emilia-en speakers, target 6–8 h each (PENDING, until 12:00) | `interim/en-synth2` |
| both labels | — | **S1 processing** (EnCodec, DAC 16k/44k, XCodec2, Mimi, DeepFilterNet3, DSP pitch/formant/tempo): 30.8 h real + 30.8 h fake, label-preserving | `interim/proc` |
| music, noise | unchanged | — | |

**Draw** (`configs/processing_run2.yaml`):
- `lang_shares` = ko 0.55, en 0.45, zh 0, ja 0, other 0.
- The both-sides rule drops every language that lacks a real side, so MLAAD's non-en/ko fakes
  and WaveFake-ja are not drawn.
- **WaveFake / LJ** (en-data track, `9a21d0d`): `domain_weights` `wavefake|` and
  `proc-*|wavefake|` ×0.25, `ljspeech/` ×25.
  - Measured on a 20,000-spec draw: LJ went from 1.2 % of drawn English *real* and 39 % of
    drawn English *fake* to 17.6 % / 15.8 %.
  - That is F1's mechanism for the **all-data** models.
  - **It cannot act on fold 0** (independent review M5): LJ and WaveFake-en sit entirely in
    fold 0's VAL, so fold-0 training never draws them. The 0.52 LJ false-alarm rate came from
    a model that never saw LJ.
  - Fold 0 (T0) therefore tests the *new data only*. The draw fix is read on the all-data models
    and through musan-speech (folds 1–2, v3 VAL).

**Label symmetry of the draw — found by the v4 rehearsal, must be fixed before launch.**
Rehearsal: v3 + proc + emilia-ko + ko-synth2 + emilia-en; fold 1, `audit_fold.py` n = 40,000
and `scripts/strategy/draw_balance.py` n = 6,000.

| drawn voice, label-conditional | real | fake | effect |
|---|---|---|---|
| slots that **repeat source audio** (`voice_unique_fraction` < 0.99) | 0.34 | 0.09 | **I1c FAILS**: voice_fake AUC 0.648 vs gate 0.61 |
| **S1-processed** share | 0.12 | 0.33 | "codec/enhancer artifacts ⇒ fake", the opposite of S1's purpose |

- **The repetition comes from bucket size.** A voice slot tiles one speaker bucket, and a bucket
  shorter than the slot is reused.
  - Emilia speakers hold a median 8.9 s (ko) and 21.4 s (en); 87 % / 85 % of their slots repeat.
  - ko-synth2 prompt speakers hold 8.1 s (93 % repeat).
  - Every v3 corpus is ≈ 0 %.
  - With Emilia at 39 % of real voice and synth2 at 10 % of fake, repetition reads *real*.
  - The consequence on the test: its real audio never repeats, so "no repetition ⇒ fake"
    would raise false alarms.
- **The fix is draw weights, not less Emilia.**
  - Down-weighting Emilia balances the numbers, but it drops Emilia to 6–8 % of real voice per language,
    which undoes F1/F3's fix.
  - Instead: round-2 clones up (they are the same tiny-bucket speakers, and F2 wants more of
    them), real processed rows up, fake processed rows down.
  - Rehearsal: `ko-synth2|` ×3, `proc-*/` (real) ×8, `proc-*|` (fake) ×0.6 give repeats
    0.195 / 0.235 and processed 0.21 / 0.28. **The audit passes** (I1c voice_fake no longer
    fails).
- **Final weights are tuned after en-synth2 lands.** Its clones add fake-side repetition, so
  `ko-synth2|` / `en-synth2|` come down.
  - Targets: |Δ repeats| ≤ 0.04; |Δ processed| ≤ 0.05; ko/en .55/.45 on both labels; audit ok.
  - Checked on fold 0 and on all-data.

### 1.2 Folds — pinned to v3 (a correctness requirement, not a preference)

`training.folds.build_folds` is a global greedy pass. It ignores any existing assignment
(`training/folds.py:630`, the `build_folds` docstring), and new groups change its order and tallies. Rebuilding folds on v4
can therefore move a v3 group to another fold, or into or out of PROBE. That would break three
things:

- **B-init leaks.** Run 1's fold-k model trained on fold k's TRAIN. If a v3 TRAIN group moves
  into v4 fold k's VAL, run 2 fold k is scored on data its initial weights were trained on.
- **PROBE contamination.** All-data run 1 models trained on every v3 non-PROBE group.
- **No like-for-like comparison.** With v3 VAL unchanged, run 2 can be scored on exactly run 1's
  6,000 VAL specs.

**Rule: v4 keeps every v3 row's `slice` and `fold`. Only groups with no v3 member are
assigned.** This is implemented by `scripts/build_folds_pinned.py` (§5). It checks:

1. Every v4 grouping atom that contains a v3 row gets that row's (slice, fold). If one atom
   holds v3 rows from two different folds, it **aborts**: a new row has bridged two v3 atoms.
   The S1 processed rows are the ones at risk: they carry their source's speaker and family,
   so they must join the source atom.
2. New atoms never go to PROBE. The v3 PROBE stays exactly as sealed.
3. VG1 (`check_split_integrity`) passes on the result.

**Placing new groups:**

- **Cloner families: fixed, one cloner per fold in both languages.** The ko and en families of
  the same model stay together, so "unseen" means unseen in any language:

  | fold | held-out cloner (ko + en) | what that fold already holds out (v3) |
  |---|---|---|
  | 0 | fishspeech | LJ + WaveFake (the clean-read-speech test, F1) |
  | 1 | maskgct | Korean XTTS (F2) |
  | 2 | seedvc (voice conversion) | ko-synth melo, openvoice |
  | 3 | cosyvoice + chatterbox | ko-synth bark, knnvc, mms |

- **Emilia prompt speakers:** a speaker whose voice prompted a clone family goes to that
  family's fold. With several families, it goes to the first one alphabetically.
  - For most Emilia speakers, VAL then holds the real speaker **and** their clone.
  - **Not all of them** (review M1). Clones prompted by *pinned* v3 speakers (Zeroth; en-synth2
    also uses LibriTTS-R) follow their family, not the speaker.
    - That is 3,278 of 12,979 ko-synth2 rows; 75 % of them sit in a different fold from their
      speaker.
    - Multi-cloner speakers are split too (1,923 Emilia speakers).
  - So some VAL reals are voices the fold trained on as *fake*.
    `scripts/diag/run_breakdown.py` tags them (`voice_spk_cloned_in_train`) and reports them as
    their own slice; headline F1 numbers use the untagged reals.
  - In run 1 the tagged reals were **not** hurt: false-alarm rate 0.000 / 0.011 / 0.003 on
    folds 1–3, vs 0.226 / 0.034 / 0.022 untagged.
- **Other Emilia speakers:** greedy by hours, filling each fold's (lang, real) VAL hours
  toward the per-fold mean. v3 ko-real VAL hours are very uneven: 34.0 / 0.2 / 6.4 / 14.9.

**Consequences to keep in mind:**

- **PROBE has no ko/en real voice** (measured: en real 0 rows; ko real 4 CV-ko files, ≈ 0.0 h).
  - Under `processing_run2.yaml` (zh 0) PROBE's voice side is empty.
  - **PROBE ranking must use `configs/processing_first_run.yaml`**, which is how run 1 was
    ranked. It stays a zh-heavy, harsh ranking tool, not an estimate.
  - The leaderboard is the real judge.
- Fold 0 VAL keeps the LJ/WaveFake block and 32.6 h of Zeroth, and gains fishspeech (ko + en).
  - v3 fold-0 VAL drew **no Korean clips** (no Korean fake was in it). Korean in fold-0 VAL is
    new in v4.
  - Fold 0 remains the F1 test on LJ.
- **`for_eval()` keeps `domain_weights`, so the training weights also shape v4 VAL** (review
  minor 1). On the rehearsal:
  - fold-0 English real in VAL is 53 % LJ;
  - fold-1 English real is 94 % Emilia, with musan-speech at 3.7 %.

  v4 VAL moves whenever the weights are retuned. The F1/F2 slices that must stay comparable
  are read on **v3 VAL** (run 1's exact specs, §4).
- On v3 VAL, ~6 % of fake voice in folds 1–3 is MLAAD in "other" languages, which run 2 never
  trains on. That costs run 2's pooled v3 score a little and says nothing about F1/F2
  (review minor 2).

## 2 · Objective

**No change for run 2.** The loss weights stay metric-proportional (`train_first_run.yaml`):

| loss | weight |
|---|---|
| file | 0.45 |
| music | 0.27 |
| voice | 0.18 |
| v_pres | 0.05 |
| m_pres | 0.05 |

The loop logs a renormalised `w_eff` (file 0.45, music ≈ 0.41, voice ≈ 0.27).

The reasons:

- The diagnosed failures are **data** shortcuts: speaker and recording style, and unseen
  generators. A loss change would compete with the data fix for attribution.
- Run 2 initialises from run 1, and `init_from` loads strictly, so the heads must not change.
- The owner asked for the highest score within about three days; an untested objective is the
  riskiest item per hour.

**One zero-cost decision rides on run 2's VAL: how FILE_FAKE is formed.**

- `file_head.mode` ∈ {learned, noisy_or, max} is applied inside `submission_probs`, so every
  mode is computable from one trained model.
- On run 1 (4 folds), mean file EER was:

  | mode | mean file EER | note |
  |---|---|---|
  | learned | 0.064 | |
  | max(v·vp, m·mp) | **0.060** | better on folds 0, 2, 3; worse on fold 1 |
  | max3 = max(learned, v·vp, m·mp) | 0.060 | |

- **max3 is label-consistent:** the ground truth is exactly voice-fake OR music-fake over
  present components. It only ever *raises* FILE where a present-gated head is surer.
- The stacker (doc 10 F6) is rejected: it failed on a real test clip.
- Choose on run 2's four folds. Implementing max3 needs a `file_head.mode` value and a package
  override: small, and tested before use.

**Deferred to run 3 or later, if run 2 shows the shortcut survives the data fix:**
- same-speaker pair-contrastive loss;
- a speaker-adversarial head;
- per-cell loss reweighting for cells 4 and 6.

## 3 · Speed

**Where the time goes** (run 1, measured):

- A step is 1.17 s at batch 16: 13.7 samples/s per GPU.
- The GPUs are ~53 % busy (nvidia-smi sampling). The speedup track reads the per-row BEATs
  fbank loop as the cause: ~35 host↔device syncs per forward (code reading; GPU profile PENDING).
- Data wait is 4–7 s per 60 s.
- The **render pool** (7 spawn workers) delivers 14.6–15.2 samples/s (speedup track, CPU
  measurement). It is about to become the ceiling:
  - its cost is ffmpeg round-trip 49 %, the float64 FFT noise 32 %, cache reads 10 %;
  - with the sibling hyperthreads, 14–16 workers reach 20.2–21.8 samples/s (**+40 %**).

**Plan:**

1. **GPU:** batched BEATs fbank/patch/window path, flag-gated, CPU-verified to ~1e-7 against the
   row loop. Also a foreach EMA (bitwise) and loss parts kept on the device (−11 syncs/step).
   GPU numbers are **PENDING** (speedup track, ~02:30–10:00).
2. **CPU:** the HT-sibling plan was dropped after measurement (below). 7 render workers per
   task, with collate in the workers.
3. **Measured since the draft** (speedup `9be8b4b`, one H200, saved batches, no render load):

   | path | s/step | peak memory |
   |---|---|---|
   | run-1 path | 0.983 | 39 GiB |
   | batched tokens | 0.944 | 39 GiB |
   | **+ no layer checkpointing** | **0.785** | 98.5 GiB of 141 |

   - The GPU is ~97 % busy in isolation, so run 1's 53 % was CPU contention and syncs.
   - 0.785 s/step = 20.4 samples/s, which matches the render supply on the HT siblings (20–22/s).
   - **Final, in real `train.py`** (speedup, one task, fold 0, v3, steps 150–400):

     | setup | s/step | samples/s |
     |---|---|---|
     | run-1 path | 1.14 | 14.0 |
     | batched tokens + ckpt on + worker collate | 1.05 | 15.2 |
     | **batched tokens, ckpt OFF** | **0.86–1.0** | 16–18.6 |

     Peak with ckpt off: 98.5 GiB; 102.7 GiB on a synthetic 16 × 60 s worst case.
   - **Measured dead ends:**
     - batch 32 (15.5 samples/s);
     - `torch.compile` per layer;
     - **render workers on the HT siblings** (0.92–0.97 s/step, no better than 7 workers:
       they slow the main process).
   - **Run-2 config** (`83e08ea`): `configs/c_run2.yaml` (`batched_tokens: true`),
     `grad_checkpointing: false`, `render_workers: 7`, batch 16. The HT flag stays off.
   - So **no exclusive node job**: `scripts/train_run2.sbatch` is an 8-task array like run 1's.
   - **UNVERIFIED:** CPU contention with 8 tasks at once (measured with one). Check s/step and
     `data_wait_s` in the first 10 min.
4. **No DDP for run 2.**
   - DDP does not raise the node's total throughput: every GPU still needs its own ~14–22
     samples/s of rendering, and that is the binding limit.
   - It only concentrates the throughput into one model.
   - Run 2's value is **8 different models at once** (§4).
   - DDP is reconsidered for run 3, only if the final recipe needs more samples on one model
     than one GPU delivers by the deadline. It costs a loop change (sampler sharding,
     rank-0 checkpoint/EMA, per-rank render pools) plus a large-batch LR change, and it is
     untested. **ESTIMATE:** half a day with tests.

**Step budget.** Run 1's learning curves (doc 10 F7) set the scale:
- fold 1 saturates by ~36,000 steps from scratch;
- fold 0 is flat after 6,000.

So ~20,000 fine-tune steps on the new data is enough to show the data effect.

**Timing.**
- Run 2 launches ≈ 13:00 and should finish by ≈ 20:00: 7 h.
- At 1.0–1.2 s/step that is ~20,000 steps (**ESTIMATE**, fixed once the speedup numbers
  land). That is 10 passes × 32,000 draws at batch 16, a checkpoint per pass for learning curves.
- Fine-tune schedule for B (`configs/train_run2.yaml`):
  - peak LR 1.0e-4 (half of run 1);
  - warmup 1,000 steps: the loop section is shared by every task, so one value;
  - cosine to 0.05;
  - frontend LR scale 0.2 as before.

  Run 1 ended at LR factor 0.05, so B re-warms from a low point. Half the peak limits
  forgetting. T4 uses `configs/train_run2_scratch.yaml` (LR 2e-4, run 1's recipe).
- **Fallback time** (review minor 11): if en-synth2 is not ingestible by 12:30, build v4
  without it (`ONLY=proc,emilia-ko,ko-synth2,emilia-en`).
  - Adding en-synth2 later means re-pinning *against v4*.
  - Its clones would then not share folds with their Emilia-en prompt speakers (M1 again). The
    breakdown's tag keeps that measurable.

## 4 · The eight tasks (ablations)

One exclusive job; task *i* on GPU *i*. The strategy-v4 draw is used unless stated.

| task | kind | init | data | answers |
|---|---|---|---|---|
| T0 | fold 0 | run 1 fold 0 | v4 | F1: real-LJ false alarms, fold-0 T3 gap; fishspeech unseen |
| T1 | fold 1 | run 1 fold 1 | v4 | F2: XTTS miss, musan false alarms; maskgct unseen |
| T2 | fold 2 | run 1 fold 2 | v4 | seedvc (VC) unseen; completes the 4-fold mean |
| T3 | fold 3 | run 1 fold 3 | v4 | cosyvoice + chatterbox unseen; 4-fold mean |
| T4 | fold 1 | **scratch** | v4, LR 2e-4 / warmup 1000 (run 1 recipe) | is B-init as good as scratch at equal **wall-clock**? B has 54k steps of run 1 behind it, so equal steps favours B. The run-3 question is the wall-clock budget (fold 1: the fold that kept improving with steps) |
| T5 | fold 0 | run 1 fold 0 | **v3 corpus**, run 2's draw (`configs/processing_run2_v3ctl.yaml`: same shares and weights, scheme stamp v3) | control: new data vs (ko/en shares + more steps). On fold-0 training the domain weights are inert (M5), so T5 does not test them |
| T6 | all-data | run 1 all_data_seed0 | v4 | **submission candidate** |
| T7 | all-data | run 1 all_data_seed2 | v4, `--seed 2` (→ `all_data_seed2/`) | second candidate (a different init and draw order from T6, so not a clean seed-spread measure) |

**Why this set:**
- Folds 0 and 1 are where run 1 failed.
- T4 and T5 are the two controls a run-3 decision needs (init choice, data attribution).
- Folds 2 and 3 now hold out new cloners, so they are informative again, and all four folds
  are needed to pick the FILE mode (§2).
- T6/T7 start from run 1's two best all-data models on PROBE (0.853, 0.846).

**Launcher** (`scripts/train_run2.sbatch`, speedup track, review B1):
- **every task writes to its own root** `RUNS_ROOT/<run>/T<i>/`. `train.py` writes
  `<out>/fold{k}/`, so T0/T5 and T1/T4 would otherwise overwrite each other.
- Common flags (as run 1):

  | flag | value |
  |---|---|
  | `--stages` | `joint` |
  | `--select` | `ema` |
  | `--draws` | `32000` (10 passes → 20k steps) |
  | `--eval-n` | `6000` |
  | `--weights` | `audio=…/beats,speech=…/xlsr-300m` |
  | env | per-task `CUDA_CACHE_PATH` |

- Model config `configs/c_run2.yaml` = `c_first_run.yaml` + `batched_tokens: true`. That is a
  flag-only change; run-1 `scored.pt` must still load strictly.

**Scoring every task:**
- **v4 VAL:** `train.py`'s end-of-run eval, 6,000 specs.
- **v3 VAL — exactly run 1's specs:**
  - Run `eval_checkpoint.py --ckpt <T>/fold<k>/scored.pt --fold k --manifest-dir …/strategy-v3
    --processing configs/processing_first_run.yaml --eval-n 6000 --out-dir <T>/v3val`.
  - It writes `val_specs.json` + `val_predictions.parquet` for `run_breakdown.py` (added in
    `6d9e9ab`, review M2).
  - Verified: its specs equal run 1's `val_specs.json` (6,000/6,000, reviewer rebuild on fold 1),
    and run 1's `scored.pt` reproduces run 1's predictions within 6e-5.
  - Valid because v4 pins v3's folds. It is the like-for-like comparison with run 1.
- `scripts/diag/run_breakdown.py` on both, for the per-corpus and per-family rates at the pooled
  threshold. **The headline numbers are F1/F2's slices, not the pooled score** (mistake 16).
- PROBE, all-data tasks only, under `processing_first_run.yaml`.

**Decision rules for run 3:**
- T0 vs T5 → does the data fix work?
- T4 vs T1 → init or scratch.
- The four folds → FILE mode.
- T6/T7 on PROBE + the LB → which submission.

## 5 · Build order and checks (before launch)

1. **04:50** — Korean synthesis ends. Check each `ko-synth2/<family>/metadata.csv` and its QC
   rejects.
2. **by 12:00** — en-synth2 ends. The en-data track reports hours, QC, WaveFake cap.
3. `scripts/extend_manifest.py --dry-run`, then the manifest into a **new** dir `strategy-v4`
   **with `--scheme strategy-v4`**. The default stamp is v3, and a wrong stamp only fails A10
   after 7 h of training. `build_strategy_v4.sh` passes it; check
   `manifest.scheme_version.unique()` before launch.
4. `scripts/build_folds_pinned.py --base-dir …/strategy-v3 --manifest-dir …/strategy-v4`
   (commit `6e7860f`). **Rehearsed** on v3 + proc + emilia-ko + ko-synth2 + emilia-en (506,926 rows):
   - VG1 ok;
   - v3 slice and fold equal on all 400,944 rows;
   - PROBE 119,904 rows = v3;
   - all 27,497 proc rows share their source's fold;
   - held-out real hours even: ko 41 / 34 / 34 / 34, en 53.4 each; ko fake 8.0 / 14.2 / 22.4 / 22.6.

   It must pass:
   - atoms consistent;
   - PROBE unchanged (row-for-row equal on v3 rows);
   - every v3 row's (slice, fold) equal;
   - VG1 ok.
5. `build_cache.py` for the new rows. (A rehearsal on the corpora already finished pre-warms
   the shared cache; the cache is idempotent.)
6. Tune `draw.domain_weights` with `scripts/strategy/draw_balance.py` (§1.1 targets), then
   `audit_fold.py` fold 0, fold 1 and all-data at n = 40,000. Read:
   - drawn ko/en shares ≈ .55/.45 on both labels;
   - drawn processed share equal on both labels;
   - the WaveFake share of English fakes.
7. The speedup track's flags verified: run-1 `scored.pt` loads strictly and gives unchanged
   outputs.
8. Launch `scripts/train_run2.sbatch` (one exclusive job, T0–T7). Check within 10 min:
   - s/step;
   - `data_wait_s`;
   - GPU busy;
   - all 8 tasks alive.

## 6 · Risks

| risk | guard |
|---|---|
| en-synth2 late or thin | launch without it: `ONLY` drops `en-synth2`; F1 still gets Emilia-en reals + WaveFake cap |
| a new row bridges two v3 atoms | the pinned builder aborts; find the row and fix its keys, don't unpin |
| hyperthread affinity starves another job | only inside the exclusive node job; nothing else can be scheduled there |
| B-init carries the shortcut | T4 (scratch) and T5 (control) show it; run 3 can switch |
| a fold task exits 1 ("not quotable") | not a crash; read `ledger_row.json` (mistake 15) |
| PROBE read under the wrong config | always `processing_first_run.yaml` (§1.2) |

## 7 · Review log

An independent review of `f03cdd2` (critic agent, read-only, CPU re-checks) returned
**REVISE**.

It verified as correct:
- the pinned-fold logic and rehearsal;
- the S1 inheritance;
- the reproduction of run 1's VAL specs (6,000/6,000);
- that the domain prefixes hit their rows;
- the config diffs;
- the PROBE claim;
- 20+ numbers.

| finding | disposition |
|---|---|
| **B1** tasks sharing a fold overwrite `<out>/fold{k}`; launcher unspecified | fixed in spec (§4 Launcher): per-task roots; the speedup track builds `train_run2.sbatch` |
| **M1** prompt speaker ≠ clone fold for pinned speakers | claim corrected (§1.2); tagged in `run_breakdown.py`; run 1 shows no bias on those reals |
| **M2** no like-for-like tool | `eval_checkpoint.py --out-dir`, accepts `scored.pt` (`6d9e9ab`), verified |
| **M3** speed flags not in the committed configs | assigned to the speedup track (loop section of both run-2 configs + a "differ only in scheme" test) |
| **M4** render workers share the trainer's threads | reserve a core per trainer (speedup track); measured at launch |
| **M5** fold 0 cannot test the draw fix | stated (§1.1, T5 row); read through the all-data models and musan |
| minor 1, 2, 4, 5, 8, 10, 11, 3 | stated in §1.2, §4, §3, §5 |
| minor 6 | the tautological PROBE check is replaced; a new family merging into a base atom now aborts (+ test) |
| minor 7 (report omits inherited proc hours) | accepted, cosmetic |
| minor 9 (DDP premise vs the new benchmark) | §3 updated with the measurements; the conclusion stands |
