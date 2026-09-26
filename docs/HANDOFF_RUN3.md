# Handoff — run 3: what is running, what is decided, what comes next

*Written 2026-09-26 ~18:00 KST by the lead session (its context is nearly full). Every state claim
was measured at writing time; the command or file that shows it is named. **UNVERIFIED** marks
what was not.*

**Read first:** `docs/HANDOFF_TRAINING.md` (environment, Slurm rules, the first 16 mistakes).
This document continues from it. **Plan of record:** `docs/training/13-run3-plan.md` (rev 2 plus
owner decisions).

**Clock:** the leaderboard closes **2026-09-29 (Tue) 10:00 KST**. 3 submissions per day;
Private = Public.

## 1 · Environment and place

| what | value |
|---|---|
| repo / branch | `/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting`, **`feat/training`**, 55+ commits over `main`, **not pushed** (no push or PR without the owner's go) |
| python | `/data/project/private/dacon-venvs/dacon311/bin/python` (no bare `python`); server mirror `/data/project/private/dacon-venvs/server-mirror` |
| data | manifests `/data/project/private/dacon-corpus/manifests/strategy-v4` (current), `strategy-v3` (run 1; never rewrite) |
| runs | `/data/project/private/dacon-runs/<run>/T<i>/…` (one root per task) |
| submissions | local `/data/project/private/dacon-submissions/<name>{,.zip}`; S3 `s3://hyeonseop-s3/dacon-deepfake-detection/submissions/` |
| weights | `/data/project/private/dacon-weights/{beats,xlsr-300m,xlsr-1b,xlsr-2b,w2v-bert-2.0}` (each has README.txt with revision, sha256 and licence) |

Slurm:
- **Every job `-J eval-hyeonseop`**, partition `debug`, per-job `CUDA_CACHE_PATH=/tmp/…`.
- **Owner reservation:** 2 GPUs for the owner's DDP / XLS-R-1B work once run 3a ends (~19:20).

```bash
cd /home/hyeonseop.shin/workspace/dacon-deepvoice-detecting && git status --short --branch && git log --oneline -1
squeue -u $USER -o "%.14i %.16j %.3t %.8M"
```

## 2 · Scores so far (the leaderboard is the judge)

| submission | LB | ADS | CPS | notes |
|---|---|---|---|---|
| `first-v3-seed0` (run 1) | 0.79188 | 0.77288 | 0.96287 | |
| **`run2-T7`** (run 2, best) | **0.81144** | 0.79302 | 0.97730 | server runtime 3 m 09 s |
| leader | 0.89871 | 0.88756 | 0.99907 | 97.5 % of our gap is ADS |

- VAL over-reads: v3 VAL 0.956, v4 VAL 0.944, PROBE 0.910 vs LB 0.811.
- PROBE over-read the run-1 → run-2 LB gain by about 3×.
- Details: `docs/training/12-run2-results.md`.

## 3 · Running now (measured 17:55 KST)

| job | what | tasks | ETA (end + VAL) | then |
|---|---|---|---|---|
| **220913** `run3a-ks` | R0: does up-weighting round-1 ko-synth recover held-out VITS-style TTS (melo in fold 2, mms in fold 3)? From run-2 T2/T3, 10k steps | T0 f2 ×2, T1 f3 ×2, T2 f2 ×3, T3 f3 ×3 (`configs/tasks_run3a.txt`) | ≈ 19:20 | score with post_table (§6) → compare melo/mms miss and EER vs run-2 T2/T3 (12 §2: melo .19, mms .17) |
| **220970** `run3-r1` | R1 ablation, proxy 300M@12 from run-2 fold models, 10k steps: (a) baseline, (b) + one-class loss (oc_weight 0.02), (c) LJ-voice fakes ×0 | T0/T1 = (a) f1/f2, T2/T3 = (b) f1/f2, T4/T5 = (c) f1/f2 (`configs/tasks_run3_r1.txt`); T4/T5 pending until run 3a frees GPUs | ≈ 22:30 (1.6 s/step with 8 tasks on the node) | score with post_table; apply the §5 rule of doc 13 |

Health at 17:54:
- all running;
- one-class term falling (voice 9.3 → 5.7, file 7.3 → 3.9 over 400 steps), classification
  losses normal;
- no errors.

**Monitoring:** this session ran a 15-minute health cron and a helper script
(`/tmp/claude-789818419/-home-hyeonseop-shin-workspace/38262ccb-4de3-4919-9715-1a85d33407ea/scratchpad/health.sh`: squeue, sacct non-ok states, name check, nvidia-smi, synth hours, errors in the last 15 min of job logs, latest `train_log.jsonl` lines). **Both die with this session**: re-create them (§8).

## 4 · Owner decisions (binding; doc 13 §9 and later chat)

- **Primary goal: the score.**
- **Model scale:** **XLS-R-1B, 24 of 48 layers + learnable layer fusion** (M1, M2). The **owner
  implements DDP and the 1B model**; the lead does not touch DDP.
- **One-class loss (O1):** tested in R1; kept only if it passes the doc 13 §5 rule.
- **MoLE (O4):** deferred.
- **WaveFake:** keep it (×0.25 in the draw), and **ablate** it (R1 (c): all LJ-voice fakes ×0).
- **Per-head LB diagnostics:** re-packaged on T7 (`run2-T7-diagA/B.zip`), to be **submitted by
  the owner tonight**. **UNVERIFIED:** whether they were submitted, and their ADS/CPS. Solve
  per-head with:
  - file EER = 1 − 2 (ADS_A − 0.25); music-presence AUC = 2 CPS_A − 0.5;
  - voice EER = 1 − 5 (ADS_B − 0.40); voice-presence AUC = 2 CPS_B − 0.5;
  - music EER = 1 − (0.79302 − 0.5 (1 − file EER) − 0.2 (1 − voice EER)) / 0.3.
- **No external eval set** (In-the-Wild): skipped.
- **Earlier and still in force:**
  - English + Korean only;
  - Korean/English fakes are our own synthesis, from dataset transcripts;
  - Emilia incl. YODAS;
  - hyperthread use only inside an exclusive node job (now unused: no gain);
  - never kill jobs you did not start; `-J eval-hyeonseop`.

## 5 · What to do next, in order (doc 13 §7 has the full table)

| # | owner | step | due |
|---|---|---|---|
| 1 | lead | Score run 3a (§6) → decide D2: does the ko-synth weight fix VITS-style TTS? | ~19:30 |
| 2 | owner | Submit `run2-T7-diagA/B.zip`; report ADS/CPS → per-head split decides whether music (D4) needs work (the 1B encoder cannot help music: music/presence heads read BEATs only) | tonight |
| 3 | lead | **D1 data:** new modern zero-shot cloners (e.g. F5-TTS, Spark-TTS, IndexTTS, GPT-SoVITS-style VC; licence check each) + more hours of the weakest held-out families (maskgct, seedvc), ko + en, prompts from Emilia, dataset transcripts. **D3:** more in-the-wild Korean real (Emilia-ko / YODAS-ko). Use GPUs only when free (R1 has 6, owner 2) | overnight |
| 4 | lead | Score R1 → the §5 decision for O1 and WaveFake | ~23:00 |
| 5 | lead | **strategy-v5** = v4 + D1 + D3 (+ the D2 fix). Build like `scripts/build_strategy_v4.sh`: pinned folds, then `draw_balance.py` tuning, then audits | Sun ~07:30 |
| 6 | lead | **R2** (Sun ~07:30–12:00, 6 GPUs): **all-data** T7 fine-tunes on v5 (± O1 if kept) **to submit Sunday** (LB test of the ablations), plus the **fallback XLS-R-300M@24 + fusion** all-data model | Sun |
| 7 | owner | **Main run: 1B@24 + fusion + v5** (+ O1 if kept), all-data, DDP. Scaled-model design questions: see §7 | Sun noon → Mon |
| 8 | lead | Package each candidate with `--file-mode max3` (O5, +≈ 0.002 file EER gain measured 8/8 folds); verify on server-mirror; real timing for 1B (the 10 min estimate is unverified); S3 | as they land |
| 9 | both | Submissions: Sun = R2 best (+ diag if not done); Mon = 1B; Tue AM = final pick | — |

## 6 · How-to (commands verified in this session)

```bash
R=/data/project/private/dacon-runs; V=/data/project/private/dacon-venvs/dacon311/bin/python
# launch a run from a task table (one output root per task; optional 6th column = model config)
TASK_TABLE=configs/tasks_run3_r1.txt MANIFEST_DIR=/data/project/private/dacon-corpus/manifests/strategy-v4 \
  sbatch --array=0-5 --export=ALL scripts/train_run2.sbatch <run name> --epochs 5
# post-run scoring on run 1's exact v3 VAL specs + breakdown (fold rows; all-data rows skipped)
sbatch --array=0-3 scripts/diag/post_table.sbatch $R/run3a-ks configs/tasks_run3a.txt
sbatch --array=0-5 scripts/diag/post_table.sbatch $R/run3-r1 configs/tasks_run3_r1.txt
#   -> $R/<run>/T<i>/v3val/fold<k>/{eval_row.json,val_predictions.parquet}, .../v3val/diag/{summary.json,slice_rates.csv}
# PROBE ranking of all-data models (+ per-file ensemble)
$V scripts/eval_probe.py --manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v3 \
  --scored <a>/scored.pt --scored <b>/scored.pt --ensemble \
  --weights audio=/data/project/private/dacon-weights/beats,speech=/data/project/private/dacon-weights/xlsr-300m --out <dir>
# package (+max3) and verify
$V scripts/package_submission.py --out /data/project/private/dacon-submissions/<name> --scored <scored.pt> \
  --weights audio=…/beats,speech=…/xlsr-300m --file-mode max3
cd /data/project/private/dacon-submissions/<name> && CUDA_VISIBLE_DEVICES="" \
  /data/project/private/dacon-venvs/server-mirror/bin/python script.py --device cpu \
  --test-dir /data/project/private/dacon-submissions/_first_test/data/test --out /tmp/x.csv
aws s3 cp /data/project/private/dacon-submissions/<name>.zip s3://hyeonseop-s3/dacon-deepfake-detection/submissions/
# draw symmetry + audits for any processing-config change
$V scripts/strategy/draw_balance.py --manifest-dir <dir> --processing <cfg> --fold 1 --n 6000
$V scripts/strategy/audit_fold.py --manifest-dir <dir> --processing <cfg> --fold 1 --n 40000
```

- `post_table.sbatch` parsing was dry-tested (`V=echo`); its GPU path is the same as
  `post_run2.sbatch`, which ran green on run 2.
- **UNVERIFIED:** `--file-mode max3` on a real checkpoint (tested on a stub only).

## 7 · Scaled-model (1B) questions still open (owner-side; lead prepared)

1. **Fusion:** a softmax-weighted sum vs attentive; per-head weights (default: per-head weighted
   sum).
2. **Head width:** 1280 + 768 in, or a 1280 → 1024 projection. **Voice and file heads cannot load
   strictly from T7** (`models/model.py:76`). Only BEATs + LoRA and the music/presence heads
   transfer.
3. **LoRA on 1B:** rank/targets; all 24 layers or the top half.
4. **Warm-up:** train the new parts (heads + fusion) first with the encoder LoRA frozen for 1–2k
   steps (default yes).
5. **DDP recipe:** global batch (e.g. 128) and LR scaling. Memory: 1B@24 takes 24.6 GiB at batch
   4 × 60 s → per-GPU 16 needs checkpointing or shorter clips. Per-rank render pools and shards;
   EMA on rank 0.
6. **Validation:** the main run is all-data. Reserve a short 1B fold-1 check during the smoke
   window (proxy → 1B transfer is otherwise unmeasured).
7. **Acceptance:** beats T7 on PROBE; real server-mirror timing ≤ 30 min; fp32 vs fp16 inference.
8. **Go/no-go:** if 1B is not stable by Sun noon, the R2 XLS-R-300M@24 model ships Monday.

## 8 · Things that bite (new this phase; continue HANDOFF_TRAINING §6 numbering)

| # | what | rule |
|---|---|---|
| 17 | `build_folds` re-rotates everything when data is added | **pin folds**: `scripts/build_folds_pinned.py` (v3 rows keep slice/fold; PROBE unchanged) — B-init and like-for-like scoring depend on it |
| 18 | Tasks that share a fold overwrite each other's `<out>/fold<k>/` | one output root per task (`train_run2.sbatch` does it) |
| 19 | Emilia/clone tiny speaker buckets make voice slots **repeat audio**; asymmetric across labels = shortcut (audit I1c failed) | tune `draw.domain_weights` with `draw_balance.py`; audits on every config change |
| 20 | Speaker buckets span corpora: MLAAD-LJ shares WaveFake's bucket, so a domain weight ×0 does not zero WaveFake | zero every family sharing the bucket (`processing_run3_nowf.yaml`) and measure the residual |
| 21 | DACON A1: post-processing without a new component (**codec round-trips**) is **REAL** | never label codec/vocoder copy-synthesis of real speech as fake; S1 trains codec round-trips as real |
| 22 | PROBE over-reads LB gains about 3× and is zh-heavy | rank with it; decide with VAL slices + the LB; always `processing_first_run.yaml` for PROBE |
| 23 | Pooled VAL looks fine while held-out families fail (v4 VAL: maskgct .41/.27, seedvc .22, melo .21, mms .22; Emilia-ko false alarm .15–.18) | judge by per-slice EER (doc 13 §5), not pooled |
| 24 | XLS-R-1B width 1280 → strict init of the voice/file heads fails | partial init |
| 25 | Monitoring cron and helper scripts live only in the session | re-create after any handoff |
| 26 | `train.py` exits 1 ("not quotable") on single-fold tasks → Slurm FAILED | read `ledger_row.json` (still true in run 2) |

## 9 · Where things are

- **Docs:** `docs/training/10` (run 1 diagnosis), `11` (run 2 design), `12` (run 2 results, sizing,
  LB), **`13` (run 3 plan)**.
- **Code this phase** (all on `feat/training`):
  - pinned folds (`processing/pinned_folds.py`, `scripts/build_folds_pinned.py`);
  - draw tools (`scripts/strategy/draw_balance.py`);
  - launcher (`scripts/train_run2.sbatch`: TASK_TABLE, per-task model column);
  - evals (`scripts/diag/{run_breakdown.py,post_run2.sbatch,post_table.sbatch,bench_encoders.py}`,
    `scripts/eval_checkpoint.py --out-dir`, `scripts/eval_probe.py --ensemble`);
  - speed (batched BEATs tokens, `c_run2.yaml`, grad checkpointing off);
  - O1 one-class + O5 max3 (`models/{heads,losses,model,config}.py`, `c_run2_oc.yaml`,
    `train_run3_oc.yaml`, `package_submission.py --file-mode`);
  - `script.py` diag constant columns.
- **Submissions** (local and S3):
  - `first-v3-seed0`, `run2-T6`, `run2-T7` (best);
  - `first-v3-diagA/B` (run 1, superseded);
  - `run2-T7-diagA/B` (to submit).

You can use oh-my-claudecode skills if needed.
