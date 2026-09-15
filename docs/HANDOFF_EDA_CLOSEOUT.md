# Handoff — closing out the EDA

Written 2026-09-16. Everything below was **measured in this session**, not recalled.

---

## 1 — Environment

| | |
|---|---|
| Repository | `/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting` (no worktrees) |
| Branch | **`feat/eda`**, clean, 39 commits ahead of `main` |
| HEAD | `5796e54` — *docs: 10 — the final EDA plan* |
| Interpreter | `/data/project/private/dacon-venvs/dacon311/bin/python` — Python 3.11.15 |
| Credentials | **none to handle.** AWS is the EC2 instance role `EC2-Slurm-GPU-Node-Role`; there is no `~/.aws` and no `AWS_*` env var |
| Corpus | `/data/project/private/dacon-corpus/{raw,interim}` |
| EDA output | `eda/out/{A,B,C,D,E,cell8}/` and `eda/out/_shared/` |

Verify you are in the right place:

```bash
cd /home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
V=/data/project/private/dacon-venvs/dacon311/bin/python
git status --short --branch                 # expect: ## feat/eda, clean
$V -c "import eda; print(eda.__file__)"     # expect: .../dacon-deepvoice-detecting/eda/__init__.py
```

**Gates, measured** (all four commands below were run this session):

```bash
$V -m pytest tests/test_eda.py tests/test_eda_signal.py tests/test_eda_report.py \
             tests/test_eda_content.py tests/test_config.py tests/test_fetch_from_s3.py \
             tests/test_eda_groupkeys.py -o addopts="" -q     # 245 passed
$V scripts/mutate_eda.py $V                                   # 100/100 mutants killed
python3 scripts/check_links.py                                # all links OK (106 files)
$V -m eda.cli gates                                           # aggregate: fail (expected -- see §3)
```

⚠️ **Two standing prohibitions**, both learned the hard way:
* **Do not run the full pytest suite while editing `configs/eda.yaml`** — several tests assert on the
  shipped config and editing it mid-run produces failures that are artifacts of the race.
* **`scripts/mutate_eda.py` edits files in place.** Never run it concurrently with anything else that
  reads the tree, and make the suite green first — a mutant that dies against an already-red test is
  not a kill.

---

## 2 — Goal of the next session

**Close the EDA**, by executing steps 1–3 of [`docs/EDA/10-final-plan.md`](EDA/10-final-plan.md) —
the plan of record, and the *complete* inventory of all 43 declared tasks.

| step | what | cost |
|---|---|---|
| **1** | **B1** — the WaveFake↔LJSpeech paired vocoder experiment | ~15 min decode |
| **2** | four screens over existing columns: B4/D6, B6, E5, X2 | no decode |
| **3** | `eda/out/_shared/roles.md` — what X6 actually asked for | no decode |

**Done when** all three have written results, and
[`data_memo.md`](EDA/data_memo.md) + [`RESULTS_FOR_ANALYSIS.md`](EDA/RESULTS_FOR_ANALYSIS.md) are
updated to match. Both currently say the EDA is complete, which
[10 §1](EDA/10-final-plan.md) shows it is not.

**Start with step 1.** It is the highest-value item left and the only one that decodes.

---

## 3 — Progress and diagnosis

### Measured state

| tier | population | state |
|---|--:|---|
| M — metadata | 382,068 files (100%) | complete, `probe_ok` 382,065 |
| S — signal | 58,885 rows | complete, 0 decode failures, both planes |
| V — vectors | 58,885 rows | complete, `[128]`×3×2 per row |
| C — content | 58,884 rows | complete (Silero VAD, chain plane) |

**Gate verdicts — 6 pass · 4 fail · 3 `na`.** The aggregate `fail` is expected and is **not** a
defect to chase:

* `G-EDA2` fail — **structural, cannot go green here.** X1 reads the corpus on disk; every fix is a
  render-time transform. It will report 1.000 on `music_fake` whatever is fixed. Closure is X5's,
  over the spec stream, in Phase 2.
* `G-EDA3` fail — 5 of 14 sources under 6 groups. Two have no publisher key; three (`ljspeech` 1,
  `sonics` 5, `wavefake` 2) are **genuinely short**. That is the count, not a gap.
* `G-EDA6` fail — 2,442 of 58,884 rows contradict their asserted components. A work list; 07 says
  reassign (F-A1) before dropping.
* `G-EDA1/allowlist/fma` fail — 2,907 of 8,000 outside the licence allowlist. A **decision**, not a
  computation.

### The three findings the EDA exists to hand forward

1. **Duration is the shortcut and only the sampler can fix it.** `music_present` holds **AUC 0.852**
   after a whole-publisher holdout, against a 0.60 gate. Pool D is 10.000 s for every file, pool C is
   30.003 s, and resampling does not change a file's length.
2. **Above 8 kHz is unavailable.** Ten of fourteen sources have nothing there to begin with.
3. **Level and metadata are publisher fingerprints.** Metadata is already neutralised by the render
   chain; level is not — 22.6 dB of median-RMS spread.

### Diagnosis for step 1 (B1)

B1 is unblocked and its inputs are confirmed on disk: **7** `ljspeech_*` vocoder directories under
`/data/project/private/dacon-corpus/interim/wavefake/zenodo-5642694/generated_audio/`, and
`/data/project/private/dacon-corpus/interim/ljspeech/mdc-1.1/LJSpeech-1.1/wavs`.

⚠️ **The existing S-tier draw cannot answer it.** Only **128 of 13,100** LJSpeech utterances are
incidentally pairable (measured: the intersection of sampled `ljspeech` and `wavefake` utterance ids),
and they do not cover the 7 vocoders evenly. B1 needs **its own targeted selection** — pick ids
present in all 7 directories, then decode real + 7 fakes for each. ~500 ids is ~4,000 files, ~7 audio
hours, ~15 min at the measured 1.33 audio-hours/min.

Do **not** redraw `sample.json` to do this. The recorded draw is part of the corpus definition and
`Sample.fingerprint` refuses to let a redraw happen silently; B1 is a separate, purpose-built
selection written to its own artifact.

---

## 4 — Consensus and standing instructions

**Working practice.** Commit to **`feat/eda`** whenever there is a solid advance. Stage explicit
paths — **never `git add -A`**, and never commit credential files. Keep the gates in §1 green before
each commit, and follow the repo's existing commit-message style.

**Decisions taken, do not re-litigate:**

| | |
|---|---|
| `cell8` is **sampled**, not full | 1,970.6 audio hours = ~25 h of decode for 19% of the rows. `spread_by` makes the 2,000-file draw 400 per generator exactly |
| **PANNs is declined** | 32 kHz model against a 16 kHz plane, and it conflates singing with Music — the same blind spot as the VAD, from the other side |
| `stratify_by` does **not** widen | `spread_by: [group_key]` divides the per-source budget instead; same 15,086 files |
| `X4` stays open and blocked | needs the organizers' `TEST_0000–0002.wav`, absent from corpus, repo and S3. A known, accepted gap |
| The S tier is **not** re-run | 8 h. Every tier rides one decode; that lesson cost a run already |

**Standing rules of the harness** (`eda/AGENTS.md` has them in full):

* **R2 — a failure is a row, not an exception.** Nothing in `eda/` drops a file.
* **Gates are tri-state, `fail > na > pass`.** `na` means *not run*, never *found nothing*.
* **Every invariant is paired with a mutation observed to fail.** New checks need new entries in
  `scripts/mutate_eda.py`, and a **STALE** report there means an invariant has lost its pairing.
* **An extractor declares the columns it emits**, enforced on every call.

---

## 5 — Anything else needed for context

* **Read [`docs/EDA/10-final-plan.md`](EDA/10-final-plan.md) first.** It supersedes
  [09](EDA/09-next-steps.md), which is marked as superseded at its top.
* [`docs/EDA/RESULTS_FOR_ANALYSIS.md`](EDA/RESULTS_FOR_ANALYSIS.md) §5 lists the **traps** — five
  misreadings that were actually walked into and corrected. Worth reading before touching the
  numbers, especially §5.1 (a speech VAD cannot evidence singing) and §5.2 (`group_key` is the
  *weaker* holdout, not the stronger).
* **Dated snapshots:** [06 X1b and X1c](EDA/06-cross-pool.md) state their own population (264,085
  rows / 13 sources, pre-WaveFake). Their numbers deliberately do not match today's
  `shortcut_audit.parquet`. Do not "fix" them.
* **Nothing is uncommitted.** The working tree is clean at `5796e54`.
* **Cost measured:** decode runs at **1.33 audio-hours/min** on 32 threads. Budget in *audio hours*,
  never in files — that error made an 8-hour pass look like 2.
* ⚠️ **This machine is shared** (`hkang`, `hgyoo` have jobs on it). Filter by user before killing
  anything, and never `pkill -f` a pattern that appears in your own command line — that self-match
  has silently idled waiter loops here more than once.

---

## 6 — Footer

You can use oh-my-claudecode skills if needed. 
