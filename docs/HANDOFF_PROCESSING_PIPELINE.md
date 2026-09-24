# Handoff — the processing pipeline, from the review to the PR and the first training run

*Written 2026-09-24 by the session that built and remediated the pipeline. Every environment
claim below was re-measured before writing. Anything not measured is marked **UNVERIFIED**.*

## 1. Environment

- **Repository**: `/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting`. One worktree only
  (`git worktree list` shows just this path). Work here.
- **Branch**: `feat/eda`, HEAD = the handoff commit itself (`git log --oneline -1`; the code HEAD is `b2e5fe4`), **12 commits ahead of `origin/feat/eda`, not pushed**,
  no PR. Working tree clean.
- **Interpreter**: `/data/project/private/dacon-venvs/dacon311/bin/python` (never bare `python`;
  it does not exist on this machine). It resolves to this tree:
  `python -c "import processing; print(processing.__file__)"` prints
  `/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting/processing/__init__.py`.
- **Credentials**: none are used. AWS goes through the EC2 instance role; there is no `~/.aws`
  and no `AWS_*` variable. Nothing to pass, nothing to leak.
- **Gates** (each measured on `b66a0ac`/`b2e5fe4`, see §3 for the one open failure):

  ```bash
  V=/data/project/private/dacon-venvs/dacon311/bin/python
  cd /home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
  # lint: ruff is NOT installed; this is the substitute. Measured: exit 1 with exactly four
  # PRE-EXISTING warnings (stream_harness.py 'math' unused; loop.py:441 three re-exported names
  # "imported but unused"). Anything else is new.
  $V -m pyflakes processing scripts/strategy scripts/train.py script.py training/dataset.py training/loop.py training/validate.py
  awk 'length > 100 {print FILENAME": "FNR}' processing/*.py            # no line over 100
  # tests: the full suite in two halves (~10 min each), never in one call
  H1=$(ls tests/test_*.py | grep -v -E "test_render.py|test_codec_roundtrip.py|test_loop.py" | awk 'NR<=24' | tr '\n' ' ')
  H2=$(ls tests/test_*.py | grep -v -E "test_render.py|test_codec_roundtrip.py|test_loop.py" | awk 'NR>24' | tr '\n' ' ')
  $V -m pytest -o addopts="" -q $H1 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
  $V -m pytest -o addopts="" -q $H2 tests/test_render.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
  $V -m pytest -o addopts="" -q tests/test_loop.py 2>&1 | tail -1     # long; ran green (173 with dataset/validate/stages/checkpoint)
  $V scripts/check_links.py                                             # "all links and anchors OK"
  ```

  Always read `${PIPESTATUS[0]}`: a `| tail` masks pytest's exit code, and that once reported a
  red gate as green.

- **Verify you are in the right place**:

  ```bash
  cd /home/hyeonseop.shin/workspace/dacon-deepvoice-detecting && git status --short --branch | head -3 && git log --oneline -1
  ls /data/project/private/dacon-corpus/manifests/strategy-v2/   # manifest, verdict, folds, gate tables
  ```

- **Stale things that would mislead you**: `docs/HANDOFF_EDA_*.md` are earlier handoffs (EDA
  phase), superseded. `configs/run_default.yaml` / `run_t1_strict.yaml` are the OLD training
  sampler's configs; `scripts/train.py` no longer reads them (it reads `--processing
  configs/processing_v1.yaml`). `strategy-v1` under `manifests/` is the pre-review corpus, kept
  for reference; the pipeline runs on **strategy-v2**. `PROGRESS.md` is dated 2026-09-11 and does
  not describe this work.

## 2. Goal of the next session

Take `feat/eda` to a merged PR and start the first training run on the built pipeline.

Checkable outcomes, in order:

1. The second half of the suite re-run green on HEAD (§3 A). Both halves and
   `check_links.py` pass on HEAD.
2. `feat/eda` is pushed and a PR to `main` is open, its description built from
   `docs/processing/04-verification-report.md` §0 table and `05-review-findings.md` resolution
   note. PR descriptions end with the attribution lines in §4.
3. A first scored model: `scripts/train.py --manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v2 --processing configs/processing_v1.yaml --model configs/a_shared_trunk.yaml --weights <BEATs dir> --folds 0 --out runs/<name> --device cuda`
   completes a fold and writes `fold0/scored.pt`, `processing.json`, `val_specs.json`.
   The BEATs weights are at `/data/project/private/dacon-weights/beats/` (measured: it holds
   `BEATs_iter3_plus_AS2M.pt`); `--weights` takes that directory and is required by
   `a_shared_trunk.yaml` (`a_stub.yaml` is weightless and is what the integration test uses).
   **UNVERIFIED**: that `train.py` on the real corpus runs end to end on a GPU — only the
   synthetic-corpus integration test and CPU smoke runs were measured; expect the first real
   run to surface the seams §3 D and the LJ-atom caveat, not silent errors.

First step: `cat` the two suite result files or re-run the second half (§1), then §3's open item.

## 3. Progress and diagnosis

### Done (all committed on `feat/eda`)

| phase | commit | what |
|---|---|---|
| review | `d3fd67f` | four independent reviews → `docs/processing/05-review-findings.md` |
| plan | `4629086` | `docs/processing/06-remediation-plan.md`, owner decisions D1–D15 |
| P1 corpus | `2431bbb` | CFAD pairs bound (regex was wrong), CFAD speakers, LJ-voice atom, artist atom, floor 2.5 s, G-EDA7 per role → **strategy-v2** manifest (351,306 rows) |
| P2 folds | `406a834`, `146e2e9` | PROBE budget on drawable rows, real-side floor, folds balanced on role-hours |
| P3 cache | — | `cache16k` extended to 301,226 files, 0 failures (`cache_report.json`) |
| P4 draw | `20dc735` | bucket tiling, `ComponentDraw.slot`, fail-closed fold views, spec-time augment scalars, dead knobs removed |
| P5 render | `17cd9f7` | overlap crossfades, per-tile + per-sample level, clip, render RNG domain, one thread |
| P6 audit | `a0142f7` | cell 9 in presence probes, presence vs transforms, warnings, folded feature, val view |
| P8 integration | `acae940` | loop/validator/train.py on processing; `processing/infer.py`; `script.py` (P0-a) |
| docs | `33fed44`, `b66a0ac` | 03 as built, 04 re-measured, 05/06 status |
| configs | `b2e5fe4` | run_*.yaml folds sections name the P2 fields (a suite failure found after the doc commits) |

Measured state (`docs/processing/04-verification-report.md` has every number and command):
fold-0 train audit 19/19 at n = 20,000; residues after the shipped chain at chance on every head
(0.48–0.52, n = 6,000); I10/I13/I14 64/64 and bitwise across thread counts; G-EDA4 pass over
18,392 pair sets; G-EDA7 pass (voice gap 0.050); P0-a 0.5000 through `script.py`; the
integration test trains a fold through `train.py` and scores it with `script.py`.

### Open

**A. The second half of the suite has not been run in full on HEAD.** On `b66a0ac` it
reported `3 failed, 716 passed`; a re-run of the same tree named the failures: all three are
`tests/test_run_config.py` (the two example run configs and the alternative-value table lacked
the five P2 `FoldConfig` fields). `b2e5fe4` fixes them and `tests/test_run_config.py
tests/test_config.py tests/test_folds.py` pass (112). The first half passed on `b66a0ac`
(`H1_EXIT=0`); nothing in `b2e5fe4` touches it. Re-run the second half once on HEAD (§1) before
pushing; expect green.

**B. G-EDA3 fails on music/fake** by design (5 composable fake-music families, D-22: four folds,
one family per VAL fold). Recorded in `folds.caveats.txt`. Not a bug; do not "fix" it.

**C. The LJ atom.** LJSpeech real + seven WaveFake LJ families + the LJ-voice MLAAD models are
one indivisible fold atom of 182 h of fake voice; fold 0's VAL is that atom and folds 2–3
validate fake voice on ~8 h. This is the consequence of the owner's D3 (one LJ voice atom) and is
a caveat, not a defect. Per-fold voice numbers carry it.

**D. Content residues are not gated** (04 §3): after the chain, spectral features still read
`voice_fake` at 0.66–0.79 (CFAD's Griffin-Lim / WaveNet vocoders' high band) and presence at
0.75–0.79 (music is denser). These are the signal, reported not gated. Whether a model overfits
CFAD's vocoders is a training-time question.

**E. Owner items still open**: D15 (listen to `g7_examples.csv`, 20 files), Korean fake voice
(owner is sourcing), S-a/S-b shadow slices, FEAT-2, MODEL-1 layers.

## 4. Consensus and standing instructions

Owner decisions of 2026-09-24 (verbatim intent, from `docs/processing/06 §0`):

- D1–D4, D6–D12 "추천대로 진행" (as recommended): PROBE seal on drawable rows with a real floor;
  real voice domain-capped; LJ and artist atoms; hours balance; clip; per-sample RMS target;
  true crossfade; repetition handled by bucket tiling; channel layout accepted under downmix;
  one torch thread ("처리 시간도 중요… 너무 느리다면 차이를 감수하고 빠르게 적용" — if too slow,
  drop the pin and accept the difference; measured cost was negligible, so it stayed); dead
  knobs deleted.
- D5: "버킷 타일링 + take U(1.5, 2.5), margin 0.5" — chosen from four options.
- D2 caveat: "한국어 fake를 확보해야 할 것… 내가 따로 한국어 fake를 찾거나 합성할 예정" — the owner
  sources Korean fake voice; do not wait for it.
- D13: "큰 작업이 예상되므로 실수 없이 잘 진행될 수 있도록 코드 수정 범위와 코드 구조를 잘 계획한 뒤에
  진행" — plan before touching integration; the plan is 06 §2 P8 and it is done.
- D15: "직접 듣겠음" — the owner listens to the G7 examples themselves.
- Earlier: keep D-2 cell mix, D-12 band `[0, 7200]`, D-22 four folds, `f8 = 1.0` (the sweep is in
  `docs/processing/f8_sweep_2026-09-24.log`).
- Reversed on the way (do not re-litigate): a per-slot file budget of 2–4 files (made repetition
  read file length); fake-music buckets by generator family (exhausts on one side); leaving an
  exhausted bucket for the pool (speakers-per-slot read the label). The rule that holds is in
  `processing/sampler.py::_tiles` and 03 DRAW-3.

**Working practice** — commit to the branch whenever there is a solid advance. Stage explicit
paths, never `git add -A`. Keep the gates green (§1) before each commit, and follow the existing
commit-message style (a one-line subject, a body that says what was measured, then the two
attribution lines):

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Lz3hGY4Gxhf5yvgiyJJz7G
```

PR descriptions end with:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Lz3hGY4Gxhf5yvgiyJJz7G
```

Repository prohibitions learned the hard way: never commit credential files; never `pkill -f` a
pattern that appears in your own command line; this machine is **shared** (`hkang`, `hgyoo` run
jobs) — filter by user before killing anything; do not run the full pytest suite while editing
`configs/eda.yaml`; `scripts/mutate_eda.py` edits files in place — never run it concurrently
with anything reading the tree.

## 5. Anything else needed for context

**Mistakes this session made — do not repeat them**

1. **`setsid nohup … &` then `$!`**: when the shell is a process-group leader, `setsid` forks and
   `$!` is the short-lived parent. "The job died" was wrong twice: a fold build and a cache build
   were both still running, and the second cache build raced the first on the same `.tmp.npy`
   files and crashed. Use the harness's `run_in_background` for long jobs, or `pgrep -u $USER -f
   <script>` before assuming a job is gone. Never start a second builder on the same output.
2. **`| tail -3` on pytest** hides which tests failed and masks the exit code. Capture
   `grep -E "^FAILED|passed|failed"` and `${PIPESTATUS[0]}`.
3. **A patch script with several replacements aborts on the first bad anchor and applies none** —
   then the tests you run are on the OLD code. After any bulk patch, confirm the change landed
   (`grep`) before interpreting a test result. This cost one full test cycle.
4. **A replacement block cut at the first blank line swallowed the validation block after it**
   (`DrawConfig.__post_init__` lost `check_mix` and every range check; caught by pyflakes'
   "unused import"). Anchor a replacement on both ends.
5. **Synthetic fixtures are not the corpus.** The synthetic corpus has 2-file real speakers,
   25-file fake families and whole-file rows for single-component cells; on it, `n_files`,
   `n_buckets` and `composed` read the label, and none of that is a pipeline property. Shape the
   fixture like the built corpus (buckets ~100 files on both sides, whole-file rows for cells 5/8
   only) before believing an audit failure — and measure on the real fold view
   (`training.folds.apply_folds`) before believing an audit pass.
6. **A sample-level feature that is a max/min over roles reads the role count**, i.e. the
   presence label (0.556 on a 2 % rate). Presence heads get sample-level features only.
7. **Level features after equalisation**: with RMS fixed, `peak_dbfs` is the crest factor. Sort
   features by what the pipeline can manufacture (level, DC, near-Nyquist) versus content before
   gating on them.
8. **The raw manifest carries no fold.** Every early measurement drew PROBE and VAL rows into
   "train". The sampler now refuses it; go through `apply_folds(manifest, folds, k)` and pass
   `slice_=`, `fold=` explicitly.
9. **Docs claim what the code did not do**: "complementary tapers" were dips to silence; the
   cache docstring described the wrong difference. Measure before describing.

**Outputs and locations**

- Manifest dir: `/data/project/private/dacon-corpus/manifests/strategy-v2/` (manifest, verdict,
  folds, `folds.caveats.txt`, `folds.vg1.txt`, `folds_report.json`, `gates_report.json`,
  `g_eda7_ledger.csv`, `g4_table.csv`, `g7_pack.csv`, `g7_examples.csv`). **Do not re-run**
  `scripts/build_corpus_manifest.py --out …/strategy-v2` or `scripts/build_folds.py` unless the
  code changed: they overwrite these tables (folds' `assigned_at` changes on every rebuild).
- Corpus root `/data/project/private/dacon-corpus`; cache `/data/project/private/dacon-corpus/cache16k`
  (76 GB + the P3 extension; `scripts/build_cache.py` is idempotent and resumable).
- Measurement outputs (gitignored): `eda/out/_strategy/shipped_residues*.parquet`,
  `reproducibility.json`, `reproducibility_specs.json`.
- Configs in force: `configs/processing_v1.yaml` (draw / render / ship / folds / loop); model
  `configs/a_shared_trunk.yaml` (needs `--weights`), `configs/a_stub.yaml` (weightless).

**Costs** (measured): a spec draw 1.2 ms; a render ~0.3 s per sample on one core (mp3 legs spawn
ffmpeg, 70 % of draws); `shipped_residues.py --n 6000 --workers 16` ~10 min;
`verify_reproducibility.py --n 64` ~3 min; the fold-0 audit at n = 20,000 ~2 min; each suite half
~10 min; `test_loop.py` with the training suites ~7 min; the corpus manifest build ~10 min; the
fold build ~5 min. The machine has plenty of cores; parallel background jobs are fine, but never
two builders on one output.

**Documents and what in them is now wrong**: `docs/processing/03` §5 config draft still lists the
three deleted knobs (`single_composed_rate`, `noise_composed_rate`,
`balance_marginal_composedness`) and `run_v1.yaml` names — the file in force is
`configs/processing_v1.yaml`. `docs/pipelines/05-invariants.md` line 22 lists "source offset" as
an I1b feature that the code excludes. `scripts/filter_track_licences.py` docstring still calls
ND undecided (they are kept on purpose). `docs/processing/05` §D lists the rest.

You can use oh-my-claudecode skills if needed. 
