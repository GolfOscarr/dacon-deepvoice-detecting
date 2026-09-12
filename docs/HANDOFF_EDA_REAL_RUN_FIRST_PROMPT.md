Read `docs/HANDOFF_EDA_REAL_RUN.md` in full before doing anything else, then read
`docs/EDA/README.md` and `docs/EDA/08-real-run.md` — the plan is current, not superseded.

**Context.** This is a DACON competition entry for AI-audio detection (deep-voice / fake-music),
leaderboard closing 2026-09-29. We built an EDA package at `eda/` and just completed the first real
run over wave 1 of the corpus: 7 sources fetched to
`/data/project/private/dacon-corpus/{raw,interim}`, 90,707 files probed, 5 partitions consolidated,
and the shortcut audit run. **It failed, and the failure is fully diagnosed.**

**The diagnosis you are inheriting.** Three metadata columns make the corpus trivially separable,
and nothing else does. `duration_s` alone predicts `MUSIC_FAKE` at **AUC 1.000** — FakeMusicCaps is
10.0 s for all 27,605 files, `fma_small` is 30.0 s at both the 5th and 95th percentile — on the head
carrying 0.27 of the metric. Format and channel count carry the rest. Remove all three and all four
heads sit at exactly 0.500, so there is no fourth confound hiding underneath.

The work lives on the **`eda`** branch, not `main` — `main` is parked at `origin/main` and the
EDA lands on it as a PR.

**Your goal**, in order:

1. **Corpus decisions the run has forced** (config-only): drop `rirs-pointsource` — 843 of 843 files
   are byte-identical MUSAN copies — and move `rirs-isotropic` out of pool E, since it is 8-channel
   with a 1.4 s median and is impulse responses, not noise. Then confront the consequence: **pool E
   becomes one independent source**, which `build_folds` cannot rotate on.
2. **Finish wave 1**: fetch SONICS so the `cell8` partition exists and whole-file rows are audited
   for the first time.
3. **Build Phase 1, the S tier**: `eda/planes.py` plus the level/timing/spectral extractors.

**Three things most likely to bite you in the first ten minutes:**

- **`G-EDA2` will keep reporting AUC 1.000 no matter what you fix, and that is correct.** X1 reads
  the corpus as it sits on disk; the fixes are *render-time transforms*. Chasing that number is
  chasing something that cannot move, and deleting features to make it green would hide the
  confound rather than close it. The verification belongs to `audit_specs` over the spec stream.
- **Use the venv**: `/data/project/private/dacon-venvs/dacon311/bin/python`. Plain `python3` lacks
  sklearn and will fail on import. `training/` has no `__init__.py` and is a namespace package, so
  `training.__file__` is legitimately `None`.
- **Do not run the full pytest suite while editing `configs/eda.yaml`.** Three tests assert on the
  shipped config; editing it mid-run produces failures that are artifacts of the race, not defects.

**Before you change anything, report your understanding of the context back to me** — what the three
confounds are, why `G-EDA2` cannot be made green by the EDA, and what you intend to do first.

You can use oh-my-claudecode skills if needed. 
