We are closing out the EDA for the DACON 236749 deep-voice detection competition.

Read `docs/HANDOFF_EDA_CLOSEOUT.md` in full first, then
`docs/EDA/10-final-plan.md` — that one is the plan of record and the complete
inventory of all 43 declared EDA tasks (19 answered, 7 data-exists, 8 runnable,
7 blocked, 2 Phase 2). `docs/EDA/09-next-steps.md` is superseded and says so at
its top; do not work from it.

Context: the M, S, V and C measurement tiers are all complete — 382,068 files
probed, 58,885 decoded on two planes (as-published and after the competition's
16 kHz chain), plus `[128]`-wide LTAS/mel vectors and Silero VAD. The headline
finding is that **duration is a shortcut that survives both the render chain and
a whole-publisher holdout at AUC 0.852**, against a 0.60 gate, and only the
sampler can fix it.

**Your goal: execute steps 1–3 of `10-final-plan.md`, then update
`docs/EDA/data_memo.md` and `docs/EDA/RESULTS_FOR_ANALYSIS.md` to match.** Both
currently claim the EDA is complete, which it is not.

Start with **step 1, B1** — the WaveFake↔LJSpeech paired vocoder experiment. It
is the only unconfounded real/fake comparison in the corpus (same speaker, same
utterance, one vocoder apart) and the only step that decodes, at ~15 minutes.

Three things most likely to bite in the first ten minutes:

1. **The existing S-tier draw cannot answer B1.** Only 128 of 13,100 LJSpeech
   utterances are incidentally pairable. B1 needs its own targeted selection of
   ids present in all 7 `ljspeech_*` vocoder directories. **Do not redraw
   `sample.json`** — the recorded draw is part of the corpus definition.
2. **`scripts/mutate_eda.py` edits files in place.** Never run it concurrently
   with anything else, and get the test suite green first. Also: do not run the
   full pytest suite while editing `configs/eda.yaml`.
3. **`eda.cli gates` exits non-zero and that is correct.** 6 pass / 4 fail /
   3 `na`; `G-EDA2` is structural and cannot go green in the EDA at all. The
   handoff §3 says which failures are work and which are facts.

Environment: branch `feat/eda`, clean at `5796e54`; interpreter
`/data/project/private/dacon-venvs/dacon311/bin/python`. Measured gates:
245 tests pass, 100/100 mutants killed, links clean over 106 files. AWS is the
EC2 instance role — there are no credentials to handle. The machine is shared
with other users; filter by user before killing any process.

**Before making any change, report back your understanding of the context** —
what is done, what you are about to do for B1, and anything in the handoff you
think is wrong. Then proceed.

You can use oh-my-claudecode skills if needed. 
