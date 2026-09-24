Read `docs/HANDOFF_PROCESSING_PIPELINE.md` in full before doing anything. Then, before making
any change, report back to me your understanding of the context: where the branch stands, what
is measured, what is open, and what you intend to do first.

The job: `feat/eda` in `/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting` holds a
data-processing pipeline (`processing/`) for the DACON 236749 deep-voice competition, built from
an EDA, reviewed by four independent reviewers, and remediated under the owner's decisions
(`docs/processing/05-review-findings.md`, `06-remediation-plan.md`). It is built, measured
(`docs/processing/04-verification-report.md`) and committed, 12 commits ahead of origin, not
pushed. The goal now is: re-run the second half of the test suite on HEAD (its last failures were
fixed in the final commit — handoff §3 A), push, open the PR to `main`, then start the first training
run on the built pipeline with `scripts/train.py --processing configs/processing_v1.yaml
--manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v2`.

Three things most likely to bite in the first ten minutes:

1. Use `/data/project/private/dacon-venvs/dacon311/bin/python`; there is no bare `python`.
   Run the test suite in the two halves the handoff gives, and always read `${PIPESTATUS[0]}` —
   `| tail` hides pytest's exit code.
2. The processing sampler refuses the raw manifest (every row is `train`, no fold). Build views
   with `training.folds.apply_folds(manifest, folds, k)` and pass `slice_=` and `fold=`.
   Do not re-run the manifest or fold builders on `strategy-v2` unless the code changed; they
   overwrite the tables.
3. This machine is shared and long jobs die confusingly under `setsid nohup … &` (`$!` is not the
   job). Use the harness's background execution and `pgrep -u $USER -f <script>` before
   assuming anything finished or died. The handoff §5 lists nine mistakes the previous session
   made; read them before running anything long.

You can use oh-my-claudecode skills if needed. 
