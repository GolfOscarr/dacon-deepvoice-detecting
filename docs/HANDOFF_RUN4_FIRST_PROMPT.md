> **Superseded (2026-09-29):** historical first prompt for run 4; the final state is in
> `docs/training/17-final-runs.md`.

You are continuing the DACON deep-voice detection competition (leaderboard closes **Tue 2026-09-29
10:00 KST**, 3 submissions/day, goal: the highest score). Read `docs/HANDOFF_RUN4.md` in
`/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting` in full first, then
`docs/training/16-lessons-learned.md`.

State in one paragraph: the best LB is **0.83046** (300M model on strategy-v5 data, file mode
max3). An **XLS-R-1B model is training on 8 GPUs** (Slurm job 221709, launched from the worktree
`/home/hyeonseop.shin/workspace/dacon-ddp-1b`, branch `feat/ddp-1b`), checkpointing every pass
(~1 h); it ends ~12:20 Monday. Its pass-12 checkpoint scores PROBE 0.9411 (300M: 0.9205) and is
packaged as `1b-v5-p12-max3.zip` in S3 for Monday's first slot. A background watcher scores passes
15/18/21 on PROBE; pass 23 / the final `scored.pt` must be scored by you after the job ends.

Most likely to bite in the first ten minutes:
1. The 15-minute health-check cron is session-only and is gone — re-create it (HANDOFF §3.5).
2. Never `pkill -f` (it killed its own shell last session); stop processes by PID
   (`_ops/probe_watch.pid`). Never cancel jobs you did not start; every job `-J eval-hyeonseop`.
3. All 1B work (packaging, PROBE, restart) runs from the `dacon-ddp-1b` worktree, and every
   package uses `--file-mode max3`.

**Your first action: report your understanding of the state and the plan to the owner before
changing anything.**

You can use oh-my-claudecode skills if needed.
