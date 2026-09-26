# Run 2 results

*2026-09-26. Run `run2-v4` (job 220668, 8 tasks, strategy-v4, launched 07:17 KST). Training ended
14:05–14:41. Design: docs/training/11. Every number below is measured. The tools are
`scripts/diag/post_run2.sbatch` (jobs 220767, 220885) and `scripts/diag/run_breakdown.py`.*

## 1 · Leaderboard calibration (run 1)

`first-v3-seed0` scored **0.79188** on the leaderboard: ADS 0.77288, CPS 0.96287.
- ADS gives a metric-weighted EER of **0.227**, against ≈ 0.05 on our VAL.
- CPS gives a mean presence AUC of 0.963, against ≈ 1.000 on VAL.
- The leader is at 0.89871 (ADS 0.88756, CPS 0.99907).
- 96 % of the gap to the leader is ADS.

The per-head split is pending the diagnostic submissions `first-v3-diagA/B`
(`script.py` `model/diag_constant.json`, `2cb5611`).

## 2 · Like-for-like on run 1's exact VAL specs (v3, `processing_first_run.yaml`, n 6000)

| fold | run 1 | run 2 | file EER | voice EER | music EER | diagnosed slice (run 1 → run 2) |
|---|---|---|---|---|---|---|
| 0 (T0) | 0.928 | **0.940** | .101 → .080 | .057 → .066 | .061 → .046 | real-LJ false alarm **.52 → .27**; T3 matched-pair gap **.153 → .083** (VG4 now passes) |
| 1 (T1) | 0.916 | **0.952** | .101 → .059 | **.179 → .090** | .024 → .020 | ko-synth/xtts miss **.31 → .09**; real musan false alarm **.49 → .25**; CV-ko false alarm .048 → .018 |
| 2 (T2) | 0.990 | 0.979 | .011 → .026 | .023 → .050 | .002 → .002 | **ko-synth/melo miss .05 → .19** (p̄ .88 → .50) |
| 3 (T3) | 0.966 | 0.955 | .044 → .061 | .012 → .036 | .043 → .040 | **ko-synth/mms miss .003 → .17** (p̄ .95 → .52); CV-ko false alarm .25 → .11 |
| mean | 0.950 | **0.956** | | | | |

Controls:
- **T5** (fold 0, run 1 init, v3 corpus, run 2's draw and length): 0.944. LJ false alarm .49,
  T3 gap .146. **The fold-0 fix comes from the new data**, not from steps or the draw.
- **T4** (fold 1, scratch, run 1 recipe, 20k steps): 0.928 vs T1 0.952 on the same specs.
  **B-init wins** at equal steps; XTTS miss .086 for both.

**Regression.** The clean VITS-style Korean TTS families (MeloTTS, MMS-TTS) are missed when held
out, and clean real read speech gets a little more suspicion: LibriTTS-R false alarm .002–.007 →
.039–.058.
- Reading: the new real side (Emilia, and LJ ×25 in the draw) removed "clean ⇒ fake".
- Unseen clean single-speaker TTS now sits nearer the real side.
- The all-data models train on melo and mms; the risk is **unseen** VITS-style TTS in the test.

## 3 · PROBE (the frozen draw run 1 was ranked on; `processing_first_run.yaml`, n 3000, seed 4321)

| model | score | file EER | voice EER | music EER |
|---|---|---|---|---|
| run 1 all_data_seed0 (control: reproduces 0.8525) | 0.8525 | .163 | .276 | .090 |
| run 2 T6 (init seed0) | 0.9067 | .119 | .068 | .101 |
| **run 2 T7 (init seed2)** | **0.9103** | .120 | .061 | .091 |

Music did not move: run 2 changed nothing on the music side.

## 4 · Submissions prepared (verified under server-mirror on 8 clips, 0 fallback; in S3)

- `run2-T7.zip` (sha256 `6d040a19d752a358…`) — **recommended**.
- `run2-T6.zip` (`78344d79ec9be8c8…`).
- `first-v3-diagA.zip`, `first-v3-diagB.zip` — run 1 head split.

## 5 · Speed (8 concurrent tasks)

- ~1.0–1.2 s/step (single task: 0.86); `data_wait_s` 1–10 s per 50 steps.
- Peak ≈ 104 GiB per GPU with checkpointing off.
- 20k steps: 6 h 50 m – 7 h 25 m wall, including the end-of-run VAL.
