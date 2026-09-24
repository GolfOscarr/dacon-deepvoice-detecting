# 04 — Verification report: the strategy-v1 pipeline against §7 of the spec

*Measured 2026-09-24 on the built `strategy-v1` artifacts (manifest 278,881 rows, 4 folds, the
16 kHz decode cache), branch `feat/eda`. Every number below was produced by a script in this
repository that can be re-run; the command is given with each item.*

The seven acceptance items of [03 §7](03-processing-and-feature-strategy.md#7--acceptance-the-strategy-is-implemented-when),
in one table, then the evidence and the **four issues that need the owner's decision** (§9).

| # | item | status | where |
|---|---|---|---|
| 1 | processing audit 18/18 at n = 20,000 over the built manifest | ✅ | §1 |
| 2 | draw features < 0.60 except the D-2 residual; edge exposure < 1 %; joins at chance | ✅ | §1 |
| 3 | acoustic residues after SHIP: `dc_offset` and `near_nyquist` gone from the top features | ✅ · every acoustic family < 0.60 under the source holdout | ❌ **level** (§3, issue B) |
| 4 | G-EDA6 at full pool-C coverage; G-EDA7 ledger; G4 table; G7 pack | ✅ written · G-EDA7 **fires** (issue A) | §4 |
| 5 | VG1 A1–A7 on `folds.parquet`; G-EDA3 / G-EDA4 | ✅ VG1, ✅ G-EDA4 · G-EDA3 fails on music/fake (5 families, the D-22 caveat) | §5 |
| 6 | I10 across processes, I13, I14 on real samples | ✅ 64/64 · 64/64 · gap 0.0 | §6 |
| 7 | P0-a contract probe returns exactly 0.5000 with the shipped chain in `script.py` | ✅ metric half · ⬜ `script.py` does not exist yet (issue D) | §7 |

---

## 1 · The audit and the draw features (items 1, 2)

`processing.audit.run_audit` over `manifest.parquet`, slice `train`, n = 20,000, seed 0 — the run
that also served as the `f8 = 1.0` arm of the D-1 sweep ([f8_sweep_2026-09-24.log](f8_sweep_2026-09-24.log)):

| check | value | gate |
|---|---|---|
| I1 transform-name independence | worst gap 0.012 (`rir`) | 0.02 |
| I1b metadata AUC | 0.506 pooled | 0.60 |
| I2 / I2c composedness | 0.000 / 0.000 | 0.02 |
| I2b mixedness | gap 0.008 | 0.032 |
| I8 positive rates | file .61 · voice .51 · music .51 · presence .69 / .70 | [0.2, 0.8] |
| I21 families | voice 221 (eff. 90) · music 5 | ≥ 3 |
| I1c draw AUC, fake heads | file 0.507 · voice 0.528 · music 0.534 | 0.60 / 0.61 |
| I1c draw AUC, presence heads | voice 0.598 · music 0.585 | 0.60 |
| I1d edge exposure | 0.000 in every pool | 0.01 |

The presence heads sit 0.002 and 0.015 under their gate. That is the D-2 residual (the
presence-pattern cue of the reference mix, 0.60 by construction in [02 §4.3](02-analysis-report.md#43-the-cell-mix-has-the-same-shape-of-gap)),
which the owner kept on 2026-09-24; it is not a draw defect and it does not move with `f8`.

## 2 · The `f8` sweep (D-1)

| `f8` | pass | what fails |
|---|---|---|
| 1.0 | 18 / 18 | — |
| 0.5 | 15 / 18 | I2c (gap 0.021); voice_present 0.635; music_present 0.624 |
| 0.0 | 13 / 18 | + voice_fake 0.636, music_fake 0.637 in the mixed stratum |

Once any mixed sample is whole-file, "composed" reads presence (single-component cells are
always composed), and at `f8 = 0` it reads fake inside the mixed stratum (cells 6 / 7 must be
composed while 8 is not). `f8 = 1.0` stays; SONICS' 1,973 h remain unused, as D-1 accepted.

## 3 · Acoustic residues on the shipped audio (item 3)

`scripts/strategy/shipped_residues.py --n 6000 --workers 16` — 6,000 specs from the processing
sampler, rendered through the cache and the full menu (augments, codec / telephone normalize),
then `processing.ship` (downmix → `dc_offset` → band `[0, 7200]`), then the EDA's own level and
spectral extractors over the **sample**, and the harness's CV AUC per head, ungrouped and under
the `source_name` holdout. The pre-ship downmix is measured as the control. Written to
`eda/out/_strategy/shipped_residues*.parquet` (gitignored; re-run to regenerate).

**What SHIP removed.** After the chain, `|dc_offset| ≤ 5.6e-8` and `near_nyquist_ratio ≤ 3.4e-5`
on every sample (pre-ship maxima 1.068 and 160). Neither appears among the top five features of
any head after the chain; before it, `dc_offset` was the top feature on `music_fake` (0.634) and
on `file_fake` (0.600). The first half of item 3 holds.

**What remains** (grouped AUC = under the source holdout; gate 0.60):

| head | stratum | pre: grouped | ship: grouped | top features after SHIP |
|---|---|---|---|---|
| file_fake | pooled | 0.510 | 0.511 | rms 0.543 |
| file_fake | mixed | 0.563 | 0.560 | rms 0.567 |
| file_fake | music-only | 0.617 | **0.630** | rms 0.647 · peak 0.634 |
| file_fake | voice-only | 0.517 | 0.571 | rms 0.600 |
| voice_fake | pooled | 0.526 | 0.523 | hf_ratio 0.542 |
| voice_fake | mixed | 0.514 | 0.518 | hf_ratio 0.530 |
| music_fake | pooled | 0.612 | **0.612** | rms 0.626 · peak 0.609 · crest 0.572 |
| music_fake | mixed | 0.606 | **0.602** | rms 0.617 · crest 0.595 |
| music_fake | music-only | 0.617 | **0.630** | rms 0.647 · peak 0.634 |
| voice_present | pooled | 0.666 | **0.667** | crest 0.618 · peak 0.618 · rms 0.595 |
| music_present | pooled | 0.777 | **0.781** | rms 0.756 · peak 0.719 · crest 0.620 |

Every failure is **level** (RMS, peak, crest), and the chain neither adds nor removes it. Two
distinct mechanisms:

* **`music_fake` reads level at 0.61–0.63 across 975 sources.** The harness predicted this:
  [02 §6](02-analysis-report.md#6--level) measured grouped music AUC 0.683 with no level
  transform, 0.453 with per-file normalisation, 0.599 with the ±12 dB jitter that shipped — and
  chose the jitter because normalisation makes a *voice* cue (0.786). The jitter lands at 0.61 on
  the sample, 0.01 over the gate. This is the D-9 trade-off, not a new defect. → issue B.
* **The presence heads read level at 0.67 / 0.78.** A mixed sample is the *sum* of two components
  drawn at comparable gains, so it is louder than a single-component one by construction; the
  0.05-weight presence heads were never gated on level in [02](02-analysis-report.md) (the
  harness's presence acoustic set was onset deficit, mean RMS, min bandwidth, measured per file
  not per sample). → issue B, second half.

`voice_fake` is clean (≤ 0.53 everywhere); `file_fake` fails only inside music-only, where it
*is* `music_fake`.

## 4 · G-EDA6, G-EDA7, G4, G7 (item 4)

`python -m eda.cli gates` re-run at the 99.97 % pool-C VAD coverage of step 7: G-EDA6 reports
3,206 rows contradicting their asserted components over 8 sources — this is the worklist OFF-2
consumed, and every row is acted on in the manifest (2,100 reassigned: C → 5 1,094, D → 8
1,006; 1,028 pool-E rows restricted; 23 degenerate dropped; 42 unevidenceable and 3 sparse-real
kept as they were, recorded).

`scripts/verify_gates.py --manifest-dir …/strategy-v1` writes `gates_report.json`,
`g_eda7_ledger.csv`, `g4_table.csv`, `g7_pack.csv`, `g7_examples.csv` beside the manifest.

**G-EDA7 ledger** (drop rate per filter and label, over the 382,068 probed files):

| filter | real: n / dropped / rate | fake: n / dropped / rate | gap | tol |
|---|---|---|---|---|
| corruption | 97,700 / 3 / 0.000 | 284,368 / 0 / 0.000 | 0.000 | 0.10 |
| licence | 8,000 / 23 / 0.003 | — | 0.003 | 0.10 |
| label_evidence | — / 0 / 0.000 | 23 / 23 / 1.000 | 1.000 | 0.10 |
| usable_duration | 97,700 / 17,219 / **0.176** | 284,368 / 85,919 / **0.302** | **0.126** | 0.10 |

**G4 loss table** (rows dropped per pool by `usable_duration`, the 4.0 s component floor):

| pool | probed | dropped | rate | hours dropped |
|---|---|---|---|---|
| A real voice | 74,846 | 17,023 | 22.7 % | 13.3 |
| B fake voice | 207,689 | 85,919 | 41.4 % | 68.3 |
| C real music | 8,660 | 3 | 0.03 % | 0.0 |
| D fake music | 27,605 | 0 | 0 | 0 |
| E noise | 14,194 | 193 | 1.4 % | 0.1 |

Both voice pools exceed G4's 5 % escalation line, as [02 §11](02-analysis-report.md#11--manifest-actions-in-hours)
said they would (pool B 22.5 % of hours); the owner fixed the floor at 4.0 s on 2026-09-24. → issue A.

**G7 pack** (reassigned per corpus; ten examples per move in `g7_examples.csv`):

| corpus | from | to cell | reassigned | of probed | rate |
|---|---|---|---|---|---|
| fma | C | 5 | 1,032 | 8,000 | 12.9 % |
| musan-music | C | 5 | 62 | 660 | 9.4 % |
| fakemusiccaps | D | 8 | 1,006 | 27,605 | 3.6 % |

The examples are listed, not yet listened to; G7's "confirm the detector is right" is a human
step. → issue C.

## 5 · Folds (item 5)

VG1 A1–A7 passed at build time (`folds.vg1.txt`, step 10). On the built table:

* **G-EDA4 ✅** — 0 of 11,154 multi-row `pair_id` sets and 0 of 38 `dup_group` sets straddle a
  fold or the PROBE / fold boundary (a PROBE row has no fold; the check spells that out rather
  than letting NaN drop out of the count).
* **G-EDA3** — atoms per (role, side): voice/real 132 · voice/fake 222 (222 families) ·
  music/real 2,203 · **music/fake 5 (5 families)** · noise 10,722. The music/fake side is the
  D-22 caveat (FakeMusicCaps' five generators; SONICS is whole-file-only and never drawn under
  `f8 = 1`): the fold count already dropped to 4 and the caveat is in `folds.caveats.txt`, which is
  the response [EDA/07](../EDA/07-order-and-gates.md) prescribes. Nothing further to build.

## 6 · Reproducibility on real samples (item 6)

`scripts/strategy/verify_reproducibility.py --n 64` → `eda/out/_strategy/reproducibility.json`:

* **I10** 64 / 64 specs identical across two fresh interpreters (sha256 of the waveform bytes and
  the frame intervals; 33 of the 64 carry a noise layer, every one goes through the menu).
* **I13** 64 / 64: every role branch equals that role's placement with tiles merged and the
  cell's label; the file branch lists each component's span with its own label, abutting
  same-label spans merged. (Two earlier versions of the *check* were wrong, not the renderer:
  it expected the file branch to be a union, then to keep abutting same-label entries apart.)
* **I14** max |solo − in-batch| = 0.0 over the valid prefix, 64 real samples, mono and stereo in
  one batch, lengths 66,666 to 955,816 samples, hostile padding.

## 7 · P0-a (item 7)

`scripts/verify_metric.py`: `[OK] all-constant submission scores exactly 0.5000 (any constant,
any column)`. The other half — the shipped chain inside the submission's `script.py` — cannot be
run: there is no `script.py` in the repository yet (it is Phase F, the packaging). → issue D.

## 8 · A defect found on the way

The shipped-residue run raised on the 42nd spec: `processing.render._compose` sized its canvas
to the piece with the most channels and `_to_channels` can broadcast mono to N or keep leading
channels, but cannot lift 2 to 8 — so a stereo piece met an **8-channel** canvas. Pool E holds
92 microphone-array recordings (RIRS isotropic noise: 90 × 8 ch, 2 × 30 ch) that no test
corpus had. Fixed by capping the composite at `MAX_CHANNELS = 2` (the test set is mono and
stereo; an array piece keeps its leading two channels), with a test that composes an 8-channel
piece over a stereo one. `training/render.py` has the same canvas rule and the same latent
defect; it is untouched, since its corpus has no such file and the processing pipeline is the
one being shipped.

## 9 · Issues for the owner

**A · G-EDA7 fires on the duration floor.** The 4.0 s floor drops 30.2 % of fake rows against
17.6 % of real (gap 0.126 over the 0.10 tolerance), because pool B is short. The gate's remedy
is "widen or drop". *Evidence that it does not manufacture a cue:* the draw never reads file
length (D-21's uncapped take), I1b's metadata AUC is 0.506 and the presence / fake draw AUCs are
at chance; the asymmetry is in what was *removed*, not in what the model sees. Options:
(1) accept and record the ledger as the standing exception, with this evidence; (2) widen the
real side to match rates (drop ~12 % more of pool A — symmetric by construction, costs ~23 h
of real voice); (3) lower the floor (declined on 2026-09-24). `label_evidence` also shows
1.000 vs 0.000, on 23 rows: those are the degenerate pool-B files (no speech at all) and there
is no real-side counterpart to a fake file with no content; recording it is the only option.

**B · Level survives SHIP.** `music_fake` reads level at 0.61–0.63 under the source holdout;
the presence heads at 0.67 / 0.78. Options: (1) accept (D-9's measured trade-off; 0.01 over the
gate on the head that matters, and the presence heads carry 0.10 of the score); (2) widen the
gain jitter (±12 → ±18 dB) and re-measure — cheap, but the harness saw diminishing returns;
(3) per-sample RMS normalisation *after* composition at render time — removes the mixed-sums-
louder presence cue by construction, but [02 §6](02-analysis-report.md#6--level) measured
per-file normalisation making a voice cue (0.786) and that would have to be re-measured at the
sample level; (4) treat it as the model's problem and gate on VAL EER. This is a strategy
change, so it is not made here.

**C · G7 wants ears.** 1,100 pool-C and 1,006 pool-D rows were reassigned by the VAD; the pack
lists ten of each to listen to. Nobody has.

**D · P0-a is half-measurable.** The metric half returns 0.5000; the packaging half needs the
submission `script.py`, which does not exist. It stays open until Phase F, unless the owner wants
a minimal `script.py` written now (decode → ship → constant 0.5 rows) to close the contract.

---

*Scripts: `scripts/verify_gates.py`, `scripts/strategy/shipped_residues.py`,
`scripts/strategy/verify_reproducibility.py`, `scripts/verify_metric.py`. Gates on this commit:
`pytest` 661 + 688 passed in the two halves (`tests/test_render.py` included in the second);
`pyflakes` clean; no line over 100 columns.*
