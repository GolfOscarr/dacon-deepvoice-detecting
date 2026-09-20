# 02 — Analysis report: what the stream harness found

**Run 2026-09-19** against the plan in [01](01-analysis-plan.md). Two scripts, no audio decoded,
every number below re-derivable from `eda/out/_strategy/`:

```bash
V=/data/project/private/dacon-venvs/dacon311/bin/python
$V scripts/strategy/stream_harness.py     # §2-§5, §11: 20,000 specs per variant, seed 0   (~25 min)
$V scripts/strategy/scalar_threads.py     # §6-§10: the scalar threads                       (~8 min)
```

> **The one-line version.** On the stream the model actually receives, the current draw
> (`configs/run_default.yaml`) hands the music head a **0.995 AUC** shortcut it manufactured itself —
> pool-D components are 10 s files on 4–60 s timelines, so a fake-music sample is 53 % digital
> silence and its first frame is exposed 87 % of the time. A crop that takes **3–8 s from strictly
> inside every file and tiles it** removes that; the **`f8 = 0` policy leaks composedness on both
> component heads and only `f8 = 1` closes it**; and with those two changes plus the G-EDA6
> reassignment every *draw*-level feature is at chance on every head. What remains is acoustic
> (level, DC, the resampler shelf) and lands in the shipped preprocess.

---

## 0 — Findings, ranked by what they change

| # | finding | number | where it lands |
|---|---|--:|---|
| 1 | 🔴 **The draw manufactures the music shortcut.** Pool-D files are shorter than the timeline, so `take = file` and `offset = 0`: fake-music samples are 52.8 % silence vs 13.9 % for real, and the first frame is exposed 87.2 % vs 32.6 % (H1, H2 confirmed) | draw AUC **0.995** on `music_fake`; **0.915** on `music_present` | §2 |
| 2 | 🔴 **A random-offset crop of 3–8 s from strictly inside the file, tiled to the timeline, puts every draw feature at chance in the music-only stratum** | 0.993 → **0.497** | §3 |
| 3 | 🔴 **`f8 = 0` (the primary policy) leaks composedness on the component heads, and no cell mix can fix it** — the stratified constraint I2 balances the *file* head only. Derived: with `f6 = f7 = 1`, the component heads balance only at `f8 = 1` | `P(composed \| music_fake=1, mixed)` **0.575** vs 1.000 → draw AUC 0.73; strict: **0.509** | §4 |
| 4 | 🔴 **The reference cell mix balances mixedness for the file head and not for the component heads** — "music present" predicts voice-fake at 0.60 and vice versa; the symmetric case needs `p6 = p7 = 0` to balance all three | gaps: file 0.002 · music **0.204** · voice **0.195** | §4 |
| 5 | 🔴 **EDA correction: the vocoder does *not* change leading silence.** On true same-utterance pairs the lead is identical in 99.2–100 % of pairs; A6b's 0.639 came from WaveFake's **JSUT** (Japanese) subsets, whose lead median is 0.05–0.10 s | paired AUC **0.50**; LJSpeech vs its own vocoded copies **0.502** | §5 |
| 6 | **Silence augmentation on, drawn for every sample, `U(0, 3)` s at `p = 1`** — brings the residual to 0.55 corpus-wide; `trim + redraw` is the only thing that reaches 0.50 within CFAD | 0.595 → **0.553** (corpus), 0.652 → 0.599 (CFAD) | §5 |
| 7 | **Level: do not normalise; jitter ±12 dB.** Peak/RMS normalisation halves the music residue (0.683 → 0.453) but *raises* the voice residue (0.586 → **0.786**, crest transfers across archives). The vocoder shifts RMS on the same utterance by −0.25…−1.24 dB in 100 % of pairs — a real but tiny cue | jitter ±12: music 0.599, voice 0.579 | §6 |
| 8 | **DC is a generator id that inverts across archives; remove it. Do not high-pass.** `\|dc\|` A-vs-B 0.806 ungrouped, **0.392** source-grouped; `cfad/gl` vs `cfad-real` 0.994. A 40 Hz high-pass removes 25–75 % of the pair difference energy for 4 of 7 vocoders | — | §7 |
| 9 | **The resampler shelf and SONICS' 36.8 kbps MP3 are acoustic fingerprints the chain leaves.** `near_nyquist_ratio_chain` 0.29 (wavefake), 0.22 (ljspeech), 0.16 (fma) vs 0.06–0.08 for native-16 kHz sources and **0.000** for sonics (7.3 kHz rolloff) — the top acoustic residue on the music head | `music_near_nyquist` 0.744 | §8 |
| 10 | **Every source is nameable from 128 LTAS numbers** (one-vs-rest ≥ 0.980, eight at ≥ 0.995) — `aug_strength` cannot be conditioned on identifiability; it is uniform | — | §9 |
| 11 | **WaveFake is 3 artifact families, not 7**, on both planes: {full_band_melgan, hifiGAN, parallel_wavegan, waveglow}, {melgan, melgan_large}, {multi_band_melgan}. Music families are 5 + 5 domains; `domain_cap` **does not reach whole-file rows** (uniform draw confirmed: 354–381 per SONICS generator) | — | §10 |
| 12 | **G-EDA6 in hours** is small: 9.0 h moved, 1.2 h restricted; the 4 s floor removes **22.5 % of pool B's hours** (68 h), 6.8 % of A's; a 2 s component floor recovers 63 h of B | — | §11 |

---

## 1 — What was run

| artifact | what |
|---|---|
| `eda/out/_strategy/manifest_stier.parquet` | a `training.manifest` over the 58,885 S-tier rows (56,885 components, 2,000 whole-file), validated |
| `stream_<policy>.parquet` | one row per drawn `SampleSpec` with every component's measurements and the effective features (`eff__*`) |
| `stream_audit.parquet` · `stream_audit_P9.parquet` · `stream_audit_final.parquet` | the audits: head × family × stratum × policy |
| `run_audit_{baseline,M1reassigned,F8strict}.txt` | `training.audit.run_audit` on the same streams |
| `silence_policies.parquet` · `level_policies.parquet` | §5, §6 grids |
| `b1_pair_tests.parquet` · `dc_per_cfad_generator.parquet` · `highpass_surviving_difference.parquet` · `vector_audit.parquet` · `source_identifiability.parquet` · `domain_hours.parquet` · `four_second_floor.parquet` · `bandwidth_per_source.parquet` | §6–§11 |
| `harness.log` · `scalar.log` | full console output |

**Method, in one paragraph.** `training.sampler.Sampler` draws 20,000 specs at seed 0 from
`configs/run_default.yaml`. Each `ComponentDraw` is joined to `signal.parquet` and
`envelope.parquet` on `file_id`, and the *effective* features are derived: component take,
timeline coverage, whether the file's first / last frame is inside the take (`offset < 10 ms`),
the file's leading silence *if* its onset is exposed, level after gain, bandwidth, DC. Two
families are audited separately because they answer different questions — **draw** features
(lengths, coverage, exposure, joins, composedness) carry no acoustic content, so any AUC above chance
*ungrouped* is a cue the pipeline created; **acoustic** residues (RMS, crest, DC, bandwidth,
near-Nyquist, onset deficit) are legitimately received, so the question is whether they survive a
`source_name` holdout. Logistic regression, 5-fold; grouped `StratifiedGroupKFold` (2-fold where a
head has under five sources — the music head has four); per stratum, worst taken.

**Policies are simulated on the drawn stream** — take, offset and timeline are redrawn with a policy
RNG while labels and files stay fixed. That is the same measurement a sampler change would produce;
the sampler change is the implementation (§13).

⚠️ Limits: the harness sees the features it is given (§14). `stream_P7.parquet` on disk is the
`+lead_U0-3` variant (a naming collision, fixed in the script for future runs); the plain P7 audit
rows are in `stream_audit.parquet`.

---

## 2 — Baseline: what `run_default.yaml` draws today

**H1, H2, H3 are measured, and all three hold.**

| music pool | `P(offset = 0)` | take median | uncovered fraction of timeline |
|---|--:|--:|--:|
| C (real) | 0.326 | 26.9 s | 0.139 |
| D (fake) | **0.872** | 10.0 s | **0.528** |

The integral in [01 §2.4](01-analysis-plan.md) predicted 0.573 / 0.164 for music-only timelines;
the mixed cells pull both down slightly. 3,795 of 58,885 S-tier rows are dropped by the 4 s floor.

**Draw-feature audit (ungrouped / `source_name`-grouped):**

| head | stratum | AUC | grouped | top features |
|---|---|--:|--:|---|
| `music_fake` | pooled | **0.995** | 0.226 | `music_take_s` 0.729, `music_coverage` 0.700, `uncovered_frac` 0.694 |
| `music_fake` | music-only | **0.993** | — | `music_take_s` **0.896**, `uncovered_frac` 0.882, `music_onset_exposed` 0.736 |
| `music_fake` | mixed | 0.993 | 0.248 | `composed` 0.712, `voice_coverage` 0.702 |
| `music_present` | pooled | 0.915 | **0.837** | `uncovered_frac` **0.848** |
| `voice_present` | pooled | 0.789 | 0.681 | `uncovered_frac` 0.698, `any_lead_silence_s` 0.633 |
| `voice_fake` | pooled / mixed | 0.843 / 0.870 | 0.317 / 0.733 | `composed` 0.715, `music_coverage` 0.748 |
| `file_fake` | pooled / music-only | 0.854 / 0.993 | 0.439 | `uncovered_frac` 0.622; music-only as `music_fake` |

Every head fails the 0.60 gate on draw features alone. ⚠️ The grouped column is *not* the gate for
draw features: a length cue that inverts across publishers is still a length cue.

**`training.audit.run_audit` on the same stream**: I1b **FAIL** 0.634 (mixed), I2 **FAIL** 0.272
(mixed), I2c FAIL 0.153; the rest pass. I2 fails because the S-tier corpus has whole-file rows for
cell 8 (SONICS) and none for cell 5, so `f5 = 1` and `f8 = 0` — the composition trap, live in the
shipped config over the real corpus (§4).

**Acoustic residues, baseline** (ungrouped / grouped): `music_fake` 0.928 / 0.594
(`music_near_nyquist` 0.744, `music_rms_dbfs` 0.717); `voice_fake` 0.854 / 0.485, voice-only stratum
0.829 / **0.787** (`voice_dc_offset` 0.822, `voice_bandwidth_hz` 0.722); `music_present` 0.751 /
0.702 (`mean_rms_dbfs` 0.728). These are what §6–§8 address.

---

## 3 — Crop policies

Draw-feature AUC on `music_fake` (ungrouped), the R2 check, and the cost:

| policy | rule | pooled | mixed | music-only | `P(off=0)` C / D | uncovered C / D |
|---|---|--:|--:|--:|---|---|
| **P0** baseline | `take = min(span, file)` | 0.995 | 0.993 | 0.993 | 0.33 / 0.87 | 0.14 / 0.53 |
| P1 | music `take ~ U(4, 9.5)`, offset random | 0.772 | 0.749 | **0.485** | 0.00 / 0.01 | 0.58 / 0.62 |
| P1x | music `take ~ U(4, 30)` — why `t_hi < 10` | 0.969 | 0.967 | 0.953 | 0.00 / **0.67** | 0.37 / 0.55 |
| P1+P2 | P1, timeline shrunk to cover | 0.767 | 0.750 | 0.510 | 0.00 / 0.00 | 0 / 0 |
| P0+P3 | tile without re-drawing take — why P1 first | 0.995 | 0.995 | 0.992 (`music_joins` 0.883) | 0.33 / 0.87 | 0 / 0 |
| P5 | every role take 4 s, tiled | 0.755 | 0.708 | 0.494 | 0.00 / 0.00 | 0 / 0 |
| P8 | every role `U(2, 6)`, tiled | 0.760 | 0.713 | 0.524 | 0.00 / 0.00 | 0 / 0 |
| **P7** | every role `U(3, 8)`, tiled | 0.766 | 0.734 | 0.497 | 0.00 / 0.00 | 0 / 0 |
| **P9** | P7 with `take ≤ file − 0.5 s` (strictly inside) | 0.762 | 0.725 | 0.510 | 0.00 / 0.00 | 0 / 0 |

Three things the table says:

1. **Any take that can equal the file length re-creates H1 for that file.** P1x leaves pool D at
   `P(offset = 0) = 0.67`; tiling without a bounded take (P0+P3) turns the join count into the same
   cue (0.883). And P7 still exposed the onset of **78 % of noise components** (CompSpoof clips are
   exactly 4.00 s, so `U(3, 8)` capped at 4 hits the file) and 20 % of voice components — which is why
   `first_onset_exposed` was the top presence-head feature (0.658) under P7. **P9's margin** takes
   onset exposure to 0.8 % (voice) and 1.9 % (noise), and `music_present` / `voice_present` draw
   AUC from 0.745 / 0.647 (P7, main run; 0.748 / 0.653 in the P9 run's own re-draw) to **0.636 / 0.635** (`sequential` and `composed` are what is left, §4).
2. **The music-only stratum is clean under every bounded-take policy** (0.485–0.524). The pooled and
   mixed numbers stay at 0.73–0.77 for a reason unrelated to cropping: `composed` (§4).
3. Take length inside 2–8 s does not matter to the audit (P5 / P7 / P8 within 0.03). The choice is a
   modelling one — ★ `[BirdCLEF playbook]` D4's 5 s default, 8–10 s for context — and the join count
   must stay label-independent, which tiling *every* role at the same rule guarantees (`*_joins`
   ≤ 0.516 on every head under P9).

**Decision: P9** — every component role takes `U(3, 8)` s from an offset drawn so the take lies
strictly inside the file (`take ≤ file − 0.5`, offset `U(0, file − take)`), tiled to its span with
A-A4 sigmoid joins. Timeline stays `U(4, 60)`, drawn first. Whole-file rows go through the same rule.

---

## 4 — 🔴 Composedness: the constraint the design has is not the one the component heads need

### 4.1 Measured

In the mixed stratum of the baseline stream:

| head | `P(composed \| fake)` | `P(composed \| real)` | draw AUC (mixed) |
|---|--:|--:|--:|
| `file_fake` | 0.728 | 1.000 | 0.830 |
| `music_fake` | **0.575** | 1.000 | 0.993 (crop-dominated) → **0.734** after P7 |
| `voice_fake` | **0.570** | 1.000 | 0.870 → 0.765 after P7 |

With the G-EDA6 reassignment applied (**M1**: 336 real cell-5 whole-file rows and 1,006 cell-8 rows
added; 1,055 dropped), `run_audit`'s **I2 passes** (0.000 gap on the file head) — and the
component heads still read `composed = 0.634` in the mixed stratum (`music_fake` mixed 0.644).
Under **`f8 = 1` (strict)** the same numbers are 0.512 (`music_fake` mixed) and 0.587 (`voice_fake`
mixed), and the pooled `music_fake` draw AUC falls from 0.766 to 0.608.

### 4.2 Why — derived, not fitted

For `music_fake` inside the mixed stratum the fake cells are 6 and 8, the real ones 5 and 7. With
`f6 = f7 = 1` forced (they cannot be scraped):

```
P(composed | music_fake = 1) = (p6 + p8·f8) / (p6 + p8)
P(composed | music_fake = 0) = (p5·f5 + p7) / (p5 + p7)
```

I2 fixes `f5 = (p6 + p7 + p8·f8) / (p6 + p7 + p8)` so the *file* head balances. Substituting the
reference mix: at `f8 = 0` the two sides are **0.568 vs 0.848**; they are equal only at `f8 = 1`.
Solving the component constraint alone gives `f5 = 0.22`, which then breaks the file head (0.725 vs
0.22). **In general — any cell mix, not only a symmetric one** — substituting I2's `f5` into the
music-head condition leaves the residual (checked with sympy)

```
p7 · p8 · (f8 − 1) · (p5 + p6 + p7 + p8) / ((p5 + p7)(p6 + p8)(p6 + p7 + p8))
```

and the voice head the mirror image with `p6 · p8`. It is zero only when `f8 = 1`, or a cell that
defines the head (`p8`, or `p6` / `p7`) is empty. The label-conditional policy was chosen in
[data/02](../data/02-label-taxonomy.md#-decided-label-conditional-is-primary-strict-is-the-fallback--and-they-are-one-knob)
against a file-head argument; the component heads were not in that derivation.

**Decision: `f8 = 1` for the first trained model.** Cost: 0 % genuine whole-file audio — SONICS'
1,970 h and the 336 natural cell-5 rows are not drawn as wholes. ⚠️ AI songs (Suno/Udio) are a real
slice of the target distribution; the trade is a leak the audit can *measure* against an
under-representation it cannot. The option that recovers them is a **composition-free path for
whole-file rows on both sides** — which the theorem says does not exist while cells 6/7 are
composed-only.

### 4.3 The cell mix has the same shape of gap

`P(mixed | FILE_FAKE)` is balanced by C3 (gap 0.002). The masked heads see a different pattern:

| head | `P(other component present \| fake)` | `\| real)` | AUC of the pattern |
|---|--:|--:|--:|
| `voice_fake` — is music present? | 0.629 | 0.824 | **0.597** |
| `music_fake` — is voice present? | 0.620 | 0.824 | **0.602** |

Measured in the final stream as `voice_coverage` / `music_coverage` at 0.60 on the pooled component
heads (§12). In the symmetric case (`p1 = p3`, `p2 = p4`, `p6 = p7 = q`) the file-head balance fixes
`p1 = p2·p5 / (2q + p8)`, and the component-head residual is then
`−p2·q·(2q + p8 + p5) / (…)` (sympy) — zero only at **`q = 0`**. The mix cannot balance mixedness for
the file head and both component heads at once while cells 6/7 exist; the asymmetric search found
nothing either. A minimax mix keeps 6/7 and spreads the gap:

| mix | p1…p9 | gap file / music / voice | AUC | rates file · voice · music · v_pres · m_pres |
|---|---|---|--:|---|
| reference | .060 .130 .060 .135 .155 .125 .125 .095 .115 | 0.002 / 0.204 / 0.195 | 0.50 / **0.60** / **0.60** | .610 · .507 · .511 · .690 · .695 |
| minimax, `p6 = p7 = 0.10` | .030 .030 .030 .030 .323 .100 .100 .197 .160 | 0.025 / 0.025 / 0.025 | **0.51** on all three | .457 · .419 · .419 · .78 · .78 |

⚠️ The minimax mix buys balance with single-component cells at 3 % each and the presence heads at
their 0.78 ceiling. Whether 3 % voice-only-fake is enough *exposure* for a deepvoice test set is a
modelling question the audit cannot answer; the honest framing is a sweep between the two with the
per-head EER on VAL. **Recommendation: keep the reference mix for the first model and record the 0.60
presence-pattern cue on both component heads as a known, bounded residual** — it is at the gate, not
past it, and it is the only draw residual left after §3 and §4.2.

---

## 5 — Silence

### 5.1 The A6b correction

On the 500 LJSpeech utterances × 7 vocoders (`b1_signal.parquet`), the leading silence of the
vocoded copy **equals the original's in 99.2–100 % of pairs** (paired AUC 0.50 for every vocoder).
A6b's *"0.639 between LJSpeech and its own vocoded copies"* was computed on the S-tier draw of
`ljspeech` (2,000) vs `wavefake` (2,000) — and 1,000 of those WaveFake rows are the **JSUT**
subsets (Japanese, lead median 0.05–0.10 s) and 144 the Common-Voice prompts. Restricted to the
`ljspeech_*` subsets the AUC is **0.502** (n = 856). The vocoder does not change the silence; the
publisher's *other* sub-corpora do. Likewise within CFAD (0.717): the real half's sub-corpora have
lead medians from 0.00 (`thchs30`) to **0.95 s** (`magicread`), and the draw shares no utterance ids
across the halves, so the number is a sub-corpus artefact, not a synthesis one.

**What stands:** corpus-wide 0.595 and within-CFAD 0.652 on the drawn stream — real, and a *corpus*
artefact. **What falls:** L4 as a "synthesis artefact that survives pairing"; [EDA/01 A6b](../EDA/01-pool-a-real-voice.md#a6b---answered-2026-09-17-the-silence-shortcut-is-real-and-it-survives-pairing)
and [data_memo §5](../EDA/data_memo.md#5---the-leakage-hypothesis) L4 row need this correction.
The decision (augment, symmetric) is unchanged; its rationale is.

### 5.2 The grid

Effective leading silence = the file's own lead (if the onset is in the take) + the drawn lead:

| `p` | `lead ~ U(0, L)` | corpus-wide | LJSpeech ↔ WaveFake | within CFAD |
|---|---|--:|--:|--:|
| 0 (today) | — | 0.595 | 0.605 | 0.652 |
| 0.2 | 2 s | 0.577 | 0.570 | 0.619 |
| 0.5 | 4 s | 0.540 | 0.552 | 0.574 |
| 1.0 | 3 s | **0.553** | 0.508 | 0.599 |
| 1.0 | 4 s | 0.548 | 0.508 | 0.577 |
| trim + redraw | any | **0.500** | 0.509 | 0.509 |

Under P9 the file's onset is exposed in < 1 % of takes, so the *effective* lead is nearly all the
drawn one — in the final stream `voice_fake` on `voice_lead_silence_s` is 0.507. ⚠️ The A-A11
`p ≈ 0.2` default leaves 80 % of samples with the raw cue and barely moves the number.

**Decision: `silence_lead_s = 3.0`, `silence_tail_s = 1.0`, drawn for every sample** (the sampler
already draws them cell-independently — `sampler.py` draws `lead` before `wanted`). Trimming stays
rejected (A5b asymmetry), and with P9 the inside-crop already removes the file's own lead from
> 99 % of samples, which is what makes the residual 0.55 rather than 0.60.

---

## 6 — Level

**B1 pair test** (same utterance, chain plane; paired AUC = fraction of pairs where fake > real):

| vocoder | RMS | peak | crest | band 0–1 kHz | \|DC\| | lead | tail | clipping |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| full_band_melgan | **1.000** (−0.32 dB) | 0.798 | 0.634 | 0.972 | 0.598 | 0.501 | 0.582 | 0.502 |
| hifiGAN | 0.998 (−0.69) | 0.798 | 0.576 | 0.588 | **0.982** | 0.500 | 0.532 | 0.502 |
| melgan | 1.000 (−0.25) | 0.730 | 0.572 | 0.996 | 0.596 | 0.500 | 0.586 | 0.502 |
| melgan_large | 1.000 (−0.27) | 0.804 | 0.632 | 0.984 | 0.920 | 0.501 | 0.586 | 0.502 |
| multi_band_melgan | 1.000 (−1.01) | 0.878 | 0.568 | 0.876 | 0.664 | 0.502 | 0.589 | 0.502 |
| parallel_wavegan | 1.000 (−1.24) | 0.984 | 0.738 | 0.958 | 0.992 | 0.503 | 0.631 | 0.502 |
| waveglow | 1.000 (−0.39) | 0.690 | 0.572 | 0.918 | 0.890 | 0.501 | 0.579 | 0.501 |

Every vocoder outputs the same utterance **0.25–1.24 dB quieter**, consistently (paired AUC 1.000)
but by a margin that is a fraction of the 22.6 dB between-source spread. The sub-1 kHz *share* moves
by +0.002…+0.024 — consistent, tiny. Lead silence and clipping do not move at all.

**Candidate normalisations, simulated on the drawn stream** (univariate / source-grouped):

| candidate | `music_fake` | `voice_fake` |
|---|--:|--:|
| none (today) | 0.717 / **0.683** | 0.762 / 0.586 |
| peak- or RMS-normalise (residual = crest) | 0.571 / **0.453** | 0.715 / **0.786** |
| none + gain jitter U(−6, 6) | 0.673 / 0.641 | 0.736 / 0.585 |
| none + gain jitter U(−12, 12) | 0.621 / **0.599** | 0.692 / **0.579** |

Normalisation trades the heads: it removes the music level fingerprint and *creates* a voice one
(crest survives the archive holdout at 0.786 — S2 confirms: `crest_factor_db` is the one scalar whose
A-vs-B AUC *rises* under grouping, 0.612 → 0.666). Jitter lowers both without restructuring.

**Decision: no normalisation stage (P-A1 off); A-A7 gain jitter `U(−12, 12)` dB at `p = 1`,
label-blind.** ⚠️ Wider than ★ `[BC2026]`'s ±6; the harness says ±6 leaves music at 0.64. Revisit
if X4 ever shows the organizers normalised.

---

## 7 — DC and the high-pass

| feature | A vs B ungrouped | A vs B `source_name`-grouped | within CFAD |
|---|--:|--:|--:|
| `\|dc_offset\|` | **0.806** | **0.392** | 0.530 |
| `band_energy_0` (0–1 kHz share) | 0.541 | 0.455 | 0.506 |
| `crest_factor_db` | 0.612 | 0.666 | 0.529 |
| `rms_dbfs` | 0.599 | 0.485 | 0.557 |

Per CFAD generator vs `cfad-real` on `|dc|`: `gl` **0.994**, `pwg` 0.923, `stylegan` 0.812,
`fasthifigan` 0.765, `world` 0.751, `straight` 0.738, `tacohifigan` 0.725; the other four 0.54–0.61.
DC is a **generator id** (some vocoders leave one) that **inverts under an archive holdout** — the
model would learn it in CV and it would not transfer. It is also the top acoustic residue on
`voice_fake` in every stream (0.82).

High-pass: fraction of each vocoder's real−fake LTAS difference energy **above** a cutoff (first mel
band centres at 13.8 / 27.9 / 42.2 / 56.9 Hz):

| cutoff | full_band_melgan | hifiGAN | melgan | melgan_large | multi_band_melgan | parallel_wavegan | waveglow |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 20 Hz | 0.636 | 0.929 | 0.666 | 0.944 | 0.965 | 0.750 | **0.462** |
| 40 Hz | 0.584 | 0.899 | 0.504 | 0.715 | 0.964 | 0.669 | **0.251** |
| 72 Hz | 0.543 | 0.889 | 0.413 | 0.457 | 0.936 | 0.565 | 0.237 |

**Decision: register DC removal as a symmetric preprocess (P-S4); `band_hz` lower edge stays 0.**
A 40 Hz rumble filter would discard 25–75 % of the paired difference for four of seven vocoders.
DC (0 Hz) is a per-file constant, not a band, and removing it costs an id that does not transfer.

---

## 8 — Bandwidth, the resampler shelf, and codec

| source | native codec / rate | chain bandwidth (Hz) | `near_nyquist_ratio_chain` |
|---|---|--:|--:|
| wavefake (B) | wav 22.05 k | 8000 | **0.293** |
| ljspeech (A) | wav 22.05 k | 8000 | 0.219 |
| rirs (E) · compspoof (E) | wav 16 k | 8000 | 0.213 · 0.180 |
| fma (C) | mp3 265 kbps 44.1 k | 8000 | **0.157** |
| mlaad (B) | wav 22.05 k | 8000 | 0.116 |
| fakemusiccaps (D) | wav 16 k | 8000 | **0.080** |
| cfad-real / cfad-fake · musan-* · zeroth | wav 16 k | 7594–7797 | 0.056–0.088 · 0.013 |
| **sonics (cell 8)** | **mp3 36.8 kbps** 16 k | **7297** | **0.000** |

Two fingerprints the chain leaves: sources resampled *by us* carry `resample_poly`'s transition band
(0.12–0.29) where native-16 kHz sources carry none (≤ 0.09) — and pool C is resampled while pool D
is native; and SONICS is low-bitrate MP3 with a 7.3 kHz rolloff. `music_near_nyquist` is the top
acoustic residue on `music_fake` (0.744 ungrouped, 0.80 in the mixed stratum); `min_bandwidth_hz`
0.63 on both presence heads.

**Decision: a symmetric upper band edge, `band_hz = [0, 7200]`, as the first candidate**, with the
A-S3 codec round-trip (`mp3 {64, 96, 128}` at `p = 0.5`, the same menu for every cell) as the second.
The band edge is one config value and removes both fingerprints at once; it costs 0.8 kHz of a band
that ten of fourteen sources do not populate distinctively anyway. Measure the residual under each
in the next harness run.

---

## 9 — The vectors

Logistic regression on the 128-band chain LTAS (+ mel skew and kurtosis, 384), ungrouped / grouped:

| head | `ltas[128]` | `ltas+skew+kurt[384]` | sources |
|---|--:|--:|--:|
| `music_fake` | 0.967 / 0.701 † | 0.976 / 0.348 † | 4 |
| `voice_fake` | 0.924 / 0.536 | 0.973 / 0.630 | 8 |
| `music_present` | 0.975 / 0.894 | 0.982 / 0.915 | 14 |
| `voice_present` | 0.989 / 0.816 | 0.994 / 0.849 | 14 |

† 2-fold archive holdout (one real + one fake source per side); the weaker measurement.

The presence heads are legitimately spectral. The fake heads' 0.92–0.98 collapses to 0.35–0.70 under
an archive holdout — the same shape as X1d, on 45 M more numbers. **Per-source one-vs-rest AUC on
LTAS within its pool is ≥ 0.980 for all twelve sources** (`ljspeech`, `fma`, `musan-music`,
`zeroth-korean` 1.000; `wavefake` 0.999). Every archive is nameable; `aug_strength` cannot be
graded by identifiability and stays uniform. This is also E-A2 (adversarial validation) between
sources within a label: AUC ≈ 1 everywhere, so VG3 will fail on any source-disjoint fold until the
symmetric chain (§6–§8) has been applied — the number to re-measure after it.

---

## 10 — Families and domains

**WaveFake**, average-linkage clustering of `b1_family_correlation_{native,chain}` at every cut from
`r ≥ 0.4` to `r ≥ 0.6`, both planes agree: **{full_band_melgan, hifiGAN, parallel_wavegan, waveglow}
· {melgan, melgan_large} · {multi_band_melgan}** — three `artifact_family` values for seven names.

**Music families**: 5 FakeMusicCaps domains (15.3–15.7 h each) + 5 SONICS domains (`chirp-v3.5`
**1,057 h**, `udio-120s` 683, `chirp-v3` 140, `chirp-v2-xxl-alpha` 46, `udio-30s` 45). By the
codec > vocoder > backbone priority the three Chirp versions and the two Udio lengths are one family
each → **7**, under the ≥ 8 floor; under `f8 = 1` (§4) SONICS is not drawn at all and the count is
**5** — leave-one-family-out with the variance caveat, as [validation/01 §3](../validation/01-split-scheme.md#-the-music-head-cannot-support-the-planned-split) option 2.

**`domain_cap` does not reach whole-file rows.** `_whole_by_cell` draws uniformly
(`sampler.py:396`): the baseline stream drew 354–381 samples per SONICS generator from 400 rows
each — proportional to row count, not capped. 54 of 11,328 domains exceed 500 files; every WaveFake
vocoder (13,100) and every FakeMusicCaps generator (5,521) does, and the weight handles those.

**Korean**: `zeroth-korean` 22,720 files / 52.9 h real vs MLAAD `ko` 359 files / 0.92 h fake — a
57× real/fake language imbalance that no split key sees.

---

## 11 — Manifest actions, in hours

| action | rows | hours | of its pool |
|---|--:|--:|--:|
| `reassign_cell` D → 8 | 1,006 | 2.83 | 3.7 % of D |
| `reassign_cell` C → 5 (`fma` 274 + `musan-music` 62) | 336 | 6.08 | 5.6 % of C's hours (12.6 % of measured files) |
| `restrict_noise` | 1,032 | 1.15 | 5.3 % of E (7.3 % of files) — **G4 fires** on the file count |
| `degenerate` | 23 | 0.09 | — |

VAD coverage is 100 % for D, E and MUSAN; **25 % for `fma`**, 15 % ljspeech, 12.5 % mlaad, 9 %
zeroth, 5 % cfad-real, 4 % sonics, 3 % cfad-fake, 1.7 % wavefake. The C → 5 action needs the
`fma` pass extended to its remaining 6,000 files before it is applied at scale.

**The 4 s floor (H3)**, census scale: pool A 17,023 files (22.7 %) = **13.3 h (6.8 %)**; pool B
85,919 files (41.4 %) = **68.3 h (22.5 %)**. A 2 s *component* floor would recover 12.3 h of A and
63.5 h of B. Under P9 every component is cropped to 3–8 s and tiled, so a 2 s file is usable if
tiled — the floor becomes a policy on the take, and the asymmetry (22.5 % vs 6.8 % of hours) goes
into G-EDA7's ledger either way.

---

## 12 — The final candidate, audited as one

`M1` (reassigned manifest) + `f8 = 1` + P9 + `lead ~ U(0, 3)` on every sample. ⚠️ Under `f8 = 1`
the 1,342 reassigned rows are never drawn as wholes — M1's only live effect here is the 1,055 drops.
Draw features ungrouped / grouped, acoustic residues ungrouped / grouped:

| head | stratum | draw | acoustic | what is left |
|---|---|--:|--:|---|
| `music_fake` | pooled | **0.597** / — | 0.807 / — | draw: `voice_coverage` 0.603 (§4.3); acoustic: `music_rms_dbfs` 0.742, `music_near_nyquist` 0.643, `music_dc_offset` 0.582 |
| `music_fake` | mixed | **0.495** | 0.808 | — |
| `music_fake` | music-only | **0.497** | 0.802 | acoustic only |
| `voice_fake` | pooled | 0.609 / 0.646 | 0.799 / 0.658 | draw: `music_coverage` 0.596 (§4.3); acoustic: `voice_dc_offset` 0.817, `voice_bandwidth_hz` 0.706, `voice_rms_dbfs` 0.665 |
| `voice_fake` | mixed / voice-only | 0.528 / 0.532 | 0.786 / 0.807 | — |
| `file_fake` | pooled / mixed | **0.512 / 0.505** | 0.752 / 0.747 | acoustic only |
| `music_present` | pooled | 0.590 / 0.611 | 0.709 / 0.658 | `sequential` 0.590 (two components ⇒ both present: a property of the label space); `mean_rms_dbfs` 0.689 |
| `voice_present` | pooled | 0.587 / 0.585 | 0.425 | `sequential` 0.591 |

`training.audit.run_audit`: **13 of 13 pass** (I2 0.000, I2c 0.000, I8 file .606 / voice .501 /
music .511 / presence .688 / .694). `P(offset = 0)` is 0.001 (C) / 0.002 (D).

**Every draw feature is inside the gate on every head and stratum except the two 0.60
presence-pattern residues §4.3 derives from the cell mix.** The acoustic residues are what the
shipped preprocess must now neutralise: level (§6), DC (§7), the shelf and bandwidth (§8) — and the
next harness run is *after* those three are registered, because the harness can only score the
effective features it is given a formula for.

---

## 13 — Decision table (D1)

| # | finding | registry | knob | value | before → after (draw AUC, `music_fake` music-only unless stated) | gate |
|---|---|---|---|---|---|---|
| 1 | H1/H2/L1/L2 | draw | component take rule | `take ~ U(3, 8)` s, `take ≤ file − 0.5`, offset `U(0, file − take)`, every role | 0.993 → **0.497** | — (a sampler change; I1b re-run) |
| 2 | H2 | render | fill | tile the take to its span with A-A4 sigmoid joins, every role | `uncovered_frac` 0.848 → 0 on `music_present` | I1b on `*_joins` |
| 3 | composedness on the component heads | draw | `f8` | **1.0** | `music_fake` mixed 0.734 → **0.509** | I2 + this audit |
| 4 | mixedness on the component heads | draw | `cell_mix` | reference kept; 0.60 residual recorded; minimax mix `{.03,.03,.03,.03,.323,.10,.10,.197,.16}` as the sweep's other end | 0.60 → 0.51 (minimax) | G6 |
| 5 | L4 (corrected: corpus artefact) | draw | `silence_lead_s` / `silence_tail_s` | **3.0** / 1.0 (⚠️ tail *not measured* — the grid covered lead only; 1.0 is a symmetric placeholder), every sample | 0.595 → **0.553** corpus-wide | I1b re-run |
| 6 | level | augment | A-A7 gain jitter | `U(−12, 12)` dB, `p = 1` | music 0.683 → 0.599, voice 0.586 → 0.579 (grouped) | G6 |
| 7 | level | preprocess | P-A1 | **off** | (on would raise voice to 0.786) | G6 |
| 8 | DC | preprocess | P-S4 DC removal | **on** | `voice_dc_offset` 0.82 ungrouped / inverts grouped | G6, I14 |
| 9 | sub-72 Hz | preprocess | `band_hz[0]` | **0** (no high-pass) | keeps 25–75 % of 4 vocoders' difference energy | — |
| 10 | shelf + SONICS rolloff | preprocess | `band_hz[1]` | **7200** (candidate 1); A-S3 mp3 `{64, 96, 128}` `p = 0.5` (candidate 2) | `music_near_nyquist` 0.744 → to measure | G6, I1b on `normalize` |
| 11 | G-EDA6 | manifest | `row_kind` / `cell` / drops | 1,342 reassigned, 1,055 dropped; `fma` VAD to 100 % first | I2 FAIL → PASS | G4, G7 |
| 12 | families | manifest | `artifact_family` | WaveFake → 3 families; music 5 (strict) | — | review gate |
| 13 | DOSS | draw | whole-file weighting | apply `_doss_weights` to `_whole_by_cell` (moot under `f8 = 1`, needed the day `f8 < 1`) | — | I21 |
| 13b | whole-file branch | draw | `sampler.py:394-410` | the whole-file branch draws **no lead/tail** and sets `duration_s = min(timeline, file)` — under `f8 < 1` both are composedness cues (a 10 s reassigned FMC row would cap its own timeline at 10 s). Moot under `f8 = 1`; fix before any `f8 < 1` run | — | I1b |
| 14 | the 4 s floor | draw | component floor | 2 s components under tiling; timeline floor stays 4 s | recovers 63.5 h of B | G-EDA7 ledger |
| 15 | `aug_strength` | manifest | — | uniform 1.0 | every source identifiable ≥ 0.98 | — |

---

## 14 — What this run does not prove

* **Simulation, not the sampler.** Policies were re-drawn on the stream; `training/sampler.py`
  still implements P0. Items 1, 2, 5, 13 are code changes, and the acceptance run is the harness over
  the *changed* sampler.
* **The harness sees the features it is given.** The onset deficit, level, DC, bandwidth and
  near-Nyquist scalars are the acoustic residues it knows; a per-generator phase artefact or a
  mastering chain passes cleanly ([EDA/07 §4](../EDA/07-order-and-gates.md#4---known-limits-of-this-plan)).
* **Grouped AUCs on the music head are 2-fold** over four sources, and the voice head's seven
  sources give high-variance folds (the voice-only stratum reads 0.79 baseline, 0.65 final, 0.40 in
  X1d on similar features). Treat grouped numbers as direction, ungrouped draw numbers as the gate.
* **Whole-file rows in the mixed cells are one file playing both roles**; their "coverage" and
  "take" are the same number twice, which is correct and is why cell 8 under `f8 = 0` reads as
  `composed` rather than as a length.
* **S-tier coverage** is 2,000 rows per source for A, B, C and cell 8; §11's hour figures for
  `reassign_cell` C → 5 are the measured rows only.
* **`run_audit` I7** (eval size floors) is skipped on a training stream by design.

---

*Numbers are from `eda/out/_strategy/` as written on 2026-09-19 (`harness.log`, `scalar.log`);
where this page and an artifact disagree, the artifact wins.*
