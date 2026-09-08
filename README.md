# DACON 236749 — Deep Voice Detection

Research and strategy for **딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회**
([competition page](https://dacon.io/competitions/official/236749/overview/description)).

Hosted by 행정안전부 and 한국지능정보사회진흥원, supervised by 국립과학수사연구원, run by DACON.
₩42,000,000 prize pool · leaderboard closes **2026-09-29**.

> **Status: research, planning, the measurement pipeline, the model and the training pipeline are
> complete. Nothing has been trained — there is no corpus yet.**
> Code ships in **[`metrics/`](metrics/AGENTS.md)**, **[`models/`](models/AGENTS.md)** and
> **[`training/`](training/AGENTS.md)** — 844 tests, every invariant paired with a mutation that
> was observed to fail. Current state and next actions: **[PROGRESS.md](PROGRESS.md)**

---

## The task

For each audio file, predict five probabilities:

| Output | Meaning | Metric weight |
|---|---|---|
| `FILE_FAKE_PROB` | the file is AI-generated | 0.45 |
| `MUSIC_FAKE_PROB` | the **music** component is AI-generated | 0.27 |
| `VOICE_FAKE_PROB` | the **voice** component is AI-generated | 0.18 |
| `VOICE_PRESENT_PROB` | voice is present | 0.05 |
| `MUSIC_PRESENT_PROB` | music is present | 0.05 |

`Score = 0.9 × ADS + 0.1 × CPS`, where ADS is built from **EER** and CPS from **ROC-AUC** — both
pure ranking metrics. Test set: 1,200 files, 4–60 s, **16 kHz**, mono and stereo, MP3/WAV/FLAC,
some telephone-channel. **No training data is provided.** Submission is a code bundle that runs
offline on one L4 GPU within 60 minutes.

## Documentation

| Directory | Contents |
|---|---|
| **[docs/competition/](docs/competition/README.md)** | Rules, metric derivation, submission contract, all official Q&A |
| **[docs/survey/](docs/survey/README.md)** | Synthesized prior art — SOTA per head, model/dataset catalogs |
| **[docs/kaggle/](docs/kaggle/README.md)** | Winners' practical recipes, incl. code pulled from top notebooks |
| **[docs/data/](docs/data/README.md)** | Data strategy — tiered method catalogs, review gates, ~70 vetted sources |
| **[docs/validation/](docs/validation/README.md)** | How we measure — splits, metric harness, decision protocol, gates, LB decomposition |
| **[docs/architecture/](docs/architecture/README.md)** | How we build the model — design envelope, pretrained candidates, ensembling, runtime budget |
| **[docs/pipelines/](docs/pipelines/README.md)** | The data pipeline — sample contract, sampler, transforms, collation, invariants |
| **[docs/training/](docs/training/README.md)** | The objective — what the metric demands, the loss, what was ruled out, the schedule, the tier list |
| **[metrics/](metrics/AGENTS.md)** | Scoring, fold aggregation, diagnostics, submissions |
| **[models/](models/AGENTS.md)** | The architecture — trunk, SED heads, frontends, losses, outputs |
| **[training/](training/AGENTS.md)** | The training pipeline — specs, sampler, folds, render, collate, stages, checkpoints, loop, gates |
| **[docs/papers/](docs/papers/INDEX.md)** | ~75 papers indexed with links; 12 read in depth |

Start at **[docs/README.md](docs/README.md)** for a reading order.

Confidence marks used throughout: ★ verified from a primary source · ☆ secondary, re-verify ·
⚠️ risk · 🔴 decision-changing · ❌ forbidden by competition rules ·
🔷 our own inference, untested (used in [docs/architecture/](docs/architecture/README.md)).
The Python sources under [`training/`](training/AGENTS.md) spell these out as words —
`Critical:`, `Caveat:`, `Strong evidence:` — rather than drawing them.

## Findings that shaped the plan

- **The music head is worth more than the voice head** (0.27 vs 0.18), has the thinnest literature,
  and is the more tractable of the two. It is the highest-EV target.
- **16 kHz standardization destroys the artifacts most published AI-music detectors rely on** —
  bitrate, 48 kHz output, upsampling traces. But **ranking metrics degrade far less than
  thresholded ones**: in one broadcast study F1 fell 0.992 → 0.186 while AUC only fell 0.998 → 0.775,
  and we are scored on ranking.
- **Never use source separation as a frozen preprocessor.** Naive separate-then-detect measurably
  hurts; either skip separation or train it jointly end-to-end.
- **Domain balance beats data volume** — one study reached a better EER with 3% of the data by
  capping per-generator domains. Our license-constrained pool is not the bottleneck.
- **Pseudo-labeling the test set is forbidden** by competition rule 2.3, which rules out the single
  most repeated technique in comparable Kaggle audio competitions.
- **A green test suite is not evidence that a check can fail.** An adversarial mutation pass over
  the training loop left **28 of 85 mutants alive** under a suite of 101 passing tests, and four
  production defects were found that way. Every serious defect in this repo was found by a review
  pass, a hand-run mutation, or someone asking whether a property was actually tested — none by a
  test going red on its own. Invariants here are paired with the mutation that was observed to
  break them; see [docs/pipelines/05](docs/pipelines/05-invariants.md).

## Setup

```bash
cp .env.example .env          # then fill in KAGGLE_API_TOKEN (and HF_TOKEN if needed)
python3 -m venv .venv && .venv/bin/pip install kaggle
```

`.env` is gitignored. The Kaggle API is used for dataset discovery and pulling public notebooks;
it is not required to read the documentation.

## Verification

```bash
.venv/bin/python -m pytest -o addopts=""    # 844 tests; -o addopts="" keeps pytest's summary line
.venv/bin/python scripts/check_links.py     # every cross-reference in 87 docs
```

`scripts/verify_metric.py` reproduces every quantitative claim in
[docs/validation/](docs/validation/README.md) — that an all-constant submission scores exactly
0.5000, that the leaderboard decomposes exactly into per-head metrics, and the EER noise /
saturation tables.

```bash
.venv/bin/pip install scikit-learn scipy
.venv/bin/python scripts/verify_metric.py
```

## Skill

`.claude/skills/dacon-talkboard-sync/` fetches the competition talkboard, diffs it against
`docs/competition/talkboard-snapshot.json`, and updates the Q&A doc. DACON is a Nuxt SSR app that
serves an empty shell to non-browser clients, so the skill parses the embedded page state directly.

```bash
python3 .claude/skills/dacon-talkboard-sync/scripts/fetch_talkboard.py \
  --out /tmp/talkboard --snapshot docs/competition/talkboard-snapshot.json
```

## Note on sources

Dataset licences recorded here are **reported from secondary sources and unverified**. The
competition requires submitting the actual training files, and forbids data whose licence
restricts third-party provision — so every source must be verified at origin before use. See
[docs/data/01-rules-check.md](docs/data/01-rules-check.md).
